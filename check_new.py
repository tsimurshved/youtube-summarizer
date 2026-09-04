#!/usr/bin/env python3
"""Check subscribed channels for NEW long-form uploads, transcribe, summarize.

First run for a newly added channel SEEDS seen_videos with the current
latest N (playlist-end 8) uploads WITHOUT summarizing them. Only videos
published after that seed will trigger summaries on later runs.

Skip YouTube Shorts, live streams, and premieres (duration < 60s,
/shorts/ URL, live_status, is_live, was_live).

Usage:
  python check_new.py
  python check_new.py --backfill 1
  python check_new.py --backfill 1 --channel NetworkChuck
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from common import (
    PLAYLIST_END,
    SLEEP_BETWEEN_CHANNELS,
    SUMMARIES_DIR,
    TRANSCRIPTS_DIR,
    YtdlpError,
    channel_matches,
    download_transcript,
    ensure_dirs,
    is_transient_error,
    list_recent_uploads,
    load_state,
    log,
    now_iso,
    published_iso,
    save_state,
    skip_reason,
    video_url_of,
)
from summarize import summarize_file, write_summary_files


def mark_seen(state: dict, video_id: str, *, channel_id: str, title: str, url: str,
              published: str, error: str | None = None, skipped: str | None = None,
              retry_count: int = 0) -> None:
    prev = state["seen_videos"].get(video_id) or {}
    rec = {
        "channel_id": channel_id,
        "title": title,
        "url": url,
        "published": published,
        "seen_at": now_iso(),
    }
    if error:
        rec["error"] = error
        rec["retry_count"] = retry_count
    elif prev.get("error") and not error:
        pass
    if skipped:
        rec["skipped"] = skipped
    state["seen_videos"][video_id] = rec


def should_retry(seen: dict | None) -> bool:
    if not seen:
        return True
    err = seen.get("error")
    if not err:
        return False
    if not is_transient_error(str(err)):
        return False
    return int(seen.get("retry_count") or 0) < 1


# write_summary_files lives in summarize.py (overview / takeaways / useful prose).


def process_video(state: dict, ch: dict, entry: dict, video_id: str) -> dict:
    """Download captions + summarize. Always marks seen. Returns report row."""
    title = entry.get("title") or video_id
    url = video_url_of(entry, video_id)
    published = published_iso(entry)
    channel_name = ch.get("name") or ch.get("id")
    row = {
        "channel": channel_name,
        "title": title,
        "url": url,
        "published": published,
        "overview": "",
        "takeaways": "",
        "useful": "",
    }
    prev = state["seen_videos"].get(video_id) or {}
    retry_count = int(prev.get("retry_count") or 0)
    try:
        log(f"    captions {video_id}")
        srt = download_transcript(video_id, url)
        log(f"    summarize {srt.name}")
        summary = summarize_file(srt, title=title)
        write_summary_files(
            video_id,
            channel=channel_name,
            title=title,
            url=url,
            published=published,
            summary=summary,
        )
        row["overview"] = summary.get("overview") or ""
        row["takeaways"] = summary.get("takeaways") or ""
        row["useful"] = summary.get("useful") or ""
        mark_seen(
            state,
            video_id,
            channel_id=ch["id"],
            title=title,
            url=url,
            published=published,
        )
        # drop leftover error keys by rewrite above
    except (YtdlpError, ValueError, OSError) as exc:
        err = str(exc)
        transient = is_transient_error(err)
        # retry_count = retries already used (0 or 1). First fail stores 0 so
        # the next run may retry once; after that (or on permanent errors) store 1.
        if transient:
            stored_retry = 1 if prev.get("error") else 0
        else:
            stored_retry = 1
        log(f"    ERROR {video_id}: {err}")
        row["error"] = err
        mark_seen(
            state,
            video_id,
            channel_id=ch["id"],
            title=title,
            url=url,
            published=published,
            error=err,
            retry_count=stored_retry,
        )
    save_state(state)
    return row


def channel_is_unseeded(state: dict, channel_id: str) -> bool:
    """First run: no seen_videos belong to this channel yet.

    Seed the current latest N as seen WITHOUT summarizing, so we do not dump
    the creator's entire backlog as 'new'. Only future uploads trigger work.
    """
    for rec in state["seen_videos"].values():
        if rec.get("channel_id") == channel_id:
            return False
    return True


def eligible_entries(entries: list[dict], include_shorts: bool) -> tuple[list[tuple[str, dict]], list[tuple[str, dict, str]]]:
    keep = []
    skipped = []
    for e in entries:
        vid = str(e.get("id") or "")
        if not vid:
            continue
        reason = skip_reason(e, include_shorts=include_shorts)
        if reason:
            skipped.append((vid, e, reason))
        else:
            keep.append((vid, e))
    return keep, skipped


def run_channel(state: dict, ch: dict, *, backfill: int, include_shorts: bool) -> list[dict]:
    reports: list[dict] = []
    channel_id = ch["id"]
    log(f"channel {ch.get('name') or channel_id} ({channel_id})")
    try:
        entries = list_recent_uploads(channel_id, limit=PLAYLIST_END)
    except YtdlpError as exc:
        log(f"  list failed: {exc}")
        return [{"channel": ch.get("name") or channel_id, "title": "", "url": ch.get("url") or "",
                 "published": "", "overview": "", "takeaways": "", "useful": "",
                 "error": f"list uploads: {exc}"}]
    except Exception as exc:
        log(f"  list failed: {exc}")
        return [{"channel": ch.get("name") or channel_id, "title": "", "url": ch.get("url") or "",
                 "published": "", "overview": "", "takeaways": "", "useful": "",
                 "error": f"list uploads: {exc}"}]

    keep, skipped = eligible_entries(entries, include_shorts)
    # FIRST-RUN must be decided before we write skipped videos into seen_videos.
    unseeded = channel_is_unseeded(state, channel_id)
    # Always record skipped items as seen so they never fire later.
    for vid, e, reason in skipped:
        if vid not in state["seen_videos"]:
            mark_seen(
                state,
                vid,
                channel_id=channel_id,
                title=e.get("title") or vid,
                url=video_url_of(e, vid),
                published=published_iso(e),
                skipped=reason,
            )
    save_state(state)
    to_summarize: list[tuple[str, dict]] = []

    if unseeded:
        # FIRST-RUN SEED: mark current latest eligible (and the raw latest N)
        # as seen without summarizing, so only FUTURE uploads trigger work.
        log(f"  first-run seed ({len(keep)} eligible / {len(entries)} listed, backfill={backfill})")
        seed_targets = keep  # already capped by playlist-end
        summarize_ids = {vid for vid, _ in seed_targets[: max(0, backfill)]}
        for vid, e in seed_targets:
            if vid in summarize_ids:
                to_summarize.append((vid, e))
            else:
                mark_seen(
                    state,
                    vid,
                    channel_id=channel_id,
                    title=e.get("title") or vid,
                    url=video_url_of(e, vid),
                    published=published_iso(e),
                )
        # also seed ineligible listed videos (already marked skipped above)
        for e in entries:
            vid = str(e.get("id") or "")
            if vid and vid not in state["seen_videos"] and vid not in summarize_ids:
                mark_seen(
                    state,
                    vid,
                    channel_id=channel_id,
                    title=e.get("title") or vid,
                    url=video_url_of(e, vid),
                    published=published_iso(e),
                    skipped=skip_reason(e, include_shorts=include_shorts) or "seed",
                )
        save_state(state)
    else:
        for vid, e in keep:
            seen = state["seen_videos"].get(vid)
            if not seen:
                to_summarize.append((vid, e))
            elif should_retry(seen):
                log(f"  retry {vid} (retry_count={seen.get('retry_count', 0)})")
                to_summarize.append((vid, e))
        # --backfill on an already-seeded channel: summarize latest N even if seen,
        # used only for a one-off smoke test. Still leaves them marked seen.
        if backfill and not to_summarize:
            for vid, e in keep[:backfill]:
                # skip if a successful summary already exists
                if (SUMMARIES_DIR / f"{vid}.json").exists() and not (state["seen_videos"].get(vid) or {}).get("error"):
                    continue
                to_summarize.append((vid, e))
                if len(to_summarize) >= backfill:
                    break

    for vid, e in to_summarize:
        reports.append(process_video(state, ch, e, vid))
        time.sleep(0.8)
    return reports


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Check channels for new long-form uploads")
    p.add_argument("--backfill", type=int, default=0, metavar="N",
                   help="On first-run (or smoke test), summarize the latest N eligible videos instead of only seeding. Default 0.")
    p.add_argument("--channel", action="append", default=[],
                   help="Only process matching channel(s) (name, @handle, UC id, URL substring). Repeatable.")
    p.add_argument("--include-shorts", action="store_true",
                   help="Do not skip Shorts / sub-60s videos (lives and premieres still skipped).")
    args = p.parse_args(argv)

    ensure_dirs()
    state = load_state()
    channels = state.get("channels") or []
    if not channels:
        log("channels.json has no channels. Add some with: python add_channels.py URL [URL...]")
        print("[]")
        return 0

    selected = channels
    if args.channel:
        selected = [c for c in channels if any(channel_matches(c, n) for n in args.channel)]
        if not selected:
            log(f"no channel matched: {args.channel}")
            print("[]")
            return 2

    reports: list[dict] = []
    for i, ch in enumerate(selected):
        if not ch.get("id"):
            continue
        reports.extend(
            run_channel(
                state,
                ch,
                backfill=max(0, args.backfill),
                include_shorts=args.include_shorts,
            )
        )
        save_state(state)
        if i < len(selected) - 1:
            time.sleep(SLEEP_BETWEEN_CHANNELS)

    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
