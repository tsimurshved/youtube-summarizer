#!/usr/bin/env python3
"""Add YouTube channels to channels.json (deduped).

Accepts channel URLs, @handles, /channel/UC..., or a video URL
(resolved to its channel via yt-dlp).

Usage:
  python add_channels.py URL [URL...]
  python add_channels.py --file channel_urls.txt
"""

from __future__ import annotations

import argparse
import sys
import time

from common import (
    YtdlpError,
    load_state,
    log,
    now_iso,
    resolve_channel,
    save_state,
)


def collect_urls(args: argparse.Namespace) -> list[str]:
    urls: list[str] = []
    if args.file:
        text = args.file.read_text(encoding="utf-8") if hasattr(args.file, "read_text") else None
        if text is None:
            from pathlib import Path
            text = Path(args.file).read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line.split()[0])
    urls.extend(args.urls or [])
    # de-dupe inputs preserving order
    seen = set()
    out = []
    for u in urls:
        key = u.strip().rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(u.strip())
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Add YouTube channels to channels.json")
    p.add_argument("urls", nargs="*", help="Channel URLs, @handles, UC ids, or video URLs")
    p.add_argument("--file", "-f", dest="file_path", help="Text file with one URL per line")
    args = p.parse_args(argv)

    from pathlib import Path

    class NS:
        pass

    ns = NS()
    ns.file = Path(args.file_path) if args.file_path else None
    ns.urls = args.urls
    if ns.file and not ns.file.exists():
        log(f"file not found: {ns.file}")
        return 2
    urls = collect_urls(ns)
    if not urls:
        log("No URLs given. Usage: python add_channels.py URL [URL...]")
        return 2

    state = load_state()
    by_id = {c.get("id"): i for i, c in enumerate(state["channels"]) if c.get("id")}
    added, updated, skipped, failed = [], [], [], []

    for i, raw in enumerate(urls, 1):
        log(f"[{i}/{len(urls)}] resolve {raw}")
        try:
            info = resolve_channel(raw)
        except YtdlpError as exc:
            log(f"  FAIL: {exc}")
            failed.append({"url": raw, "error": str(exc)})
            time.sleep(1.0)
            continue
        except Exception as exc:
            log(f"  FAIL: {exc}")
            failed.append({"url": raw, "error": str(exc)})
            time.sleep(1.0)
            continue

        ch_id = info["id"]
        if ch_id in by_id:
            idx = by_id[ch_id]
            existing = state["channels"][idx]
            if info["name"] and existing.get("name") != info["name"]:
                existing["name"] = info["name"]
                updated.append(info)
                log(f"  already present, name updated: {info['name']} ({ch_id})")
            else:
                skipped.append(info)
                log(f"  already present: {existing.get('name') or ch_id}")
        else:
            rec = {
                "id": ch_id,
                "name": info["name"],
                "url": info["url"],
                "added": now_iso(),
            }
            state["channels"].append(rec)
            by_id[ch_id] = len(state["channels"]) - 1
            added.append(rec)
            log(f"  added {info['name']} ({ch_id})")
        save_state(state)
        if i < len(urls):
            time.sleep(0.9)

    log(
        f"done: added={len(added)} updated={len(updated)} "
        f"already={len(skipped)} failed={len(failed)} "
        f"total_channels={len(state['channels'])}"
    )
    if failed:
        for f in failed:
            log(f"  failed {f['url']}: {f['error']}")
        return 1 if not added and not skipped and not updated else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
