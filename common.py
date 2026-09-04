"""Shared helpers for the YouTube channel monitor (no network UI)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
CHANNELS_FILE = ROOT / "channels.json"
TRANSCRIPTS_DIR = ROOT / "transcripts"
SUMMARIES_DIR = ROOT / "summaries"
PLAYLIST_END = 8
SLEEP_BETWEEN_CHANNELS = 1.6
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
CHANNEL_ID_RE = re.compile(r"^UC[A-Za-z0-9_-]{22}$")

SKIP_LIVE_STATUS = {"is_live", "is_upcoming", "was_live", "post_live"}


class YtdlpError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ytdlp_bin() -> str:
    venv = ROOT / ".venv" / "bin" / "yt-dlp"
    if venv.exists():
        return str(venv)
    return "yt-dlp"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def empty_state() -> dict:
    return {"channels": [], "seen_videos": {}}


def load_state() -> dict:
    if not CHANNELS_FILE.exists():
        return empty_state()
    try:
        data = json.loads(CHANNELS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return empty_state()
    if not isinstance(data, dict):
        return empty_state()
    data.setdefault("channels", [])
    data.setdefault("seen_videos", {})
    if not isinstance(data["channels"], list):
        data["channels"] = []
    if not isinstance(data["seen_videos"], dict):
        data["seen_videos"] = {}
    return data


def save_state(state: dict) -> None:
    CHANNELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHANNELS_FILE.with_suffix(".json.tmp")
    payload = json.dumps(state, indent=2, ensure_ascii=False) + "\n"
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(CHANNELS_FILE)


def ensure_dirs() -> None:
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)


def run_ytdlp(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
    cmd = [ytdlp_bin(), "--no-warnings", "--no-progress", "--ignore-config", *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


def ytdlp_json(url: str, extra: list[str] | None = None, timeout: int = 180) -> dict:
    extra = extra or []
    last_err = None
    for attempt in range(2):
        try:
            proc = run_ytdlp([*extra, "-J", "--", url], timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            last_err = f"yt-dlp timed out after {timeout}s"
            log(f"  retry after timeout ({attempt + 1}/2): {url}")
            time.sleep(2.0)
            continue
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip() or f"yt-dlp exit {proc.returncode}"
            last_err = _short_err(err)
            if _is_transient(err) and attempt == 0:
                log(f"  retry after error ({attempt + 1}/2): {last_err}")
                time.sleep(2.5)
                continue
            raise YtdlpError(last_err)
        raw = (proc.stdout or "").strip()
        if not raw:
            last_err = "yt-dlp returned empty JSON"
            if attempt == 0:
                time.sleep(1.5)
                continue
            raise YtdlpError(last_err)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}\s*$", raw, re.S)
            if m:
                return json.loads(m.group(0))
            last_err = "yt-dlp returned non-JSON output"
            if attempt == 0:
                time.sleep(1.5)
                continue
            raise YtdlpError(last_err)
    raise YtdlpError(last_err or "yt-dlp failed")


def _short_err(err: str) -> str:
    line = err.strip().splitlines()[-1] if err.strip() else "yt-dlp failed"
    line = re.sub(r"^ERROR:\s*", "", line)
    return line[:400]


def _is_transient(err: str) -> bool:
    low = err.lower()
    needles = (
        "timeout",
        "timed out",
        "429",
        "503",
        "500",
        "502",
        "temporarily",
        "try again",
        "connection",
        "reset",
        "unavailable",
        "http error 5",
        "network",
        "rate-limit",
        "rate limit",
        "sign in to confirm",
        "please wait",
    )
    return any(n in low for n in needles)


def normalize_input(raw: str) -> str:
    text = (raw or "").strip()
    if not text or text.startswith("#"):
        return ""
    text = text.split()[0]
    if CHANNEL_ID_RE.fullmatch(text):
        return f"https://www.youtube.com/channel/{text}"
    if text.startswith("@"):
        return f"https://www.youtube.com/{text}"
    if re.fullmatch(r"[A-Za-z0-9_.-]{3,32}", text) and not text.startswith("http"):
        # bare handle without @
        if not VIDEO_ID_RE.fullmatch(text):
            return f"https://www.youtube.com/@{text}"
    if text.startswith("http://") or text.startswith("https://"):
        return text
    if text.startswith("youtube.com") or text.startswith("www.youtube.com") or text.startswith("youtu.be/"):
        return "https://" + text
    return text


def is_video_url(url: str) -> bool:
    if VIDEO_ID_RE.fullmatch(url):
        return True
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().lstrip("www.")
    path = parsed.path or ""
    qs = parse_qs(parsed.query)
    if host in {"youtu.be"} and path.strip("/"):
        return True
    if "youtube.com" in host or host == "youtube-nocookie.com":
        if qs.get("v"):
            return True
        parts = [p for p in path.split("/") if p]
        if parts and parts[0] in {"watch", "shorts", "embed", "live", "v", "e"}:
            return True
    return False


def extract_video_id(raw: str) -> str | None:
    text = (raw or "").strip()
    if VIDEO_ID_RE.fullmatch(text):
        return text
    parsed = urlparse(text if re.match(r"^https?://", text, re.I) else "https://" + text)
    host = (parsed.hostname or "").lower().lstrip("www.")
    if host in {"youtu.be"}:
        vid = parsed.path.strip("/").split("/")[0] if parsed.path.strip("/") else ""
        if VIDEO_ID_RE.fullmatch(vid):
            return vid
    qs = parse_qs(parsed.query)
    if "v" in qs and VIDEO_ID_RE.fullmatch(qs["v"][0]):
        return qs["v"][0]
    parts = [p for p in parsed.path.split("/") if p]
    for i, part in enumerate(parts):
        if part in {"shorts", "embed", "live", "v", "e"} and i + 1 < len(parts):
            if VIDEO_ID_RE.fullmatch(parts[i + 1]):
                return parts[i + 1]
    m = re.search(r"(?:v=|/shorts/|/embed/|/live/|/v/|youtu\.be/)([A-Za-z0-9_-]{11})", text)
    return m.group(1) if m else None


def _clean_channel_name(name: str) -> str:
    name = (name or "").strip()
    for suffix in (" - Videos", " - Streams", " - Shorts", " - Live", " - Home"):
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()
    return name


def resolve_channel(url: str) -> dict:
    """Return {id, name, url} using yt-dlp. Accepts channel, @handle, or video URL."""
    url = normalize_input(url)
    if not url:
        raise YtdlpError("empty URL")

    if is_video_url(url):
        data = ytdlp_json(url, extra=["--skip-download", "--no-playlist"], timeout=120)
        ch_id = data.get("channel_id") or ""
        name = data.get("channel") or data.get("uploader") or ""
        if not ch_id:
            raise YtdlpError("could not resolve channel id from video URL")
        return {
            "id": ch_id,
            "name": _clean_channel_name(str(name)) or ch_id,
            "url": f"https://www.youtube.com/channel/{ch_id}",
        }

    data = ytdlp_json(url, extra=["--flat-playlist", "--playlist-end", "1"], timeout=120)
    ch_id = data.get("channel_id") or ""
    name = data.get("channel") or data.get("uploader") or data.get("title") or ""
    if not CHANNEL_ID_RE.fullmatch(str(ch_id)):
        maybe = data.get("id") or data.get("uploader_id") or data.get("channel_url") or ""
        m = re.search(r"(UC[A-Za-z0-9_-]{22})", str(maybe))
        if m:
            ch_id = m.group(1)
        elif CHANNEL_ID_RE.fullmatch(str(maybe)):
            ch_id = maybe
    if not CHANNEL_ID_RE.fullmatch(str(ch_id)):
        entries = data.get("entries") or []
        if entries and isinstance(entries[0], dict):
            e = entries[0]
            ch_id = e.get("channel_id") or ch_id
            name = name or e.get("channel") or e.get("uploader") or ""
    if not CHANNEL_ID_RE.fullmatch(str(ch_id)):
        # last try: channel_url field
        m = re.search(r"(UC[A-Za-z0-9_-]{22})", json.dumps({
            "channel_url": data.get("channel_url"),
            "uploader_url": data.get("uploader_url"),
            "webpage_url": data.get("webpage_url"),
            "original_url": data.get("original_url"),
        }))
        if m:
            ch_id = m.group(1)
    if not CHANNEL_ID_RE.fullmatch(str(ch_id)):
        raise YtdlpError(f"could not resolve channel id from {url}")
    name = _clean_channel_name(str(name)) or ch_id
    return {
        "id": ch_id,
        "name": name,
        "url": f"https://www.youtube.com/channel/{ch_id}",
    }


def list_recent_uploads(channel_id: str, limit: int = PLAYLIST_END) -> list[dict]:
    url = f"https://www.youtube.com/channel/{channel_id}/videos"
    data = ytdlp_json(
        url,
        extra=["--flat-playlist", "--playlist-end", str(limit)],
        timeout=180,
    )
    entries = data.get("entries") or []
    out = []
    for e in entries:
        if not e or not isinstance(e, dict):
            continue
        vid = e.get("id") or extract_video_id(e.get("url") or "")
        if not vid or not VIDEO_ID_RE.fullmatch(str(vid)):
            continue
        out.append(e)
    return out


def video_url_of(entry: dict, video_id: str) -> str:
    url = entry.get("url") or entry.get("webpage_url") or entry.get("original_url") or ""
    if url and url.startswith("http") and "/channel/" not in url:
        return url
    return f"https://www.youtube.com/watch?v={video_id}"


def published_iso(entry: dict) -> str:
    ts = entry.get("timestamp") or entry.get("release_timestamp")
    if ts:
        try:
            return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except (OSError, ValueError, OverflowError, TypeError):
            pass
    d = str(entry.get("upload_date") or entry.get("release_date") or "")
    if len(d) == 8 and d.isdigit():
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
    return ""


def skip_reason(entry: dict, include_shorts: bool = False) -> str | None:
    """Return why this entry should be skipped, or None to keep it.

    Defaults: skip YouTube Shorts, live streams, and premieres.
    """
    url = str(entry.get("url") or entry.get("webpage_url") or entry.get("original_url") or "")
    title = str(entry.get("title") or "")
    live_status = str(entry.get("live_status") or "")
    if live_status in SKIP_LIVE_STATUS:
        return f"live_status={live_status}"
    if entry.get("is_live"):
        return "is_live"
    if entry.get("was_live"):
        return "was_live"
    if "premiere" in title.lower() and live_status in {"is_upcoming", ""}:
        # upcoming premiere titles often include the word; don't over-filter VODs
        if live_status == "is_upcoming":
            return "premiere"
    if not include_shorts:
        if "/shorts/" in url:
            return "shorts_url"
        duration = entry.get("duration")
        try:
            dur = float(duration) if duration is not None else None
        except (TypeError, ValueError):
            dur = None
        if dur is not None and 0 < dur < 60:
            return "short_duration"
    return None


def is_transient_error(msg: str) -> bool:
    return _is_transient(msg)


def channel_matches(ch: dict, needle: str) -> bool:
    if not needle:
        return True
    n = needle.strip().rstrip("/").lower()
    n = n.lstrip("@")
    hay = " ".join(
        [
            ch.get("id") or "",
            ch.get("name") or "",
            ch.get("url") or "",
        ]
    ).lower()
    return n in hay


def find_sub_files(directory: Path, video_id: str) -> list[Path]:
    hits = []
    for p in directory.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".srt", ".vtt", ".ass"}:
            continue
        if p.name.startswith(video_id):
            hits.append(p)
    return sorted(hits, key=_sub_rank)


def _sub_rank(path: Path) -> tuple:
    name = path.name.lower()
    # prefer official en srt, then en vtt, then any srt, then any vtt
    official = 0 if ".auto" not in name and "auto" not in name else 1
    lang = 0
    if re.search(r"(^|[._-])en([._-]|$)", name) or ".en." in name or name.endswith(".en.srt") or name.endswith(".en.vtt"):
        lang = 0
    elif "en" in name:
        lang = 1
    else:
        lang = 2
    ext = 0 if path.suffix.lower() == ".srt" else 1
    return (official, lang, ext, path.name)


def download_transcript(video_id: str, url: str) -> Path:
    """Download SRT/VTT only (no media). Prefer official English, else auto, else any.

    Returns path to transcripts/{video_id}.srt
    """
    ensure_dirs()
    dest = TRANSCRIPTS_DIR / f"{video_id}.srt"
    if dest.exists() and dest.stat().st_size > 40:
        return dest

    outtmpl = str(TRANSCRIPTS_DIR / f"{video_id}.%(id)s")
    # yt-dlp appends lang; use a stable prefix via -o
    outtmpl = str(TRANSCRIPTS_DIR / video_id)

    attempts = [
        ["--skip-download", "--no-playlist", "--write-subs", "--sub-langs", "en.*,en",
         "--convert-subs", "srt", "--sub-format", "srt/vtt/best",
         "-o", outtmpl, "--", url],
        ["--skip-download", "--no-playlist", "--write-auto-subs", "--sub-langs", "en.*,en",
         "--convert-subs", "srt", "--sub-format", "srt/vtt/best",
         "-o", outtmpl, "--", url],
        ["--skip-download", "--no-playlist", "--write-subs", "--write-auto-subs",
         "--sub-langs", "all,-live_chat",
         "--convert-subs", "srt", "--sub-format", "srt/vtt/best",
         "-o", outtmpl, "--", url],
    ]
    last_err = "no subtitles"
    for args in attempts:
        try:
            proc = run_ytdlp(args, timeout=150)
        except subprocess.TimeoutExpired:
            last_err = "subtitle download timed out"
            continue
        err = (proc.stderr or proc.stdout or "").strip()
        files = find_sub_files(TRANSCRIPTS_DIR, video_id)
        if files:
            return _normalize_sub_file(files[0], dest)
        if proc.returncode != 0 and err:
            last_err = _short_err(err)
        elif "has no subtitles" in err.lower() or "no subtitle" in err.lower():
            last_err = "no subtitles available"
        else:
            last_err = last_err or "no subtitle file written"
    raise YtdlpError(last_err)


def _normalize_sub_file(src: Path, dest: Path) -> Path:
    text = src.read_text(encoding="utf-8", errors="replace")
    if src.suffix.lower() == ".vtt" or text.lstrip().startswith("WEBVTT"):
        dest.write_text(vtt_to_srt(text), encoding="utf-8")
    else:
        if src.resolve() != dest.resolve():
            dest.write_text(text, encoding="utf-8")
    # cleanup extras
    for extra in find_sub_files(TRANSCRIPTS_DIR, dest.stem):
        if extra.resolve() != dest.resolve():
            try:
                extra.unlink()
            except OSError:
                pass
    if not dest.exists() or dest.stat().st_size < 20:
        raise YtdlpError("subtitle file empty after convert")
    return dest


def vtt_to_srt(vtt: str) -> str:
    body = vtt.replace("\ufeff", "")
    if body.lstrip().startswith("WEBVTT"):
        # drop header up to first blank after WEBVTT
        parts = body.split("\n")
        i = 0
        while i < len(parts) and parts[i].strip() and not re.search(r"-->", parts[i]):
            i += 1
        body = "\n".join(parts[i:])
    cues = []
    blocks = re.split(r"\n\s*\n", body.strip())
    idx = 1
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip() != "NOTE"]
        if not lines:
            continue
        time_i = None
        for i, ln in enumerate(lines):
            if "-->" in ln:
                time_i = i
                break
        if time_i is None:
            continue
        timing = lines[time_i]
        m = re.search(
            r"(\d{1,2}:)?\d{1,2}:\d{2}[.,]\d{3}\s*-->\s*(\d{1,2}:)?\d{1,2}:\d{2}[.,]\d{3}",
            timing,
        )
        if not m:
            continue
        start, end = [p.strip() for p in re.split(r"\s*-->\s*", timing.split()[0] + " --> " + timing.split("-->", 1)[1].split()[0])]
        # safer split
        a, b = [x.strip().split()[0] for x in timing.split("-->", 1)]
        start_s = _vtt_ts_to_srt(a)
        end_s = _vtt_ts_to_srt(b)
        text_lines = []
        for ln in lines[time_i + 1 :]:
            ln = re.sub(r"<[^>]+>", "", ln)
            ln = ln.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            ln = re.sub(r"\s+", " ", ln).strip()
            if ln:
                text_lines.append(ln)
        text = " ".join(text_lines).strip()
        if not text:
            continue
        cues.append((idx, start_s, end_s, text))
        idx += 1
    out = []
    for i, start_s, end_s, text in cues:
        out.append(f"{i}\n{start_s} --> {end_s}\n{text}\n")
    return "\n".join(out) + ("\n" if out else "")


def _vtt_ts_to_srt(ts: str) -> str:
    ts = ts.replace(".", ",")
    parts = ts.split(":")
    if len(parts) == 2:
        ts = "00:" + ts
    return ts
