# YouTube channel monitor

Lightweight CLI for a Grok Bot / agent workflow: watch creator channels, grab English captions for **new long-form uploads**, write tight prose summaries (`overview`, `takeaways`, `useful`). No web UI, no server.

Pair this with the **Transcript** bot template (or any agent that runs these scripts on a schedule).

## Setup

```bash
cd youtube-summarizer   # or /workspace/youtube-summarizer on the bot computer
python3 -m venv .venv
.venv/bin/pip install -U -r requirements.txt
```

## Add your channels

```bash
.venv/bin/python add_channels.py https://www.youtube.com/@SOME_CREATOR
.venv/bin/python add_channels.py --file channel_urls.txt
```

Dedupes by channel id. Accepts channel URLs, `@handles`, `/channel/UC…`, or a video URL (resolved to its channel).

## Check for new videos

```bash
.venv/bin/python check_new.py
```

JSON array of summaries from **this run** goes to stdout (progress on stderr).

### First-run seeding

When a channel has no rows yet in `seen_videos`, the checker marks the current latest uploads (`--playlist-end 8`) as seen **without summarizing**. Only later uploads trigger summaries.

### Smoke test

```bash
.venv/bin/python check_new.py --backfill 1 --channel HANDLE_OR_NAME
```

### Skipped by default

- YouTube Shorts (`/shorts/` or duration under 60s)
- Lives and premieres

Captions: SRT/VTT only — never the media file.

## Summarizer

`summarize.py` reads an SRT. Uses Ollama if a local daemon is already running; otherwise an offline paraphrase path. Output is prose, not bullet lists.

## Layout

```
add_channels.py
check_new.py
summarize.py
common.py
channels.json
channel_urls.txt
transcripts/
summaries/
```

## Privacy

Keep this repo private if you only want your community to have the scripts. Do not commit personal `channels.json` or transcript dumps.
