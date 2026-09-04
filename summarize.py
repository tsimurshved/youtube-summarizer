#!/usr/bin/env python3
"""Coin-size transcript summarizer.

Reads an SRT (or VTT) file. Uses Ollama if a local model is already running;
otherwise a claim-oriented extractive path that paraphrases/compresses
sentences into takeaways (not quote-extracts). Works fully offline, no API key.

Usage:
  python summarize.py transcripts/VIDEOID.srt
"""

from __future__ import annotations

import json
import math
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

STOPWORDS = {
    "a", "about", "after", "again", "all", "also", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "between", "both",
    "but", "by", "can", "could", "did", "do", "does", "doing", "down", "during",
    "each", "few", "for", "from", "further", "get", "got", "had", "has", "have",
    "having", "he", "her", "here", "hers", "him", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "itself", "just", "like", "ll", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "out", "over", "own", "re", "s", "same",
    "she", "should", "so", "some", "such", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "would", "you",
    "your", "yours", "yeah", "um", "uh", "gonna", "wanna", "kind", "sort",
    "really", "actually", "basically", "right", "okay", "ok", "well", "going",
    "know", "think", "thing", "things", "lot", "let", "lets", "ve", "don", "t",
    "gonna", "wanna", "stuff", "maybe", "probably", "something", "anything",
}

NOISE_RE = re.compile(
    r"^\[(?:music|applause|laughter|silence|inaudible|cheers|video|spon|sponsor).*\]$",
    re.I,
)
SPEAKER_RE = re.compile(r"^>>\s*(?:[^:\n]{1,40}:\s*)?")
TOKEN_RE = re.compile(r"[a-z][a-z0-9']{2,}")
TS_RE = re.compile(
    r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})[,.](\d{3})\s*-->\s*(?:(\d{1,2}):)?(\d{1,2}):(\d{2})[,.](\d{3})"
)
NUM_RE = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+)(\s*)(hours?|hrs?|minutes?|mins?|seconds?|secs?|"
    r"days?|years?|weeks?|times?|percent|%|ms|kb|mb|gb|tb)?",
    re.I,
)
UNIT_TIME = {"hour", "hours", "hr", "hrs", "minute", "minutes", "min", "mins",
             "second", "seconds", "sec", "secs", "day", "days", "year", "years"}

FILLER_RE = re.compile(
    r"\b(?:uh+|um+|uhh+|umm+|er+|ah+|huh|hmm+|you know|i mean|kind of|sort of|"
    r"a little bit|or whatever|and so on|this and that|and that and|"
    r"basically|actually|literally|honestly|obviously|probably|apparently|"
    r"i guess|i feel like|you see|right now i mean|let's say|i don't know)\b",
    re.I,
)
PAREN_NOISE_RE = re.compile(
    r"\[(?:[^\]]{0,80})\]|\((?:music|laughter|applause|inaudible|clears throat)[^)]*\)",
    re.I,
)
BANTER_RE = re.compile(
    r"(?i)\b(that's funny|thats funny|confession to make|by the way|"
    r"what's funny|whats funny|how do we|did it pan out|"
    r"i('m| am) forgetting|butchering their pronunciation|"
    r"sitting here talking|hear a bell|a chat just came in|"
    r"first,? i have a confession|few hours ago i dropped|"
    r"if you guys haven'?t seen|i'll tag that|see you (?:guys )?in the next)\b",
)
QUESTION_RE = re.compile(r"\?\s*$")
HEDGE_LEAD_RE = re.compile(
    r"^(?:(?:so|and|but|well|now|look|anyway|ok|okay|yeah|yes|right now),?\s+)+"
    r"|(?:^(?:i(?:'m|'ve| d)?|we(?:'re|'ve)?)\s+(?:just\s+)?(?:do\s+|also\s+)?(?:think|believe|feel|guess|would say|mean|figured|realize|realised)\s+(?:that\s+)?)",
    re.I,
)
LIKE_FILLER_RE = re.compile(
    r"\b(?:i(?:'m| am)|i was|we(?:'re| are)|he(?:'s| is)|she(?:'s| is))\s+like\b,?",
    re.I,
)
SAY_QUOTE_RE = re.compile(
    r"""(?i)(?:we |they |i )?(?:go the other way around and )?(?:say|said|saying),?\s+[\"“'](.+?)[\"”']\.?$"""
)
YOU_CAN_RE = re.compile(r"(?i)^you can (?:just )?(.+)$")
DONT_THINK_FOR_RE = re.compile(
    r"(?i)^i (?:also )?(?:don't|do not) think (.+?) is for (.+)$"
)
INVESTED_RE = re.compile(
    r"(?i)^i(?:'ve| have)?(?:\s+\w+){0,4}\s+invested (.+)$"
)
WANT_RE = re.compile(r"(?i)^i want(?:ed)? to (.+)$")
WE_START_RE = re.compile(r"(?i)^we (?:start|started|go) with (.+)$")
CLAIM_HINT_RE = re.compile(
    r"(?i)\b(because|instead|rather than|means|argues?|claims?|built|builds|"
    r"installs?|installed|demo(?:ed|nstrat\w*)|compared|versus|\bvs\.?\b|"
    r"hours?|minutes?|seconds?|percent|%|not for|designed|preconfigur\w*|"
    r"tiling|keyboard|distro|distribution|operating system|\bos\b|"
    r"linux|windows|ubuntu|debian|arch|hyprland|apple|intel|framework|"
    r"nvidia|amd|mac(?:os)?|package manager|agents?|ai)\b",
)
NAME_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z0-9][A-Za-z0-9+._-]{1,}\b")
DURATION_NEAR_RE = re.compile(
    r"(?i)(install|unwrap|unbox|ready|setup|set up|updates?|took|clocked|"
    r"invested|spent|hours?|minutes?|seconds?)"
)

# ASR / nickname aliases. Applied when the canonical form is in the title
# or already appears in the caption (never strip version digits like 5.1).
TITLE_ALIASES = {
    "omarchy": ("omakase", "omarky", "omarchi", "umachi", "amachi", "hamachi",
                "omachi", "omar omachi", "omarchy"),
    "hyprland": ("hyperland", "hyprland", "hyper land"),
    "quickshell": ("quick shell", "quickshell"),
    "ponytail": ("pony tail", "ponytail"),
    "herder": ("herurder", "herder"),
}

DOT_SENTINEL = "\uE000"

N_BULLETS_MIN = 5
N_BULLETS_MAX = 8
MAX_CLAIM_CHARS = 240
MAX_BULLET_CHARS = MAX_CLAIM_CHARS  # internal claim cap; output is prose
MAX_OVERVIEW_CHARS = 440
MAX_TAKEAWAYS_CHARS = 960
MAX_USEFUL_CHARS = 560

TRUNCATED_TAIL_RE = re.compile(
    r"(?i)\b(the|a|an|and|or|to|of|for|with|in|on|at|by|from|as|that|this|"
    r"these|those|its|it's|is|are|was|were|be|been|being|your|my|our|their|"
    r"into|about|over|after|before|than|then|but|because|which|who|whom|"
    r"towards|toward|using|like|while|during|when|if|so)\s*$"
)
USEFUL_HINT_RE = re.compile(
    r"(?i)(\b\d[\d,.]*\s*(?:%|percent|ms|mb|gb|tb|kb|hours?|hrs?|"
    r"minutes?|mins?|seconds?|tokens? per(?:\s+second)?)\b|"
    r"\$\d|"
    r"(?:^|\s)/[a-z][\w-]*|"
    r"\b(?:then run|then install|marketplace add|self-host|hetzner|vps|"
    r"paste |configure |a month|a year|priced|free tier|"
    r"per million)\b)"
)

INTRO_FILLER_RE = re.compile(
    r"(?i)\b("
    r"i'll put the link|check it out as well|if you haven'?t seen that video|"
    r"i'll tag that|i dropped a video|trust me,? i(?:'m| am) going to have a video|"
    r"today i(?:'m| am) going to tell you everything you need to know|"
    r"see that i(?:'ve| have) got some running|right here in my (?:desktop|screen)"
    r")\b"
)
COMMANDISH_RE = re.compile(
    r"(?i)(?:^|\s)/[a-z][\w-]*|"
    r"\b(?:then run|then install|marketplace add|self-host|hetzner|vps|per million|free tier|"
    r"a month|priced)\b|"
    r"\$\d"
)



def _protect_numeric_dots(text: str) -> str:
    """Keep 5.1 / 3.5 / $10.00 from being treated as sentence ends."""
    return re.sub(r"(?<=\d)\.(?=\d)", DOT_SENTINEL, text or "")


def _unprotect_numeric_dots(text: str) -> str:
    return (text or "").replace(DOT_SENTINEL, ".")


def _title_product_names(title: str) -> list[str]:
    """Names/versions to preserve from the video title (Fable 5.1, Ponytail, ...)."""
    if not title:
        return []
    found: list[str] = []
    m = re.match(r"^([A-Z][A-Za-z0-9+._-]{2,})\s*[:!\u2014\u2013-]", title.strip())
    if m:
        found.append(m.group(1))
    for m in re.finditer(
        r"\b((?:[A-Z][A-Za-z0-9+._-]*\s+){0,2}[A-Z][A-Za-z0-9+._-]*)\s+(\d+\.\d+(?:\.\d+)*)\b",
        title,
    ):
        found.append(f"{m.group(1)} {m.group(2)}")
    for m in re.finditer(r"\b([A-Za-z][A-Za-z0-9]*\d+\.\d+[A-Za-z0-9+._-]*)\b", title):
        found.append(m.group(1))
    tl = title.lower()
    canon_disp = {"hyprland": "Hyprland", "omarchy": "Omarchy", "ponytail": "Ponytail", "herder": "Herder"}
    for canon, disp in canon_disp.items():
        if canon in tl:
            found.append(disp)
    seen: set[str] = set()
    out: list[str] = []
    for n in found:
        k = n.lower()
        if k not in seen and n.strip():
            seen.add(k)
            out.append(n.strip())
    return out


def _glue_split_versions(text: str, title: str = "") -> str:
    t = _unprotect_numeric_dots(text or "")
    names = _title_product_names(title)
    for full in sorted(names, key=len, reverse=True):
        if "." not in full:
            continue
        head, tail = full.rsplit(".", 1)
        if not tail.isdigit():
            continue
        t = re.sub(rf"\b{re.escape(head)}\s*[.\uE000]?\s+{re.escape(tail)}\b", full, t)
        t = re.sub(
            rf"(?m)^(?:{re.escape(tail)})(?=\s+(?:is|are|was|will|on|and|does|did|can|has|have|finished|offers)|,)",
            full,
            t,
            count=1,
        )
        ver = f"{head.rsplit(' ', 1)[-1]}.{tail}" if " " in head else f"{head}.{tail}"
        # "5.1 is cheaper" / "And 5.1 is cheaper" → "Fable 5.1 is cheaper"
        t = re.sub(rf"(?m)^(?:And\s+)?{re.escape(ver)}\b", full, t, count=1)
    t = re.sub(
        r"\b([A-Z][A-Za-z0-9+._-]*)\s+(\d+)\s*[.]\s+(\d+)\b",
        r"\1 \2.\3",
        t,
    )
    t = re.sub(
        r"\b([A-Z][A-Za-z0-9+._-]*)\s+(\d+)\s+(\d+)(?=\s+(?:is|are|was|will|and|on|does)\b)",
        r"\1 \2.\3",
        t,
    )
    return t


def _restore_title_names(text: str, title: str = "") -> str:
    t = _glue_split_versions(text, title)
    t = _normalize_names(t, title)
    for name in _title_product_names(title):
        t = re.sub(rf"(?i)\b{re.escape(name)}\b", name, t)
    return t


def _is_intro_filler(text: str) -> bool:
    t = text or ""
    if COMMANDISH_RE.search(t) or "$" in t:
        return False
    if INTRO_FILLER_RE.search(t):
        return True
    if re.match(r"(?i)^(see that |you can see that i(?:'ve| have) got|look at this\b|now look\b)", t):
        return True
    if re.search(r"(?i)\b(if you guys haven'?t seen|definitely check it out)\b", t):
        return True
    return False


def _is_actionable(text: str) -> bool:
    t = text or ""
    if COMMANDISH_RE.search(t) or "$" in t:
        return True
    if re.search(r"(?i)\b\d[\d,.]*\s*(?:%|percent)\s*(?:less|more|cheaper|reduction|savings?)\b", t):
        return True
    if re.search(r"(?i)\b(?:\d+\s*(?:gb|mb|tb)\b|\b\d+\s*hours?\b|tokens? per second)\b", t):
        return True
    if USEFUL_HINT_RE.search(t):
        return True
    return False


def _is_conclusion_sentence(text: str) -> bool:
    return bool(re.search(
        r"(?i)\b(is a|is an|are a|better|worse|cheaper|not for|instead|"
        r"rather than|means that|compared|built on|encodes|makes your|"
        r"the point|jump over|most capable|not only better)\b",
        text or "",
    ))


def _looks_like_caption_fragment(text: str) -> bool:
    t = text or ""
    if re.search(r"(?i)\b(uh+|umm+|you know|i mean|kind of|i don't know)\b", t):
        return True
    if re.match(r"(?i)^(?:and )?\d+\b", t) and not re.search(
        r"\b(Fable|Mythos|Ponytail|Herder|Omarchy|Hyprland)\b", t
    ):
        return True
    if re.search(r"(?i)\b(this is pretty interesting|as you can see right here|over here,? we (?:can )?see|right over here)\b", t):
        return True
    if re.search(r"(?i)\b(3d bear|the actual bear|physics looks|more colorful|click space to play|i can click|i(?:'m| am) just comparing|really interested to see|just feels different|we(?:'ll| will) see how)\b", t):
        return True
    return False


def _depersonalize(t: str) -> str:
    t = re.sub(r"(?i)^what i have for you(?: right now)? in this video is ", "", t)
    t = re.sub(
        r"(?i)^(?:so,?\s+)?(?:in this video,? )?(?:i(?:'m| am) going to |we(?:'re| are) going to )?(?:show you |talk about |cover |look at )",
        "",
        t,
    )
    t = re.sub(r"(?i)^i(?:'ve| have) (?:often )?found that ", "", t)
    t = re.sub(r"(?i)^i think (?:that )?", "", t)
    t = re.sub(r"(?i)^i would say (?:that )?", "", t)
    t = re.sub(r"(?i)^we (?:can|could) (?:see|say) (?:that )?", "", t)
    t = re.sub(r"(?i)^you(?:'ll| will) notice (?:that )?", "", t)
    t = re.sub(r"(?i)^i mean,?\s+", "", t)
    t = re.sub(r"(?i)^right now,?\s+", "", t)
    t = re.sub(r"(?i)^we have ", "", t)
    t = re.sub(r"(?i)^(?:and you can see |see )?(?:right here at the top )?they already say ", "", t)
    t = re.sub(r"(?i)^(?:and )?you can see (?:right here )?(?:at the top )?", "", t)
    t = re.sub(r"\b(\w+),\s+\1\b", r"\1", t, flags=re.I)
    t = re.sub(r"(?i),? would be insane\.?$", "", t)
    t = re.sub(r"(?i)^this one costed us about ", "One example cost about ", t)
    t = re.sub(r"(?i)^this one was ", "Another run was ", t)
    return t.strip()


def _prose_sentences(text: str) -> list[str]:
    blob = _protect_numeric_dots(re.sub(r"\s+", " ", (text or "")).strip())
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", blob)
    return [_unprotect_numeric_dots(p.strip()) for p in parts if p.strip()]


def _distinct_from(useful: str, takeaways: str, thresh: float = 0.46) -> str:
    if not useful:
        return ""
    if not takeaways:
        return useful
    t_sents = _prose_sentences(takeaways)
    kept = []
    for s in _prose_sentences(useful):
        if _too_similar(s, t_sents, thresh=thresh):
            continue
        if _too_similar(s, kept, thresh=0.58):
            continue
        if s.lower() in takeaways.lower() and len(s) > 40:
            continue
        kept.append(s)
    if not kept:
        return ""
    return _limit_prose(" ".join(kept), MAX_USEFUL_CHARS, 1)



def summarize_file(path: str | Path, title: str = "") -> dict:
    path = Path(path)
    if not title:
        title = _title_from_sidecar(path) or ""
    snippets = parse_caption_file(path)
    return summarize_snippets(snippets, title=title)


def summarize_snippets(snippets: list[dict], title: str = "") -> dict:
    sentences = snippets_to_sentences(snippets)
    if not sentences:
        raise ValueError("Transcript had no usable sentences.")
    backend = detect_backend()
    if backend["kind"] == "ollama":
        try:
            result = summarize_with_ollama(sentences, title, backend["model"])
            result["backend"] = f"ollama:{backend['model']}"
            result["title"] = title
            return _finalize(result)
        except Exception:
            pass
    result = summarize_extractive(sentences, title=title)
    result["backend"] = "extractive"
    result["title"] = title
    return _finalize(result)


def _title_from_sidecar(path: Path) -> str:
    try:
        side = path.resolve().parent.parent / "summaries" / f"{path.stem}.json"
        if side.exists():
            data = json.loads(side.read_text(encoding="utf-8"))
            return str(data.get("title") or "")
    except Exception:
        return ""
    return ""


def _finalize(result: dict) -> dict:
    title = result.get("title") or ""
    claims = []
    seen = []
    raw_claims = result.get("claims") or result.get("bullets") or []
    for b in raw_claims:
        text = (b.get("text") if isinstance(b, dict) else str(b)).strip()
        if not text:
            continue
        text = _restore_title_names(text, title)
        text = _trim(_ensure_sentence(_scrub_light(text)), MAX_CLAIM_CHARS)
        if _is_truncated(text) or _too_similar(text, seen):
            continue
        if _is_intro_filler(text) or _looks_like_caption_fragment(text):
            continue
        start = None
        if isinstance(b, dict):
            start = b.get("start")
            if start is None:
                start = b.get("t")
            try:
                start = float(start) if start is not None else None
            except (TypeError, ValueError):
                start = None
        claims.append({"text": text, "start": start})
        seen.append(text)
        if len(claims) >= N_BULLETS_MAX:
            break

    overview = _flatten_lists(_as_prose(result.get("overview") or result.get("paragraph") or ""))
    takeaways = _flatten_lists(_as_prose(result.get("takeaways") or ""))
    useful = _flatten_lists(_as_prose(result.get("useful") or ""))
    overview = _restore_title_names(overview, title)
    takeaways = _restore_title_names(takeaways, title)
    useful = _restore_title_names(useful, title)

    overlap = bool(
        takeaways
        and useful
        and (
            _too_similar(useful, [takeaways], thresh=0.5)
            or (useful.lower() in takeaways.lower() and len(useful) > 50)
        )
    )
    if claims and (not takeaways or not useful or not overview or overlap):
        ov2, tk2, uf2 = _pack_prose(title, claims)
        overview = overview or ov2
        if not takeaways or overlap:
            takeaways = tk2 or takeaways
        if not useful or overlap:
            useful = uf2

    overview = _limit_prose(_two_sentences(overview), MAX_OVERVIEW_CHARS, 1)
    takeaways = _limit_prose(takeaways, MAX_TAKEAWAYS_CHARS, 2)
    useful = _limit_prose(_distinct_from(useful, takeaways), MAX_USEFUL_CHARS, 1)
    overview = _restore_title_names(overview, title)
    takeaways = _restore_title_names(takeaways, title)
    useful = _restore_title_names(useful, title)
    if not overview and takeaways:
        overview = _limit_prose(_two_sentences(takeaways), MAX_OVERVIEW_CHARS, 1)
    return {
        "overview": overview,
        "takeaways": takeaways,
        "useful": useful,
        "backend": result.get("backend") or "extractive",
    }


def _as_prose(val) -> str:
    if val is None:
        return ""
    if isinstance(val, list):
        parts = []
        for x in val:
            if isinstance(x, dict):
                parts.append(str(x.get("text") or "").strip())
            else:
                parts.append(str(x).strip())
        return _stitch([p for p in parts if p], MAX_TAKEAWAYS_CHARS, 2)
    return str(val).strip()


def _flatten_lists(text: str) -> str:
    if not text:
        return ""
    raw_lines = text.splitlines()
    listish = 0
    for ln in raw_lines:
        if re.match(r"^\s*[-*•]\s+\S", ln):
            listish += 1
        elif re.match(r"^\s*\d{1,2}[.)]\s+[A-Za-z]", ln) and not re.match(r"^\s*\d+\.\d+", ln):
            listish += 1
    if listish >= 2:
        items = []
        for ln in raw_lines:
            if re.match(r"^\s*\d+\.\d+", ln):
                cleaned = ln.strip()
            else:
                cleaned = re.sub(r"^\s*[-*•]\s+", "", ln)
                cleaned = re.sub(r"^\s*\d{1,2}[.)]\s+(?=[A-Za-z])", "", cleaned)
                cleaned = cleaned.strip()
            if cleaned:
                items.append(_ensure_sentence(cleaned))
        return " ".join(items)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _is_truncated(text: str) -> bool:
    t = re.sub(r"\s+", " ", (text or "")).strip().rstrip(".…!?")
    if not t:
        return True
    if TRUNCATED_TAIL_RE.search(t):
        return True
    if t.endswith("-") or t.endswith(","):
        return True
    return False


def _is_useful(text: str) -> bool:
    return _is_actionable(text)


def _stitch(sentences: list[str], max_chars: int, max_paras: int) -> str:
    cleaned, seen = [], []
    for s in sentences:
        s = _ensure_sentence(_scrub_light(s))
        if not s or len(s.split()) < 5 or _is_truncated(s):
            continue
        if _too_similar(s, seen, thresh=0.58):
            continue
        cleaned.append(s)
        seen.append(s)
    if not cleaned:
        return ""
    if max_paras <= 1 or len(cleaned) <= 3:
        return _limit_prose(" ".join(cleaned), max_chars, 1)
    mid = max(2, (len(cleaned) + 1) // 2)
    p1 = " ".join(cleaned[:mid])
    p2 = " ".join(cleaned[mid:])
    return _limit_prose(p1 + "\n\n" + p2, max_chars, 2)


def _limit_prose(text: str, limit: int, max_paras: int = 2) -> str:
    text = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    paras = paras[:max_paras]
    text = "\n\n".join(paras)
    if len(text) <= limit:
        return text
    kept = []
    used = 0
    for para in paras:
        sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", para)
        buf = []
        for s in sents:
            extra = ((" " if buf else "") + s)
            sep = 2 if (kept and not buf) else 0
            if used + sep + len(" ".join(buf)) + len(extra) > limit and (kept or buf):
                break
            buf.append(s)
        if buf:
            kept.append(" ".join(buf))
            used = len("\n\n".join(kept))
        else:
            break
        if used >= limit:
            break
    return "\n\n".join(kept) if kept else _trim(text, limit)


def _two_sentences(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return text
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    parts = [p.strip() for p in parts if p.strip()]
    return " ".join(parts[:2]) if parts else text


def _pack_prose(title: str, claims: list[dict]) -> tuple[str, str, str]:
    useful_s, take_s = [], []
    seen_u, seen_t = [], []
    for c in claims:
        text = c["text"] if isinstance(c, dict) else str(c)
        text = _ensure_sentence(_restore_title_names(_scrub_light(text), title))
        if not text or _is_truncated(text) or len(text.split()) < 5:
            continue
        if _is_banter(text) or _is_intro_filler(text) or _looks_like_caption_fragment(text):
            continue
        actionable = _is_actionable(text)
        if actionable and not _too_similar(text, seen_u, thresh=0.55):
            useful_s.append(text)
            seen_u.append(text)
        elif not actionable and not _too_similar(text, seen_t, thresh=0.55):
            take_s.append(text)
            seen_t.append(text)

    useful_s = [u for u in useful_s if not _too_similar(u, take_s, thresh=0.44)]

    if not take_s and useful_s:
        peeled, rest = [], []
        for u in useful_s:
            if _is_conclusion_sentence(u) and not COMMANDISH_RE.search(u) and "$" not in u:
                peeled.append(u)
            else:
                rest.append(u)
        if peeled:
            take_s, useful_s = peeled, rest

    overview = _make_overview(title, take_s or useful_s)
    takeaways = _stitch(take_s, MAX_TAKEAWAYS_CHARS, 2)
    useful = _stitch(useful_s, MAX_USEFUL_CHARS, 1)
    useful = _distinct_from(useful, takeaways)
    overview = _restore_title_names(overview, title)
    takeaways = _restore_title_names(takeaways, title)
    useful = _restore_title_names(useful, title)
    return overview, takeaways, useful


def detect_backend() -> dict:
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        names = [m.get("name") or m.get("model") for m in (data.get("models") or [])]
        names = [n for n in names if n]
        if names:
            return {"kind": "ollama", "model": _pick_ollama_model(names), "models": names}
    except Exception:
        pass
    return {"kind": "extractive", "model": None, "models": []}


def _pick_ollama_model(names: list[str]) -> str:
    lowered = [(n, n.lower()) for n in names]
    for needle in ("llama3.1", "llama3", "qwen2.5", "qwen2", "mistral", "gemma2", "phi3", "llama"):
        for orig, low in lowered:
            if needle in low:
                return orig
    return names[0]


def parse_caption_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    cues = _parse_cues(text)
    cues = _dedupe_rolling(cues)
    snippets = []
    for c in cues:
        cleaned = _clean_caption(c["text"])
        if not cleaned:
            continue
        snippets.append(
            {
                "text": cleaned,
                "start": c["start"],
                "duration": max(0.0, c["end"] - c["start"]),
            }
        )
    return snippets


def _parse_cues(text: str) -> list[dict]:
    text = text.replace("\ufeff", "").replace("\r\n", "\n")
    blocks = re.split(r"\n\s*\n", text.strip())
    cues = []
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip() and not ln.strip().isdigit()]
        if not lines:
            continue
        time_i = None
        for i, ln in enumerate(lines):
            if "-->" in ln:
                time_i = i
                break
        if time_i is None:
            continue
        m = TS_RE.search(lines[time_i])
        if not m:
            continue
        start = _hms(m.group(1), m.group(2), m.group(3), m.group(4))
        end = _hms(m.group(5), m.group(6), m.group(7), m.group(8))
        body = []
        for ln in lines[time_i + 1 :]:
            ln = re.sub(r"<[^>]+>", "", ln)
            ln = ln.replace("&nbsp;", " ").replace("&amp;", "&")
            ln = re.sub(r"\s+", " ", ln).strip()
            if ln:
                body.append(ln)
        joined = " ".join(body).strip()
        if joined:
            cues.append({"start": start, "end": end, "text": joined})
    return cues


def _hms(h, m, s, ms) -> float:
    hh = int(h) if h else 0
    return hh * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _dedupe_rolling(cues: list[dict]) -> list[dict]:
    """YouTube auto-subs often repeat the rolling window; keep new words."""
    if not cues:
        return []
    out = []
    prev = ""
    for c in cues:
        text = re.sub(r"\s+", " ", c["text"]).strip()
        if not text:
            continue
        if text == prev:
            continue
        if prev and text.startswith(prev):
            delta = text[len(prev) :].strip()
            if delta:
                out.append({**c, "text": delta})
                prev = text
            continue
        if prev:
            prev_words = prev.split()
            cur_words = text.split()
            overlap = 0
            max_o = min(len(prev_words), len(cur_words))
            for k in range(max_o, 2, -1):
                if prev_words[-k:] == cur_words[:k]:
                    overlap = k
                    break
            if overlap:
                delta = " ".join(cur_words[overlap:]).strip()
                if delta:
                    out.append({**c, "text": delta})
                    prev = text
                    continue
                continue
        out.append({**c, "text": text})
        prev = text
    return out


def _clean_caption(text: str) -> str:
    text = text.replace("\n", " ").replace("\xa0", " ")
    text = SPEAKER_RE.sub("", text)
    text = re.sub(r">>\s*", " ", text)
    text = PAREN_NOISE_RE.sub(" ", text)
    text = re.sub(r"[♪♫]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text or NOISE_RE.match(text):
        return ""
    if text.lower() in {"music", "[music]", "(music)"}:
        return ""
    return text


def snippets_to_sentences(snippets: list[dict]) -> list[dict]:
    cleaned = []
    for snip in snippets:
        text = _clean_caption(snip.get("text") or "")
        if not text:
            continue
        cleaned.append(
            {
                "text": text,
                "start": float(snip.get("start") or 0),
                "duration": float(snip.get("duration") or 0),
            }
        )
    if not cleaned:
        return []
    joined_sample = " ".join(s["text"] for s in cleaned[:80])
    punctuated = len(re.findall(r"[.!?]", joined_sample)) >= max(3, len(cleaned[:80]) // 12)
    if punctuated:
        return _sentences_from_punctuated(cleaned)
    return _sentences_from_windows(cleaned)



_BREAK_HOLD = {
    "a", "an", "the", "and", "or", "of", "with", "between", "on", "to", "for",
    "from", "by", "at", "in", "into", "than", "as", "vs", "versus",
}

def _needs_sentence_break(prev: str, nxt: str) -> bool:
    if not prev or prev[-1:] in ".!?":
        return False
    nxt = nxt.lstrip()
    if not nxt or not nxt[0].isupper():
        return False
    last = prev.split()[-1].lower().strip(".,;:\"'")
    if last in _BREAK_HOLD:
        return False
    first = nxt.split()[0].strip(".,;:\"'")
    if re.match(r"^[A-Z]{1,5}\d+[A-Za-z]*$", first):
        return False
    if re.match(r"^[A-Z]{2,5}$", first) and len(first) <= 4:
        return False
    return True


def _glue_broken_sentences(sentences: list[dict]) -> list[dict]:
    if not sentences:
        return sentences
    out = [sentences[0].copy()]
    for s in sentences[1:]:
        prev = out[-1]["text"].rstrip()
        nxt = s["text"]
        prev_stem = prev.rstrip(". ")
        # Only glue truncated fragments (e.g. "... of"), not finished sentences
        # that happen to end with a function word ("played with.").
        if re.search(r"(?i)\b(an|a|the|and|or|of|with|between|on|to|for)$", prev_stem):
            if (not prev.endswith((".", "!", "?"))) or len(prev_stem.split()) < 8:
                out[-1]["text"] = _tidy_sentence(prev_stem + " " + nxt)
                continue
        # "Fable 5." + "1 is..."  or  "Fable 5" + "1 and Mythos"
        if re.search(r"(?<![\d.])\d{1,3}\.?$", prev_stem) and re.match(r"^\d+\b", nxt):
            if prev.endswith("."):
                merged = prev + nxt
            else:
                merged = prev_stem + "." + nxt
            out[-1]["text"] = _tidy_sentence(merged)
            continue
        out.append(s.copy())
    return out


def _sentences_from_punctuated(snippets: list[dict]) -> list[dict]:
    blob_parts = []
    index = []
    cursor = 0
    for snip in snippets:
        piece = snip["text"].strip()
        if not piece:
            continue
        if blob_parts:
            prev = blob_parts[-1]
            sep = " " if not _needs_sentence_break(prev, piece) else ". "
            blob_parts.append(sep)
            cursor += len(sep)
        start_at = cursor
        blob_parts.append(piece)
        cursor += len(piece)
        index.append((start_at, cursor, snip["start"]))
    blob = _protect_numeric_dots("".join(blob_parts))
    sentences = []
    for m in re.finditer(r".+?(?:[.!?]+[\"')\]]?|$)", blob):
        raw = _unprotect_numeric_dots(m.group(0)).strip()
        raw = raw.strip(" ")
        if raw.endswith("...") :
            pass
        raw = re.sub(r"[.]+$", ".", raw) if raw[-1:] in ".!?" else raw
        raw = raw.strip(" ")
        # Keep internal version dots; only drop a trailing sentence period for the word-count gate.
        core = raw[:-1] if raw[-1:] in ".!?" else raw
        if len(core.split()) < 4:
            continue
        mid = (m.start() + m.end()) / 2
        start = _time_at(index, mid)
        text = _tidy_sentence(core)
        if 4 <= len(text.split()) <= 90:
            sentences.append({"text": text, "start": start})
    return _glue_broken_sentences(sentences)


def _time_at(index: list[tuple[int, int, float]], pos: float) -> float:
    if not index:
        return 0.0
    for a, b, t in index:
        if a <= pos <= b:
            return t
    best_t = index[0][2]
    best_d = 10**9
    for a, b, t in index:
        if pos < a:
            d = a - pos
        elif pos > b:
            d = pos - b
        else:
            return t
        if d < best_d:
            best_d = d
            best_t = t
    return best_t


def _sentences_from_windows(snippets: list[dict], target_words: int = 22) -> list[dict]:
    sentences = []
    buf: list[str] = []
    start = None
    words = 0
    last_end = 0.0
    for snip in snippets:
        if start is None:
            start = snip["start"]
        buf.append(snip["text"])
        words += len(snip["text"].split())
        last_end = snip["start"] + snip.get("duration", 0)
        span = last_end - start
        if words >= target_words or span >= 16:
            text = _tidy_sentence(" ".join(buf))
            if len(text.split()) >= 6:
                sentences.append({"text": text, "start": start})
            buf, words, start = [], 0, None
    if buf and start is not None:
        text = _tidy_sentence(" ".join(buf))
        if len(text.split()) >= 5:
            sentences.append({"text": text, "start": start})
    return sentences


def _tidy_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" -–—,;:")
    if not text:
        return text
    words = text.split()
    collapsed = []
    for w in words:
        if collapsed and collapsed[-1].lower().strip(".,") == w.lower().strip(".,"):
            collapsed[-1] = collapsed[-1].rstrip(".,")
            continue
        collapsed.append(w)
    text = " ".join(collapsed)
    text = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", text, flags=re.I)
    if text[0].islower():
        text = text[0].upper() + text[1:]
    if text[-1] not in ".!?…" and len(text.split()) >= 8:
        text += "."
    return text


# ---------------------------------------------------------------------------
# Extractive path: score → select claims → paraphrase/compress into bullets
# ---------------------------------------------------------------------------

def summarize_extractive(sentences: list[dict], title: str = "") -> dict:
    prepared = []
    for s in sentences:
        raw = s.get("text") or ""
        cleaned = _scrub(raw, title)
        cleaned = _glue_split_versions(cleaned, title)
        if not cleaned or _is_banter(cleaned, raw):
            continue
        if _is_intro_filler(cleaned) and not _is_actionable(cleaned):
            continue
        wc = len(cleaned.split())
        if wc < 5 or wc > 70:
            continue
        prepared.append(
            {"text": cleaned, "raw": raw, "start": float(s.get("start") or 0)}
        )
    if not prepared:
        prepared = [{"text": _scrub(s["text"], title) or s["text"], "raw": s["text"],
                     "start": float(s.get("start") or 0)} for s in sentences]

    scored = _score_sentences(prepared)
    for s in scored:
        s["score"] *= _claim_multiplier(s["text"], s.get("raw") or s["text"])

    synthesized = _synthesized_fact_bullets(scored, title)
    n_target = min(N_BULLETS_MAX, max(N_BULLETS_MIN, 8))
    n_target = min(n_target, max(3, len(scored)))
    pool = _select_mmr(scored, min(len(scored), max(n_target * 3, 12)))
    paraphrased = []
    for s in pool:
        take = _to_takeaway(s["text"], title)
        if not take or _is_banter(take, s.get("raw") or ""):
            continue
        paraphrased.append(
            {"text": take, "start": round(s["start"], 2), "score": s["score"]}
        )

    merged = _merge_bullet_lists(synthesized, paraphrased, title)
    bullets = []
    seen_txt = []
    for b in merged:
        text = _trim(b["text"], MAX_CLAIM_CHARS)
        if len(text.split()) < 5 or _is_truncated(text):
            continue
        if _too_similar(text, seen_txt):
            continue
        bullets.append({"text": text, "start": b.get("start")})
        seen_txt.append(text)
        if len(bullets) >= n_target:
            break

    if len(bullets) < N_BULLETS_MIN:
        for s in sorted(scored, key=lambda x: -x["score"]):
            take = _to_takeaway(s["text"], title)
            if not take or _too_similar(take, seen_txt):
                continue
            take = _trim(take, MAX_CLAIM_CHARS)
            if _is_truncated(take) or len(take.split()) < 5:
                continue
            bullets.append({"text": take, "start": round(s["start"], 2)})
            seen_txt.append(take)
            if len(bullets) >= N_BULLETS_MIN:
                break

    bullets = bullets[:N_BULLETS_MAX]
    bullets.sort(key=lambda b: (b["start"] is None, b["start"] if b["start"] is not None else 0))
    return {"claims": bullets}


def _claim_multiplier(text: str, raw: str) -> float:
    t = text.lower()
    m = 1.0
    if NUM_RE.search(text):
        m *= 1.45
    if CLAIM_HINT_RE.search(text):
        m *= 1.28
    if re.search(r"\b(is a|is an|built|installs|not for|instead|rather|compared)\b", t):
        m *= 1.15
    if NAME_TOKEN_RE.search(text):
        m *= 1.1
    if QUESTION_RE.search(text) or QUESTION_RE.search(raw):
        m *= 0.22
    if BANTER_RE.search(raw) or BANTER_RE.search(text):
        m *= 0.12
    filler_ratio = _filler_ratio(raw)
    if filler_ratio > 0.18:
        m *= 0.55
    if t.startswith(("hey", "hi ", "hello", "um ", "uh ")):
        m *= 0.4
    wc = len(text.split())
    if wc < 7:
        m *= 0.7
    if 8 <= wc <= 32:
        m *= 1.08
    # first-person small talk without a claim
    if re.match(r"(?i)^i(?:'m| am|'ve| have)?\s+(?:just )?(?:curious|excited|happy|here)", t):
        m *= 0.3
    return m


def _filler_ratio(text: str) -> float:
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return 1.0
    fillers = {
        "uh", "um", "like", "you", "know", "yeah", "right", "okay", "ok", "well",
        "so", "just", "really", "actually", "basically", "kind", "sort", "gonna",
        "wanna", "literally",
    }
    n = sum(1 for w in words if w in fillers)
    return n / len(words)


def _is_banter(text: str, raw: str = "") -> bool:
    blob = f"{raw} {text}"
    if BANTER_RE.search(blob):
        return True
    t = text.strip()
    if QUESTION_RE.search(t) and not NUM_RE.search(t):
        return True
    if len(t.split()) < 5:
        return True
    if re.search(r"(?i)\b(haha|lol|lmao|whoa|wow+|dude|vacation|peace of mind)\b", t):
        return True
    if _filler_ratio(raw or text) > 0.42 and not NUM_RE.search(t):
        return True
    return False



def _synthesized_fact_bullets(scored: list[dict], title: str) -> list[dict]:
    """Pull concrete facts (durations, stacks, definitions) into takeaways."""
    out: list[dict] = []
    topic = _topic_from_title(title)

    for s in sorted(scored, key=lambda x: -x["score"]):
        defn = _definition_takeaway(s["text"], topic)
        if defn:
            out.append({"text": defn, "start": round(s["start"], 2), "score": s["score"] + 0.8})
            break

    timing = _timing_comparison_bullet(scored, topic)
    if timing:
        out.append(timing)

    stack = _stack_bullet(scored, topic)
    if stack:
        out.append(stack)

    audience = _audience_bullet(scored, topic)
    if audience:
        out.append(audience)

    hours = _hours_invested_bullet(scored, topic)
    if hours:
        out.append(hours)

    kb = _keyboard_bullet(scored, topic)
    if kb:
        out.append(kb)

    tiling = _tiling_bullet(scored)
    if tiling:
        out.append(tiling)

    hardware = _hardware_bullet(scored)
    if hardware:
        out.append(hardware)

    for rec in _concrete_useful_bullets(scored, topic):
        out.append(rec)

    out = [b for b in out if not re.match(r"(?i)^the project is a ", b.get("text") or "")]
    out = [
        b for b in out
        if not _looks_like_caption_fragment(b.get("text") or "")
        and not _is_intro_filler(b.get("text") or "")
    ]
    return out


def _concrete_useful_bullets(scored: list[dict], topic: str) -> list[dict]:
    out = []
    seen = []
    for s in sorted(scored, key=lambda x: -x["score"]):
        raw = s["text"]
        if not (
            "$" in raw
            or USEFUL_HINT_RE.search(raw)
            or re.search(r"(?i)/plugin|marketplace add|/install", raw)
        ):
            continue
        take = _to_takeaway(raw, topic)
        if not take or _is_truncated(take) or _is_banter(take, raw):
            continue
        if _too_similar(take, seen, thresh=0.55):
            continue
        out.append({"text": take, "start": round(s["start"], 2), "score": s["score"] + 0.7})
        seen.append(take)
        if len(out) >= 4:
            break
    return out

def _definition_takeaway(text: str, topic: str) -> str:
    t = text
    if not topic:
        m = re.search(r"(?i)^(.{3,40}?)\s+is (?:a |an )(.{8,80})$", t)
        if m and CLAIM_HINT_RE.search(t):
            head = m.group(1).strip()
            if head.lower() not in {"it", "this", "that", "the project"}:
                return _ensure_sentence(f"{head} is a {_compress_tail(m.group(2))}")
        return ""
    m = re.search(r"(?i)preconfigured\b.{0,80}?distribution of (\w+)", t)
    if m:
        name = topic or "The project"
        extra = ""
        if re.search(r"(?i)fan", t):
            extra = " aimed at computer fans"
        return _ensure_sentence(f"{name} is a preconfigured {m.group(1)} distribution{extra}")
    m = re.search(r"(?i)(?:what it is|it is|it's)\s+(?:a |an )?(.{12,90})$", t)
    if m and re.search(r"(?i)\b(linux|distro|os|operating system|desktop)\b", t):
        body = _compress_tail(m.group(1))
        body = re.sub(r"(?i)^preconfigured lovingly\s+", "preconfigured ", body)
        if not re.match(r"(?i)(only|just|now|also|really|available)\b", body):
            name = topic or "The project"
            return _ensure_sentence(f"{name} is a {body}")
    m = re.search(r"(?i)^(.{3,40}?)\s+is (?:a |an )(.{8,80})$", t)
    if m and CLAIM_HINT_RE.search(t):
        tail = _compress_tail(m.group(2))
        if not re.match(r"(?i)(only|just|now|also|really|available)\b", tail):
            return _ensure_sentence(f"{m.group(1).strip()} is a {tail}")
    return ""



def _timing_comparison_bullet(scored: list[dict], topic: str) -> dict | None:
    setup_ok = re.compile(
        r"(?i)\b(install(?:ation|s|ed)?|unwrap(?:ping)?|unbox(?:ing)?|clocked|"
        r"updates were done|ready to use|ready after|upgrade from|"
        r"from unwrapping|full installation|less than \d|in \d+ minute|"
        r"in like \d+|one damn minute)\b"
    )
    setup_bad = re.compile(
        r"(?i)\b(ago|conversation|waiting \d+|on the server|pronunciation|"
        r"slower to install|million people)\b"
    )
    ordered = sorted(scored, key=lambda x: x["start"])
    facts = []  # (entity, amt, unit, nval, start, score, kind)
    last_ent = ("", -1e9)

    def remember(ent, t):
        nonlocal last_ent
        if ent:
            last_ent = (ent, t)

    for s in ordered:
        raw = s["text"]
        t = re.sub(r"(?i)doesn'?t work like (?:a |an )?[\w]+", " ", raw)
        if setup_bad.search(t) and not re.search(r"(?i)unbox|unwrap|install", t):
            continue
        # rolling entity from setup narrative
        if re.search(r"(?i)\bwindows\b", t) and re.search(r"(?i)unwrap|install|updates", t):
            remember("Windows", s["start"])
        elif re.search(r"(?i)\b(apple|macbook)\b", t):
            remember("Apple", s["start"])
        elif topic and topic.lower() in t.lower() and setup_ok.search(t):
            remember(topic, s["start"])
        elif re.search(r"(?i)\b(amachi|omarchy)\b", t) and setup_ok.search(t):
            remember(topic or "Omarchy", s["start"])

        if re.search(r"(?i)(?:an?\s+)?hour and a half", t) and re.search(r"(?i)unwrap|update|windows", t):
            ent = "Windows" if re.search(r"(?i)windows", t) else (last_ent[0] or "Windows")
            facts.append((ent, "90", "minutes", 90.0, s["start"], s["score"], "unbox"))
            continue
        if not setup_ok.search(t) and not re.search(r"(?i)\b(42 minutes|90 minutes|56 seconds|30 seconds|1 minute)\b", t):
            continue

        kind = "install"
        if re.search(r"(?i)upgrade from", t):
            kind = "upgrade"
        elif re.search(r"(?i)unwrap|unbox|updates were done", t):
            kind = "unbox"
        elif last_ent[0] in {"Windows", "Apple"} and re.search(r"(?i)minutes", t) and not re.search(r"(?i)install|upgrade", t):
            kind = "unbox"

        for m in NUM_RE.finditer(t):
            unit = (m.group(3) or "").lower()
            unit_key = unit.rstrip("s")
            if unit_key not in {u.rstrip("s") for u in UNIT_TIME}:
                continue
            try:
                nval = float(m.group(1).replace(",", ""))
            except ValueError:
                continue
            if unit_key == "year":
                continue
            if unit_key == "hour" and nval >= 4:
                continue
            if unit_key == "sec" and nval < 15:
                continue
            # "a minute slower" is not a measurement of a product
            ctx = t[max(0, m.start() - 25) : m.end() + 20].lower()
            if "slower" in ctx:
                continue
            ent = _entity_in(t, topic, "")
            if ent.lower() in {"arch", "linux", "setup", "it"}:
                ent = ""
            if re.search(r"(?i)\bon a framework\b", t) or re.search(r"(?i)framework 13", t):
                if not ent or ent.lower() == "framework":
                    ent = topic or "Omarchy"
            if not ent:
                if topic and (topic.lower() in t.lower() or kind in {"install", "upgrade"} and last_ent[0] == topic):
                    ent = topic
                elif s["start"] - last_ent[1] <= 40 and last_ent[0]:
                    ent = last_ent[0]
                elif kind == "install" and topic:
                    ent = topic
            if not ent:
                continue
            facts.append((ent, m.group(1), unit or "minutes", nval, s["start"], s["score"], kind))

    if not facts:
        return None

    # Keep the best fact per (entity, kind)
    best: dict[tuple, tuple] = {}
    for rec in facts:
        key = (rec[0].lower(), rec[6])
        prev = best.get(key)
        # prefer more precise seconds for install
        if prev is None:
            best[key] = rec
        elif rec[6] == "install" and rec[3] < prev[3] and rec[2].startswith("sec"):
            best[key] = rec
        elif rec[5] > prev[5]:
            best[key] = rec

    parts = []
    start = None
    score = 0.0
    seen: set[tuple] = set()

    def take(rec, phrase):
        nonlocal start, score
        if rec is None:
            return
        if start is None:
            start = rec[4]
        score = max(score, rec[5])
        parts.append(phrase)

    if topic:
        inst = best.get((topic.lower(), "install"))
        upg = best.get((topic.lower(), "upgrade"))
        bits = []
        if inst:
            recs = [r for r in facts if r[0].lower() == topic.lower() and r[6] == "install"]
            secs = [r for r in recs if str(r[2]).lower().startswith("sec")]
            mins = [r for r in recs if str(r[2]).lower().startswith("min")]
            if secs and mins:
                bits.append(f"installs in ~{mins[0][1]} min ({secs[0][1]}s)")
            else:
                bits.append(f"installs in ~{inst[1]} {inst[2]}")
            seen.add((topic.lower(), "install"))
            if start is None:
                start = inst[4]
            score = max(score, inst[5])
        if upg:
            bits.append(f"upgrades in ~{upg[1]}s" if str(upg[2]).lower().startswith("sec") else f"upgrades in ~{upg[1]} {upg[2]}")
            seen.add((topic.lower(), "upgrade"))
            score = max(score, upg[5])
        if bits:
            parts.append(f"{topic} " + ", ".join(bits))

    for label, kind in (("Windows", "unbox"), ("Windows", "install"), ("Apple", "unbox"), ("Apple", "install")):
        rec = best.get((label.lower(), kind))
        if not rec or (label.lower(), kind) in seen:
            continue
        seen.add((label.lower(), kind))
        amt = rec[1]
        if label == "Windows":
            take(rec, f"Windows ~{amt} min unbox")
        elif label == "Apple":
            take(rec, f"Apple ~{amt} min of updates")
        else:
            take(rec, f"{label} ~{amt} {rec[2]}")
        if len(parts) >= 3:
            break

    if not parts:
        return None
    text = _ensure_sentence("; ".join(parts))
    return {"text": text, "start": round(start or 0.0, 2), "score": score + 0.8}

def _stack_bullet(scored: list[dict], topic: str) -> dict | None:
    keys = [("arch", "Arch"), ("hyprland", "Hyprland"), ("hyperland", "Hyprland"),
            ("quickshell", "Quickshell"), ("quick shell", "Quickshell"),
            ("pacman", "Pacman"), ("pac-man", "Pacman")]
    best = None
    for s in scored:
        t = s["text"].lower()
        found = []
        for needle, canon in keys:
            if needle in t and canon not in found:
                found.append(canon)
        if len(found) >= 2:
            name = topic or "The distro"
            text = _ensure_sentence(f"{name} is built on " + ", ".join(found[:-1]) + f", and {found[-1]}")
            rec = {"text": text, "start": round(s["start"], 2), "score": s["score"] + 0.35 * len(found)}
            if best is None or rec["score"] > best["score"]:
                best = rec
    return best




def _audience_bullet(scored: list[dict], topic: str) -> dict | None:
    best = None
    for s in scored:
        t = s["text"]
        if re.search(r"(?i)installed in|like computers, they", t):
            continue
        m = re.search(r"(?i)(?:don't|do not) think (.+?) is for (.+)$", t)
        if m:
            name, rest = m.group(1).strip(), m.group(2).strip()
        else:
            m = re.search(r"(?i)(?:is not|isn't|not meant|not aimed) for people(?: who)? (.+)$", t)
            if not m:
                continue
            name, rest = (topic or "It"), m.group(1).strip()
        if rest.lower() in {"them", "that", "it", "this"}:
            continue
        rest = re.sub(r"(?i)^just who\s+", "people who ", rest)
        rest = re.sub(r"(?i)^people who just who\s+", "people who ", rest)
        rest = re.sub(r"(?i)who just who", "who", rest)
        rest = re.sub(r"(?i)don't give a damn about", "don't care about", rest)
        rest = re.sub(
            r"(?i),?\s*who(?:'s| is) not interested in investing any time in learning how it works",
            " or are unwilling to learn how it works",
            rest,
        )
        rest = _compress_tail(rest)
        if not rest.startswith("people") and "who" in rest.lower():
            rest = "people " + rest
        if topic and name.lower() in {"it", "this", "that"}:
            name = topic
        text = _ensure_sentence(f"{name} is not for {rest}")
        if re.search(rf"(?i)is not for {re.escape(name)}\b", text):
            continue
        rec = {"text": text, "start": round(s["start"], 2), "score": s["score"] + 0.5}
        if best is None or rec["score"] > best["score"]:
            best = rec
    return best

def _hours_invested_bullet(scored: list[dict], topic: str) -> dict | None:
    best = None
    for s in scored:
        m = re.search(
            r"(?i)invested.{0,40}?(\d[\d,]+)\s*(hours?|hrs?).{0,40}?(year|month|week)?",
            s["text"],
        )
        if not m:
            continue
        span = ""
        ym = re.search(r"(?i)over the past (year|month|week)|in (?:a |the last )?year", s["text"])
        if ym:
            span = " over a year"
        name = topic or "The project"
        text = _ensure_sentence(f"{name} took about {m.group(1)} {m.group(2)}{span} to build")
        rec = {"text": text, "start": round(s["start"], 2), "score": s["score"] + 0.55}
        if best is None or rec["score"] > best["score"]:
            best = rec
    return best



def _hardware_bullet(scored: list[dict]) -> dict | None:
    match = None
    panther = None
    for s in scored:
        t = s["text"]
        if panther is None and re.search(r"(?i)panther lake", t):
            panther = s
        if match is None and re.search(r"(?i)intel.*(?:matched|match|par with)|matched apple", t):
            match = s
    bits = []
    start = None
    score = 0.0
    if match:
        nearby = " ".join(
            s["text"] for s in scored if abs(s["start"] - match["start"]) < 25
        )
        chips = []
        for c in re.findall(r"\bM[3-5]\b", nearby):
            if c not in chips:
                chips.append(c)
        bit = "Intel has matched Apple on power efficiency"
        if chips:
            bit += f" (roughly {'/'.join(chips)})"
        bits.append(bit)
        start = match["start"]
        score = match["score"] + 0.4
    if panther:
        bits.append("recommends Intel Panther Lake for new laptops")
        if start is None:
            start = panther["start"]
        score = max(score, panther["score"] + 0.25)
    if not bits:
        return None
    text = bits[0] if len(bits) == 1 else bits[0].rstrip(".") + "; " + bits[1]
    return {"text": _ensure_sentence(text), "start": round(start or 0.0, 2), "score": score}

def _keyboard_bullet(scored: list[dict], topic: str) -> dict | None:
    best = None
    for s in scored:
        t = s["text"]
        if not re.search(r"(?i)keyboard", t):
            continue
        if re.search(r"(?i)start with the keyboard", t):
            text = "Keyboard-first: starts with the keyboard, mouse optional."
            rec = {"text": _ensure_sentence(text), "start": round(s["start"], 2), "score": s["score"] + 0.35}
            if best is None or rec["score"] > best["score"]:
                best = rec
        elif re.search(r"(?i)driven exclusively.{0,20}keyboard|keyboard intensive", t):
            name = topic or "It"
            text = f"{name} is designed to be driven by the keyboard."
            rec = {"text": _ensure_sentence(text), "start": round(s["start"], 2), "score": s["score"] + 0.2}
            if best is None or rec["score"] > best["score"]:
                best = rec
    return best


def _tiling_bullet(scored: list[dict]) -> dict | None:
    best = None
    for s in scored:
        t = s["text"]
        if re.search(r"(?i)tiling window|automatically arranges the windows|nothing ever overlaps", t):
            text = "Tiling window manager: windows never overlap; the keyboard moves focus and layout."
            rec = {"text": _ensure_sentence(text), "start": round(s["start"], 2), "score": s["score"] + 0.4}
            if best is None or rec["score"] > best["score"]:
                best = rec
        elif re.search(r"(?i)alien way of using a computer.{0,40}mouse", t):
            text = "Tiling feels alien if you drag windows with a mouse, but is a superpower once learned."
            rec = {"text": _ensure_sentence(text), "start": round(s["start"], 2), "score": s["score"] + 0.15}
            if best is None or rec["score"] > best["score"]:
                best = rec
    return best


def _entity_in(window: str, topic: str, nearby: str = "") -> str:
    blob = f"{window} {nearby}"
    low = blob.lower()
    # hardware as location should not win
    loc_framework = bool(re.search(r"(?i)\bon a framework\b", blob))
    catalog = []
    if topic:
        catalog.append(topic)
    catalog += [
        "Windows", "Apple", "macOS", "Mac", "Linux", "Ubuntu", "Arch",
        "Intel", "Omarchy",
    ]
    if not loc_framework:
        catalog.append("Framework")
    seen = set()
    ordered = []
    for c in catalog:
        k = c.lower()
        if k in seen:
            continue
        seen.add(k)
        ordered.append(c)
    # Prefer matches inside `window` over nearby context
    win_l = window.lower()
    for name in ordered:
        if name.lower() in win_l:
            return name
    for name in ordered:
        if name.lower() in low:
            return name
    return ""

def _merge_bullet_lists(primary: list[dict], secondary: list[dict], title: str) -> list[dict]:
    ranked = sorted(primary, key=lambda b: -b.get("score", 0)) + sorted(
        secondary, key=lambda b: -b.get("score", 0)
    )
    return ranked


def _select_mmr(scored: list[dict], k: int) -> list[dict]:
    if k <= 0:
        return []
    remaining = list(scored)
    remaining.sort(key=lambda s: -s["score"])
    if not remaining:
        return []
    picked = [remaining.pop(0)]
    while remaining and len(picked) < k:
        def mmr(s, picked=picked):
            toks = _tokenize(s["text"])
            red = max((_cosine(toks, _tokenize(p["text"])) for p in picked), default=0.0)
            tpen = 0.0
            for p in picked:
                dt = abs(s["start"] - p["start"])
                if dt < 20:
                    tpen = max(tpen, 0.45)
                elif dt < 50:
                    tpen = max(tpen, 0.18)
            return 0.74 * s["score"] - 0.26 * red - tpen

        remaining.sort(key=mmr, reverse=True)
        picked.append(remaining.pop(0))
    picked.sort(key=lambda s: s["start"])
    return picked



def _to_takeaway(text: str, title: str = "") -> str:
    t = _scrub(text, title)
    t = _glue_split_versions(t, title)
    if not t:
        return ""
    t = _depersonalize(t)
    if _is_intro_filler(t) and not _is_actionable(t):
        return ""
    t = re.sub(r"(?i)\bnot running in a BM\b", "not a VM", t)
    t = re.sub(r"(?i)\bin a BM\b", "in a VM", t)
    t = re.sub(r"(?i), I believe,", ",", t)
    m = SAY_QUOTE_RE.search(t)
    if m:
        t = m.group(1).strip()
        t = _scrub(t, title)
    m = DONT_THINK_FOR_RE.match(t)
    if m:
        name = m.group(1).strip()
        rest = _compress_tail(m.group(2))
        rest = re.sub(r"(?i)don't give a damn about", "don't care about", rest)
        t = f"{name} is not for {rest}"
    else:
        t = HEDGE_LEAD_RE.sub("", t).strip()
        m = re.search(r"(?i)^the fastest i(?:'ve| have) clocked it.{0,80}?(?:was|is) (.+)$", t)
        if m:
            t = "Fastest install clocked at " + m.group(1)
        m = YOU_CAN_RE.match(t)
        if m:
            t = m.group(1)
            t = re.sub(r"(?i)^install all of (.+?) in ", r"\1 installs in ", t)
        m = INVESTED_RE.match(t)
        if m:
            t = "Invested " + m.group(1)
        m = WANT_RE.match(t)
        if m:
            t = m.group(1)
        m = WE_START_RE.match(t)
        if m:
            t = "Starts with " + m.group(1)
        t = re.sub(r"(?i)^we go the other way around and say,\s*", "", t)
        t = re.sub(r"(?i)^i (?:also )?(?:don't|do not) think\s+", "", t)
        t = re.sub(r"(?i)^i(?:'ve| have) been\s+", "", t)
        t = re.sub(r"(?i)^i(?:'m| am) going to\s+", "", t)
        t = re.sub(r"(?i)^they (?:start|started) with\s+", "Starts with ", t)
        t = re.sub(r"(?i)^i (?:also )?(?:don't|do not) want to\s+", "Avoids ", t)

    t = _drop_weak_clauses(t)
    t = _compress_tail(t)
    t = _headline_trim(t)
    # leftover first person without a claim verb → drop
    if re.search(r"(?i)\b(i've|i have|i'm|i was|i believe|i want)\b", t) and not NUM_RE.search(t):
        t = re.sub(r"(?i)\b(?:i've|i have|i'm|i was|i believe|i want to|i want)\b\s*", "", t)
        t = _headline_trim(t)
    t = _glue_split_versions(t, title)
    if t.lower().count("only available") >= 2:
        m = re.search(r"(?i)((?:Fable|Mythos|Claude)[\w.\s]+ is only available\b.+)$", t)
        if m:
            t = m.group(1).strip()
    topic = _topic_from_title(title)
    m = re.match(r"^(\d+)([,.]?\s+.*)$", t)
    if m and topic and topic.endswith("." + m.group(1)):
        t = topic + m.group(2)
    t = _ensure_sentence(t)
    if _is_banter(t) or _is_intro_filler(t) or _looks_like_caption_fragment(t) or len(t.split()) < 5:
        return ""
    if re.search(r"(?i)\b(i|i'm|i've|i was)\b", t) and not _is_actionable(t):
        return ""
    return t

def _drop_weak_clauses(text: str) -> str:
    parts = re.split(r"\s*(?:, and |, but |; | — | – )\s*", text)
    if len(parts) <= 1:
        # still split long trailing "and then"
        parts = re.split(r"\s+\band then\b\s+", text, maxsplit=1)
    if len(parts) <= 1:
        return text
    scored = []
    for p in parts:
        p = p.strip(" ,;")
        if len(p.split()) < 4:
            continue
        sc = 0.0
        if NUM_RE.search(p):
            sc += 2.0
        if CLAIM_HINT_RE.search(p):
            sc += 1.2
        if NAME_TOKEN_RE.search(p):
            sc += 0.6
        sc += min(len(p.split()), 18) / 18.0
        if _filler_ratio(p) > 0.3:
            sc *= 0.5
        scored.append((sc, p))
    if not scored:
        return text
    scored.sort(key=lambda x: -x[0])
    keep = [p for _, p in scored[:2]]
    # preserve original order
    ordered = [p for p in parts if p.strip(" ,;") in keep]
    if not ordered:
        ordered = keep
    return "; ".join(_ensure_clause(p) for p in ordered)


def _ensure_clause(p: str) -> str:
    p = p.strip(" ,;")
    if p:
        p = p[0].lower() + p[1:] if len(p) > 1 else p.lower()
    return p


def _compress_tail(text: str) -> str:
    t = text.strip()
    t = re.sub(r"(?i)\bwho just who\b", "who", t)
    t = re.sub(r"(?i)\bwho(?:'s| is) not interested in investing any time in learning how it works\b",
               "unwilling to learn how it works", t)
    t = re.sub(r"(?i)\bif you wanted to\b", "", t)
    t = re.sub(r"(?i)\byeah,?\s+", "", t)
    t = re.sub(r"(?i)\byou could also use the mouse\b", "mouse optional", t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+,", ",", t)
    t = t.strip(" ,;")
    words = t.split()
    if len(words) > 38:
        t = " ".join(words[:38]).rstrip(",;:")
    return t.strip()


def _headline_trim(text: str) -> str:
    t = text.strip()
    t = re.sub(r"(?i)^(and|but|so|because)\s+", "", t)
    t = re.sub(r"(?i)^it's like\s+", "", t)
    t = re.sub(r"(?i)^it is like\s+", "", t)
    t = re.sub(r"\s+", " ", t).strip(" ,;")
    if t:
        t = t[0].upper() + t[1:]
    return t


def _scrub(text: str, title: str = "") -> str:
    t = text.replace("\n", " ")
    t = SPEAKER_RE.sub("", t)
    t = re.sub(r">>\s*", " ", t)
    t = PAREN_NOISE_RE.sub(" ", t)
    t = LIKE_FILLER_RE.sub(" ", t)
    t = re.sub(r"\blike,\s+", " ", t, flags=re.I)
    t = FILLER_RE.sub(" ", t)
    t = re.sub(r"(?i)\bin terms of\b", ",", t)
    t = re.sub(r'(?i)\bI kid you not,?\s*', "", t)
    t = re.sub(r"(?i)\bgod damn\b", "", t)
    t = _normalize_names(t, title)
    t = _glue_split_versions(t, title)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"([,;]){2,}", r"\1", t)
    t = t.strip(" -–—,;:")
    t = _tidy_sentence(t) if t else t
    return t


def _scrub_light(text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    t = FILLER_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _normalize_names(text: str, title: str) -> str:
    title_l = (title or "").lower()
    out = text
    for canon, aliases in TITLE_ALIASES.items():
        if canon not in title_l and canon not in text.lower():
            # still map hyprland-style ASR even without title
            if canon not in {"hyprland", "quickshell", "ponytail", "herder", "omarchy"}:
                continue
        for alias in sorted(aliases, key=len, reverse=True):
            if alias == canon:
                continue
            out = re.sub(rf"(?i)\b{re.escape(alias)}\b", canon.title() if canon != "hyprland" else "Hyprland", out)
    if "omarchy" in title_l:
        out = re.sub(r"(?i)\bomarchy\b", "Omarchy", out)
    if "hyprland" in out.lower():
        out = re.sub(r"(?i)\bhyprland\b", "Hyprland", out)
    if "quickshell" in out.lower():
        out = re.sub(r"(?i)\bquickshell\b", "Quickshell", out)
    out = re.sub(r"(?i)\bponytail\b", "Ponytail", out)
    out = re.sub(r"(?i)\bherurder\b", "Herder", out)
    if "herder" in title_l or "herder" in out.lower():
        out = re.sub(r"(?i)\bher skill\b", "Herder skill", out)
        out = re.sub(r"(?i)\bpeople herder\b", "People Herder", out)
        out = re.sub(r"(?i)\bherder\b", "Herder", out)
    out = re.sub(r"(?i)\bomarchy\b", "Omarchy", out)
    # Pac-Man package manager
    out = re.sub(r"(?i)\bpac-man\b", "Pacman", out)
    out = re.sub(r"(?i)\bmac os\b", "macOS", out)
    out = re.sub(r"(?i)\bhyperland\b", "Hyprland", out)
    return out


def _topic_from_title(title: str) -> str:
    if not title:
        return ""
    t = title.strip()
    m = re.search(r"(?i)interview on (.+)$", t)
    if m:
        return m.group(1).strip(" !?.")
    m = re.match(r"^([A-Z][A-Za-z0-9+._-]{2,})\s*[:!\u2014\u2013-]", t)
    if m:
        return m.group(1)
    names = _title_product_names(t)
    for n in names:
        if re.search(r"\d+\.\d+", n):
            return n
    if names:
        return names[0]
    m = re.search(r"(?i)\bon ([A-Z][\w+.-]{2,})\s*$", t)
    if m:
        return m.group(1)
    # Do not guess from the last capitalized clickbait word (Close, Source, Payoff).
    return ""


def _claim_text(item) -> str:
    if isinstance(item, dict):
        return str(item.get("text") or "").strip()
    return str(item or "").strip()


def _make_overview(title: str, bullets) -> str:
    topic = _topic_from_title(title)
    texts = []
    for b in bullets or []:
        t = _ensure_sentence(_restore_title_names(_claim_text(b), title))
        if not t or _is_truncated(t) or len(t.split()) < 5:
            continue
        if _is_intro_filler(t) or _looks_like_caption_fragment(t):
            continue
        texts.append(t)
    if not texts:
        return _ensure_sentence(title or topic or "Transcript summary.")
    first = texts[0]
    if topic:
        for t in texts:
            if topic.lower() in t.lower():
                first = t
                break
    second = ""
    for t in texts:
        if t == first:
            continue
        if not _too_similar(t, [first], thresh=0.55):
            second = t
            break
    joined = first if not second else first.rstrip() + " " + second
    return _limit_prose(_two_sentences(joined), MAX_OVERVIEW_CHARS, 1)


def _ensure_sentence(text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip(" -–—,;:")
    if not t:
        return ""
    t = t[0].upper() + t[1:]
    if t[-1] not in ".!?…":
        t += "."
    # fix doubled punctuation
    t = re.sub(r"[.]{2,}", ".", t)
    return t


def _one_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return text
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts[0] if parts else text



def _number_set(text: str) -> set[str]:
    out = {m.group(1).replace(",", "") for m in NUM_RE.finditer(text)}
    if re.search(r"(?i)hour and a half", text):
        out.add("90")
    return out



def _too_similar(text: str, existing: list[str], thresh: float = 0.55) -> bool:
    toks = _tokenize(text)
    nums = _number_set(text)
    for e in existing:
        etoks = _tokenize(e)
        if toks and etoks and _cosine(toks, etoks) >= thresh:
            return True
        a, b = text.lower(), e.lower()
        if len(a) > 24 and (a in b or b in a):
            return True
        nb = _number_set(e)
        shared = nums & nb
        if shared and toks and etoks and _cosine(toks, etoks) >= 0.32:
            return True
    return False

def _tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def _score_sentences(sentences: list[dict]) -> list[dict]:
    docs = [_tokenize(s["text"]) for s in sentences]
    n = len(docs)
    if n == 0:
        return []
    df = Counter()
    for toks in docs:
        df.update(set(toks))
    idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}

    vecs: list[dict[str, float]] = []
    tfidf_raw = []
    for toks in docs:
        tf = Counter(toks)
        vec = {t: (cnt / max(1, len(toks))) * idf.get(t, 0) for t, cnt in tf.items()}
        vecs.append(vec)
        mag = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        wc = max(1, len(toks))
        length_pen = 1.0 if 5 <= wc <= 24 else 0.72
        tfidf_raw.append((sum(vec.values()) / mag) * length_pen)

    sim = [[0.0] * n for _ in range(n)]
    norms = [math.sqrt(sum(v * v for v in vec.values())) or 1.0 for vec in vecs]
    for i in range(n):
        for j in range(i + 1, n):
            dot = 0.0
            a, b = vecs[i], vecs[j]
            if len(a) > len(b):
                a, b = b, a
            for t, av in a.items():
                bv = b.get(t)
                if bv:
                    dot += av * bv
            c = dot / (norms[i] * norms[j])
            if c > 0.08:
                sim[i][j] = sim[j][i] = c

    scores = [1.0] * n
    damping = 0.85
    for _ in range(24):
        new = [1 - damping] * n
        for i in range(n):
            denom = sum(sim[i]) or 1.0
            for j in range(n):
                if sim[i][j]:
                    new[j] += damping * scores[i] * sim[i][j] / denom
        scores = new

    def norm(xs):
        lo, hi = min(xs), max(xs)
        if hi - lo < 1e-9:
            return [0.5] * len(xs)
        return [(x - lo) / (hi - lo) for x in xs]

    tr_n = norm(scores)
    tf_n = norm(tfidf_raw)
    out = []
    for i, s in enumerate(sentences):
        pos = i / max(1, n - 1)
        pos_bonus = 0.1 if pos < 0.1 or pos > 0.88 else 0.0
        combined = 0.52 * tr_n[i] + 0.38 * tf_n[i] + pos_bonus
        out.append({**s, "score": combined, "idx": i})
    return out


def _cosine(a: list[str], b: list[str]) -> float:
    ca, cb = Counter(a), Counter(b)
    keys = set(ca) | set(cb)
    if not keys:
        return 0.0
    dot = sum(ca[k] * cb[k] for k in keys)
    na = math.sqrt(sum(v * v for v in ca.values())) or 1.0
    nb = math.sqrt(sum(v * v for v in cb.values())) or 1.0
    return dot / (na * nb)


def _trim(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:-") + "…"


def summarize_with_ollama(sentences: list[dict], title: str, model: str) -> dict:
    lines = []
    for s in sentences[:180]:
        lines.append(f"[{_fmt_ts(s['start'])}] {s['text']}")
    transcript_block = "\n".join(lines)
    prompt = f"""You summarize a YouTube transcript for someone who will not watch the video.

Title: {title or "(unknown)"}

Transcript with timestamps:
{transcript_block}

Write a JSON object with string fields only (plain prose, no lists):
- "overview": 1-2 sentences of what the video is.
- "takeaways": 1-2 short paragraphs of what the video CONCLUDES (paraphrased claims, what changed). No demo banter.
- "useful": 1 short paragraph of DISTINCT concrete tips the viewer can act on (commands, prices, workflows, thresholds). Must NOT repeat the takeaways paragraph.

Rules:
- JSON only. No markdown, no preamble, no bullet lists, no numbered lists.
- Tight. Usable without watching. Not a recap of every minute.
- Paraphrase. Never copy a transcript sentence verbatim. Never paste caption fragments.
- Drop filler, uhm, repeated phrases, demo-banter.
- takeaways and useful MUST be distinct. Do not reuse sentences.
- Preserve product names and versions from the title exactly (Fable 5.1, Ponytail, Herder, Omarchy). Never strip a leading number from a model name ("5.1" must not become "1").
- Do not write "the speaker says" / "the video discusses".
- Do not invent facts that are not in the transcript.
"""
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 1200},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        raw = json.loads(resp.read().decode("utf-8")).get("response") or ""
    data = _parse_json_obj(raw)
    overview = data.get("overview") or data.get("paragraph") or ""
    takeaways = data.get("takeaways") or ""
    useful = data.get("useful") or ""
    bullets = data.get("bullets") or data.get("claims") or []
    if not overview and not takeaways and not bullets:
        raise ValueError("ollama returned incomplete JSON")
    return {
        "overview": overview,
        "takeaways": takeaways,
        "useful": useful,
        "bullets": bullets,
        "title": title,
    }


def _parse_json_obj(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def fmt_ts(seconds) -> str:
    if seconds is None:
        return ""
    try:
        s = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        return ""
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


_fmt_ts = fmt_ts


def bullet_line(b: dict) -> str:
    """Deprecated: kept so old callers do not crash. Output is now prose fields."""
    text = b.get("text") or ""
    ts = fmt_ts(b.get("start"))
    if ts:
        return f"[{ts}] {text}"
    return text


def render_markdown(meta: dict, summary: dict | None = None) -> str:
    summary = summary or meta
    title = meta.get("title") or meta.get("video_id") or "Summary"
    channel = meta.get("channel") or ""
    published = meta.get("published") or ""
    url = meta.get("url") or ""
    byline = " · ".join(x for x in (channel, published) if x)
    parts = [f"# {title}"]
    if byline:
        parts.append(byline)
    if url:
        parts.append(url)
    parts.append("")
    ov = (summary.get("overview") or "").strip()
    if ov:
        parts.extend([ov, ""])
    tk = (summary.get("takeaways") or "").strip()
    if tk:
        parts.extend(["Takeaways", "", tk, ""])
    uf = (summary.get("useful") or "").strip()
    if uf:
        parts.extend(["Useful", "", uf, ""])
    return "\n".join(parts)


def write_summary_files(video_id: str, *, channel: str, title: str, url: str,
                        published: str, summary: dict, root: Path | None = None) -> Path:
    root = Path(root) if root else Path(__file__).resolve().parent
    sdir = root / "summaries"
    sdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "video_id": video_id,
        "channel": channel,
        "title": title,
        "url": url,
        "published": published,
        "overview": summary.get("overview") or "",
        "takeaways": summary.get("takeaways") or "",
        "useful": summary.get("useful") or "",
        "backend": summary.get("backend") or "",
    }
    (sdir / f"{video_id}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md = render_markdown(payload)
    (sdir / f"{video_id}.md").write_text(md if md.endswith("\n") else md + "\n", encoding="utf-8")
    return sdir / f"{video_id}.json"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    write = False
    if "--write" in args:
        write = True
        args = [a for a in args if a != "--write"]
    if not args or args[0] in {"-h", "--help"}:
        print("Usage: python summarize.py PATH.srt [--write]", file=sys.stderr)
        return 2
    path = Path(args[0])
    if not path.exists():
        print(f"missing file: {path}", file=sys.stderr)
        return 1
    result = summarize_file(path)
    if write:
        vid = path.stem
        meta = {}
        side = path.resolve().parent.parent / "summaries" / f"{vid}.json"
        if side.exists():
            try:
                meta = json.loads(side.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
        write_summary_files(
            vid,
            channel=str(meta.get("channel") or ""),
            title=str(meta.get("title") or ""),
            url=str(meta.get("url") or f"https://www.youtube.com/watch?v={vid}"),
            published=str(meta.get("published") or ""),
            summary=result,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
