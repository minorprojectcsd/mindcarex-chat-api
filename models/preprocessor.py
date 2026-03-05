"""
models/preprocessor.py
─────────────────────────────────────────────────────────────
Parses WhatsApp .txt exports into a clean DataFrame.

Handles ALL known export formats:
  - DD/MM/YY, H:MM\u202fpm - User: msg    (Android India/UK narrow-space lowercase)
  - DD/MM/YY, H:MM am - User: msg         (Android regular space lowercase)
  - MM/DD/YY, H:MM AM - User: msg         (Android US uppercase)
  - DD/MM/YYYY, HH:MM - User: msg         (Android 24-hr)
  - [DD/MM/YYYY, H:MM:SS am] User: msg    (iOS)
  - DD.MM.YYYY, HH:MM - User: msg         (dot-separated locales)
"""

import re
import pandas as pd

# ── Normalise Unicode before any matching ──────────────────────────────────────
# \u202f = NARROW NO-BREAK SPACE (used by WhatsApp between time and am/pm)
# \u200e = LEFT-TO-RIGHT MARK    (prepended on system messages)
# \u200f = RIGHT-TO-LEFT MARK
# \u202a/\u202c = LTR/RTL embedding marks

def _clean_raw(raw: str) -> str:
    """Normalise special unicode characters in the raw export."""
    raw = raw.replace("\u202f", " ")   # narrow no-break → regular space
    raw = raw.replace("\u00a0", " ")   # non-breaking space → regular space
    raw = raw.replace("\u200e", "")    # LTR mark
    raw = raw.replace("\u200f", "")    # RTL mark
    raw = raw.replace("\u202a", "")    # LTR embedding
    raw = raw.replace("\u202c", "")    # POP directional formatting
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    return raw


# ── Regex patterns — most-specific first ──────────────────────────────────────
# After normalisation, am/pm has a regular space before it.
# We match case-insensitively so both 'am'/'pm' and 'AM'/'PM' work.

_PATTERNS = [
    # iOS: [DD/MM/YYYY, H:MM:SS am] User: msg
    r"^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}:\d{2}\s[aApP][mM])\]\s([^:]+):\s([\s\S]*)",
    # Android 12-hr with am/pm  (DD/MM/YY or MM/DD/YY)
    r"^(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}\s[aApP][mM])\s-\s([^:]+):\s([\s\S]*)",
    # Android 24-hr  (DD/MM/YY or MM/DD/YY)
    r"^(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{2}:\d{2})\s-\s([^:]+):\s([\s\S]*)",
    # Dot-separated date, 24-hr
    r"^(\d{1,2}\.\d{1,2}\.\d{2,4}),\s(\d{1,2}:\d{2}(?:\s[aApP][mM])?)\s-\s([^:]+):\s([\s\S]*)",
]
_COMPILED = [re.compile(p) for p in _PATTERNS]

# ── Special message detectors ──────────────────────────────────────────────────
_MEDIA_RE   = re.compile(
    r"<Media omitted>|image omitted|video omitted|audio omitted"
    r"|sticker omitted|document omitted|GIF omitted|Voice message omitted",
    re.I,
)
_DELETED_RE  = re.compile(r"This message was deleted|You deleted this message", re.I)
_MISSED_RE   = re.compile(r"Missed (?:voice|video) call", re.I)
_POLL_RE     = re.compile(r"^POLL:", re.I)
_LINK_RE     = re.compile(r"https?://\S+")

# System message patterns (skip these lines)
_SYSTEM_MSGS = re.compile(
    r"Messages (?:and calls )?are end.to.end encrypted"
    r"|end-to-end encrypted"
    r"|created group|added|removed|left|changed the subject"
    r"|changed this group|pinned a message|turned on|security code changed"
    r"|Your security code|Tap to learn more",
    re.I,
)


def _try_parse(raw: str) -> list:
    """
    Walk lines, match message headers, accumulate multiline bodies.
    Returns list of [date_str, time_str, user, message].
    """
    results, cur = [], None
    for line in raw.split("\n"):
        matched = False
        for pat in _COMPILED:
            m = pat.match(line)
            if m:
                if cur:
                    results.append(cur)
                cur = list(m.groups())   # [date_str, time_str, user, message]
                matched = True
                break
        if not matched and cur:
            cur[3] += "\n" + line
    if cur:
        results.append(cur)
    return results


def _parse_time_to_hour(time_str: str) -> int:
    """Extract hour (0–23) from various time string formats."""
    try:
        t = time_str.strip()
        t_up = t.upper()
        # Remove seconds  e.g. "10:30:45 PM" → "10:30 PM"
        t_up = re.sub(r"(\d{1,2}:\d{2}):\d{2}(\s[AP]M)", r"\1\2", t_up)
        if "AM" in t_up or "PM" in t_up:
            from datetime import datetime
            return datetime.strptime(t_up, "%I:%M %p").hour
        return int(t.split(":")[0])
    except Exception:
        return -1


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Try multiple date formats; fall back to pandas inference."""
    fmts = [
        "%d/%m/%y", "%d/%m/%Y",
        "%m/%d/%y", "%m/%d/%Y",
        "%d.%m.%y", "%d.%m.%Y",
    ]
    for fmt in fmts:
        try:
            df["date"] = pd.to_datetime(df["date_str"], format=fmt)
            return df
        except Exception:
            continue
    # Last resort
    df["date"] = pd.to_datetime(df["date_str"], dayfirst=True, infer_datetime_format=True)
    return df


def preprocess(raw: str) -> pd.DataFrame:
    """
    Parse raw WhatsApp export string → enriched DataFrame.

    Columns: date_str, time_str, user, message,
             date, only_date, year, month_num, month, day_name,
             hour, minute,
             is_media, is_deleted, is_missed_call, is_poll, has_link,
             message_type, word_count, char_count, emoji_count,
             response_time_min
    """
    # Normalise unicode first
    raw = _clean_raw(raw)

    records = _try_parse(raw)
    if not records:
        raise ValueError(
            "Could not parse chat. Ensure it is a valid WhatsApp .txt export.\n"
            "Export: Open chat → ⋮ → More → Export Chat → Without Media"
        )

    df = pd.DataFrame(records, columns=["date_str", "time_str", "user", "message"])

    # ── Strip whitespace ────────────────────────────────────────────────────
    df["user"]    = df["user"].str.strip()
    df["message"] = df["message"].str.strip()

    # ── Drop system / group notification lines ──────────────────────────────
    mask_system = (
        df["user"].isin(["group_notification", ""]) |
        df["message"].str.contains(_SYSTEM_MSGS, na=False)
    )
    df = df[~mask_system].copy().reset_index(drop=True)

    if df.empty:
        raise ValueError("No user messages found after filtering system messages.")

    # ── Parse dates ─────────────────────────────────────────────────────────
    df = _parse_dates(df)
    df["only_date"] = df["date"].dt.date
    df["year"]      = df["date"].dt.year
    df["month_num"] = df["date"].dt.month
    df["month"]     = df["date"].dt.month_name()
    df["day_name"]  = df["date"].dt.day_name()
    df["hour"]      = df["time_str"].apply(_parse_time_to_hour)
    df["minute"]    = df["time_str"].str.extract(r":(\d{2})")[0].astype(int, errors="ignore")

    # ── Message type flags ──────────────────────────────────────────────────
    df["is_media"]       = df["message"].str.contains(_MEDIA_RE, na=False)
    df["is_deleted"]     = df["message"].str.contains(_DELETED_RE, na=False)
    df["is_missed_call"] = df["message"].str.contains(_MISSED_RE, na=False)
    df["is_poll"]        = df["message"].str.match(_POLL_RE, na=False)
    df["has_link"]       = df["message"].str.contains(_LINK_RE, na=False)

    def _msg_type(row):
        if row["is_media"]:        return "media"
        if row["is_deleted"]:      return "deleted"
        if row["is_missed_call"]:  return "missed_call"
        if row["is_poll"]:         return "poll"
        if row["has_link"]:        return "link"
        return "text"

    df["message_type"] = df.apply(_msg_type, axis=1)

    # ── Word / char / emoji counts ──────────────────────────────────────────
    text_mask = ~df["is_media"]
    df["word_count"] = df["message"].where(text_mask, "").apply(lambda x: len(x.split()))
    df["char_count"] = df["message"].where(text_mask, "").apply(len)

    try:
        import emoji as emoji_lib
        df["emoji_count"] = df["message"].apply(
            lambda x: sum(1 for ch in str(x) if ch in emoji_lib.EMOJI_DATA)
        )
    except Exception:
        df["emoji_count"] = 0

    # ── Response time between consecutive messages ──────────────────────────
    df = df.sort_values("date").reset_index(drop=True)
    df["prev_user"] = df["user"].shift(1)
    df["prev_time"] = df["date"].shift(1)
    df["response_time_min"] = (
        (df["date"] - df["prev_time"]).dt.total_seconds() / 60
    ).where(df["user"] != df["prev_user"]).clip(upper=1440)
    df.drop(columns=["prev_user", "prev_time"], inplace=True)

    return df.reset_index(drop=True)
