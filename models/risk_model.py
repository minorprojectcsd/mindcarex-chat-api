"""
models/risk_model.py
Medical-grade risk analysis — answers WHO, WHAT, WHY, DIRECTION, TIMELINE.

Every risk field has plain-English meaning.
No ambiguity about who the risk is about or who it is directed at.
"""

import re
from collections import defaultdict
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# KEYWORD REGISTRY
# (phrase, weight, category, direction)
# direction:
#   "self"  = this person is at risk themselves (e.g. "I want to die")
#   "other" = this person is posing a risk to someone else (e.g. "I'll hurt you")
#   "both"  = context-dependent
# ═══════════════════════════════════════════════════════════════════════════════

_REGISTRY = [
    # Self-harm / Suicidal ideation
    ("suicide",                    10.0, "self_harm",          "self"),
    ("suicidal",                   10.0, "self_harm",          "self"),
    ("kill myself",                10.0, "self_harm",          "self"),
    ("end my life",                10.0, "self_harm",          "self"),
    ("not worth living",            9.0, "self_harm",          "self"),
    ("want to die",                 9.0, "self_harm",          "self"),
    ("rather be dead",              9.0, "self_harm",          "self"),
    ("cut myself",                  9.0, "self_harm",          "self"),
    ("hurt myself",                 8.0, "self_harm",          "self"),
    ("self harm",                   9.0, "self_harm",          "self"),
    ("no reason to live",           9.0, "self_harm",          "self"),
    ("disappear forever",           7.0, "self_harm",          "self"),
    ("everyone better without me",  8.0, "self_harm",          "self"),
    ("don't want to be here",       7.0, "self_harm",          "self"),
    ("life is pointless",           7.0, "self_harm",          "self"),

    # Emotional distress / Mental health signals
    ("depressed",                   6.0, "emotional_distress", "self"),
    ("depression",                  6.0, "emotional_distress", "self"),
    ("anxiety",                     5.0, "emotional_distress", "self"),
    ("panic attack",                6.0, "emotional_distress", "self"),
    ("hopeless",                    7.0, "emotional_distress", "self"),
    ("worthless",                   7.0, "emotional_distress", "self"),
    ("can't cope",                  6.0, "emotional_distress", "self"),
    ("breaking down",               6.0, "emotional_distress", "self"),
    ("falling apart",               6.0, "emotional_distress", "self"),
    ("nobody cares",                6.0, "emotional_distress", "self"),
    ("all alone",                   5.0, "emotional_distress", "self"),
    ("no one understands",          5.0, "emotional_distress", "self"),
    ("tired of everything",         5.0, "emotional_distress", "self"),
    ("can't take it",               6.0, "emotional_distress", "self"),
    ("mental breakdown",            7.0, "emotional_distress", "self"),
    ("losing my mind",              6.0, "emotional_distress", "self"),
    ("not okay",                    4.0, "emotional_distress", "self"),
    ("hate myself",                 7.0, "emotional_distress", "self"),
    ("feel empty",                  6.0, "emotional_distress", "self"),
    ("feel nothing",                6.0, "emotional_distress", "self"),

    # Aggression / Threats directed AT someone else
    ("kill you",                   10.0, "aggression_toward_other", "other"),
    ("murder",                     10.0, "aggression_toward_other", "other"),
    ("i will hurt you",             9.0, "aggression_toward_other", "other"),
    ("beat you",                    8.0, "aggression_toward_other", "other"),
    ("attack you",                  9.0, "aggression_toward_other", "other"),
    ("destroy you",                 7.0, "aggression_toward_other", "other"),
    ("make you pay",                7.0, "aggression_toward_other", "other"),
    ("you will regret",             6.0, "aggression_toward_other", "other"),
    ("weapon",                      8.0, "aggression_toward_other", "other"),
    ("shoot you",                   9.0, "aggression_toward_other", "other"),
    ("stab you",                    9.0, "aggression_toward_other", "other"),

    # Harassment / Control toward someone else
    ("blackmail",                   9.0, "harassment_of_other",    "other"),
    ("expose you",                  8.0, "harassment_of_other",    "other"),
    ("revenge",                     6.0, "harassment_of_other",    "other"),
    ("stalking you",                8.0, "harassment_of_other",    "other"),
    ("following you",               5.0, "harassment_of_other",    "other"),
    ("control you",                 6.0, "harassment_of_other",    "other"),
    ("you belong to me",            7.0, "harassment_of_other",    "other"),
    ("won't leave you alone",       7.0, "harassment_of_other",    "other"),
    ("watching you",                5.0, "harassment_of_other",    "other"),

    # Substance use
    ("cocaine",                     9.0, "substance_use",          "self"),
    ("heroin",                     10.0, "substance_use",          "self"),
    ("meth",                        9.0, "substance_use",          "self"),
    ("getting high",                5.0, "substance_use",          "self"),
    ("need a fix",                  7.0, "substance_use",          "self"),
    ("can't stop drinking",         7.0, "substance_use",          "self"),
    ("drunk every day",             6.0, "substance_use",          "self"),
    ("scoring drugs",               7.0, "substance_use",          "self"),

    # Social isolation
    ("stopped eating",              6.0, "isolation",              "self"),
    ("not sleeping",                5.0, "isolation",              "self"),
    ("can't get out of bed",        6.0, "isolation",              "self"),
    ("no friends left",             5.0, "isolation",              "self"),
    ("cutting people off",          6.0, "isolation",              "self"),
    ("disconnected from everything", 5.0, "isolation",             "self"),
    ("haven't left the house",      6.0, "isolation",              "self"),
]

# ── Human-readable labels for frontend display ────────────────────────────────
CATEGORY_LABELS = {
    "self_harm":               "Self-harm / Suicidal thoughts",
    "emotional_distress":      "Emotional distress / Mental health",
    "aggression_toward_other": "Threatening behaviour toward others",
    "harassment_of_other":     "Harassment or controlling behaviour",
    "substance_use":           "Substance use / Addiction signals",
    "isolation":               "Social isolation / Withdrawal",
}

# What this means for clinical context
CATEGORY_CLINICAL_NOTE = {
    "self_harm":
        "Person may be experiencing suicidal ideation or self-harm urges. "
        "Requires immediate clinical assessment.",
    "emotional_distress":
        "Person shows signs of psychological distress. "
        "Recommend mental health check-in and supportive intervention.",
    "aggression_toward_other":
        "Person is expressing violent or threatening language toward another individual. "
        "Safety of the recipient should be assessed.",
    "harassment_of_other":
        "Person is exhibiting controlling, manipulative, or harassing patterns "
        "directed at another individual.",
    "substance_use":
        "Signs of substance dependency or misuse. "
        "Consider referral for substance use support.",
    "isolation":
        "Person is showing signs of social withdrawal which may worsen mental health outcomes. "
        "Engagement and community support recommended.",
}

RISK_LEVEL_DESCRIPTIONS = {
    "critical": "Immediate clinical attention required. Multiple severe indicators present.",
    "high":     "Significant risk indicators present. Professional review strongly advised.",
    "medium":   "Moderate risk signals detected. Monitor closely and consider follow-up.",
    "low":      "Minor risk language present. Likely contextual — worth noting.",
    "none":     "No significant risk indicators detected.",
}


def _build_index():
    return [
        (re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE), phrase, weight, cat, direction)
        for phrase, weight, cat, direction in _REGISTRY
    ]

_INDEX = _build_index()


def _score_text(text: str):
    score = 0.0
    hits  = defaultdict(list)
    for pat, phrase, weight, cat, direction in _INDEX:
        if pat.search(text):
            score += weight
            hits[cat].append({"phrase": phrase, "direction": direction, "weight": weight})
    return score, dict(hits)


def _level(score: float) -> str:
    if score >= 20: return "critical"
    if score >= 12: return "high"
    if score >= 5:  return "medium"
    if score > 0:   return "low"
    return "none"


def _norm_score(raw: float) -> float:
    return round(min(raw / 30.0 * 100, 100), 1)


def risk_analysis(df: pd.DataFrame) -> dict:
    """
    Full risk analysis. Returns clear per-person breakdown:
    - who is at risk (self-directed signals)
    - who poses a risk to others (other-directed signals)
    - exact messages as evidence
    - whether it is escalating
    """
    text_df = df[~df["is_media"]].copy()

    # ── Per-person analysis ───────────────────────────────────────────────────
    per_person = []

    for sender in df["user"].unique():
        s_df    = text_df[text_df["user"] == sender].copy()
        s_text  = " ".join(s_df["message"].tolist())
        s_score, s_hits = _score_text(s_text)

        if s_score == 0:
            per_person.append({
                "person":                  sender,
                "risk_level":              "none",
                "risk_score_0_to_100":     0,
                "person_is_at_risk":       False,
                "person_poses_risk_to_others": False,
                "plain_english_summary":   f"{sender} shows no risk indicators in this chat.",
                "clinical_notes":          [],
                "categories":              [],
                "flagged_messages":        [],
                "risk_is_escalating":      False,
                "escalation_note":         "",
            })
            continue

        # Separate direction
        self_cats  = {}   # cat → [phrases]
        other_cats = {}

        for cat, hits in s_hits.items():
            for h in hits:
                if h["direction"] in ("self", "both"):
                    self_cats.setdefault(cat, []).append(h["phrase"])
                if h["direction"] in ("other", "both"):
                    other_cats.setdefault(cat, []).append(h["phrase"])

        person_is_at_risk           = bool(self_cats)
        person_poses_risk_to_others = bool(other_cats)
        level                       = _level(s_score)

        # Plain English summary
        summary_lines = []
        if person_is_at_risk:
            cat_labels = [CATEGORY_LABELS[c] for c in self_cats if c in CATEGORY_LABELS]
            summary_lines.append(
                f"⚠️ {sender} themselves may be at risk — signals detected: {', '.join(cat_labels)}."
            )
        if person_poses_risk_to_others:
            cat_labels = [CATEGORY_LABELS[c] for c in other_cats if c in CATEGORY_LABELS]
            summary_lines.append(
                f"🚨 {sender} is expressing language that may pose a risk to others — "
                f"signals: {', '.join(cat_labels)}."
            )

        # Clinical notes per category
        clinical_notes = []
        for cat in {**self_cats, **other_cats}:
            note = CATEGORY_CLINICAL_NOTE.get(cat, "")
            if note:
                clinical_notes.append({
                    "category":      cat,
                    "category_label": CATEGORY_LABELS.get(cat, cat),
                    "direction":     "self" if cat in self_cats else "toward others",
                    "clinical_note": note,
                    "phrases_found": list(set(
                        self_cats.get(cat, []) + other_cats.get(cat, [])
                    )),
                })

        # Flagged individual messages
        flagged = []
        for _, row in s_df.iterrows():
            msg = str(row["message"])
            msg_score, msg_hits = _score_text(msg)
            if msg_hits:
                flagged.append({
                    "date":              str(row["only_date"]),
                    "time":              str(row.get("time_str", "")),
                    "message_text":      msg[:300],
                    "risk_categories":   list(msg_hits.keys()),
                    "category_labels":   [CATEGORY_LABELS.get(c, c) for c in msg_hits.keys()],
                    "phrases_triggered": [p["phrase"] for hits in msg_hits.values() for p in hits],
                    "this_message_severity": _level(msg_score),
                })
            if len(flagged) >= 15:
                break

        # Escalation: compare first third vs last third of sender messages
        n3 = max(len(s_df) // 3, 1)
        sc_early, _ = _score_text(" ".join(s_df.iloc[:n3]["message"].tolist()))
        sc_late,  _ = _score_text(" ".join(s_df.iloc[-n3:]["message"].tolist()))
        escalating  = sc_late > sc_early * 1.5 and sc_late > 3
        esc_note    = (
            f"Risk language from {sender} appears to be INCREASING over time. "
            f"Early period score: {round(sc_early,1)}, Recent period score: {round(sc_late,1)}."
        ) if escalating else ""

        per_person.append({
            "person":                      sender,
            "risk_level":                  level,
            "risk_score_0_to_100":         _norm_score(s_score),
            "person_is_at_risk":           person_is_at_risk,
            "person_poses_risk_to_others": person_poses_risk_to_others,
            "plain_english_summary":       " ".join(summary_lines) or f"{sender}: {level} risk.",
            "self_risk_categories":        [
                {"category": c, "label": CATEGORY_LABELS.get(c, c), "phrases": ps}
                for c, ps in self_cats.items()
            ],
            "risk_to_others_categories":   [
                {"category": c, "label": CATEGORY_LABELS.get(c, c), "phrases": ps}
                for c, ps in other_cats.items()
            ],
            "clinical_notes":              clinical_notes,
            "flagged_messages":            flagged,
            "risk_is_escalating":          escalating,
            "escalation_note":             esc_note,
        })

    per_person.sort(key=lambda x: x["risk_score_0_to_100"], reverse=True)

    # ── Conversation-level overall ────────────────────────────────────────────
    all_text    = " ".join(text_df["message"].tolist())
    conv_score, _ = _score_text(all_text)
    conv_level  = _level(conv_score)

    mid   = len(text_df) // 2
    sc1, _ = _score_text(" ".join(text_df.iloc[:mid]["message"].tolist()))
    sc2, _ = _score_text(" ".join(text_df.iloc[mid:]["message"].tolist()))
    conv_esc = sc2 > sc1 * 1.5 and sc2 > 5

    # Overall human summary
    at_risk_persons = [p for p in per_person if p["risk_level"] != "none"]
    if not at_risk_persons:
        overall_summary = "✅ No significant risk indicators detected in this conversation."
    else:
        lines = [p["plain_english_summary"] for p in at_risk_persons]
        if conv_esc:
            lines.append("⚠️ Overall risk language in this conversation is increasing over time.")
        overall_summary = " ".join(lines)

    # Weekly risk timeline
    text_df["week"] = pd.to_datetime(text_df["only_date"]).dt.to_period("W").astype(str)
    weekly_risk = []
    for week, grp in text_df.groupby("week"):
        ws, _ = _score_text(" ".join(grp["message"].tolist()))
        if ws > 0:
            weekly_risk.append({
                "week":        week,
                "risk_score":  _norm_score(ws),
                "risk_level":  _level(ws),
            })

    return {
        "overall_risk_level":        conv_level,
        "overall_risk_score_0_to_100": _norm_score(conv_score),
        "overall_summary":           overall_summary,
        "risk_level_description":    RISK_LEVEL_DESCRIPTIONS.get(conv_level, ""),
        "conversation_escalating":   conv_esc,

        # The main section — per person breakdown
        "per_person_risk":           per_person,

        # For quick frontend display — who needs attention
        "persons_at_risk":           [
            p["person"] for p in per_person if p["person_is_at_risk"]
        ],
        "persons_posing_risk_to_others": [
            p["person"] for p in per_person if p["person_poses_risk_to_others"]
        ],

        "risk_timeline_weekly":      weekly_risk,
        "total_flagged_messages":    sum(len(p["flagged_messages"]) for p in per_person),
    }
