"""
test_chat.py
────────────────────────────────────────────────────────────
Standalone WhatsApp Chat Analyzer — NO server needed.

Usage:
    python test_chat.py whatsapp.txt
    python test_chat.py whatsapp.txt --user "Alice"
    python test_chat.py whatsapp.txt --full
"""

import sys
import os
import argparse

# ── make sure imports resolve from this folder ─────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.preprocessor    import preprocess
from models.analysis_model  import full_analysis, fetch_stats, get_participants, word_cloud_data, emoji_analysis, response_time_stats
from models.sentiment_model import aggregate_sentiment, analyze_message
from models.risk_model      import risk_analysis

# ── pretty print helpers ───────────────────────────────────────────────────────
W = 60

def sep(char="─"):  print(char * W)
def header(title):  sep("═"); print(f"  {title}"); sep("═")
def section(title): print(); sep(); print(f"  {title}"); sep()

COLORS = {
    "green":  "\033[92m", "red":    "\033[91m",
    "yellow": "\033[93m", "cyan":   "\033[96m",
    "bold":   "\033[1m",  "reset":  "\033[0m",
}
def c(text, color): return f"{COLORS.get(color,'')}{text}{COLORS['reset']}"


def print_stats(stats: dict):
    section("📊 BASIC STATS")
    items = [
        ("Total Messages",    stats["total_messages"],    "cyan"),
        ("Total Words",       stats["total_words"],       "cyan"),
        ("Media Shared",      stats["media_shared"],      "yellow"),
        ("Links Shared",      stats["links_shared"],      "yellow"),
        ("Deleted Messages",  stats["deleted_messages"],  "red"),
        ("Total Emojis",      stats["total_emojis"],      "green"),
        ("Avg Words/Message", stats["avg_words_per_msg"], "cyan"),
        ("Messages/Day",      stats["messages_per_day"],  "cyan"),
        ("Date Range (days)", stats["date_range_days"],   "cyan"),
    ]
    for label, val, color in items:
        print(f"  {label:<24} {c(str(val), color)}")


def print_participants(participants: list):
    section("👥 PARTICIPANTS")
    for i, p in enumerate(participants, 1):
        print(f"  {i}. {c(p, 'cyan')}")


def print_most_active(most_active: list):
    section("🏆 MOST ACTIVE SENDERS")
    for row in most_active[:10]:
        bar_len = int(row["percent"] / 2)
        bar = c("█" * bar_len, "green") + "░" * (50 - bar_len)
        print(f"  {row['user']:<20} {row['messages']:>5} msgs  {row['percent']:>5}%  {bar}")


def print_sentiment(sentiment: dict):
    section("😊 SENTIMENT ANALYSIS")
    agg = sentiment.get("aggregate", {})
    if not agg:
        print("  No sentiment data available.")
        return

    label = agg.get("overall_label", "neutral")
    color_map = {"positive": "green", "negative": "red", "neutral": "yellow"}
    score = agg.get("avg_score_0_100", 50)

    print(f"  Overall Label    : {c(label.upper(), color_map.get(label, 'cyan'))}")
    print(f"  Score (0–100)    : {c(str(score), 'cyan')}")
    print(f"  Avg Confidence   : {agg.get('avg_confidence', 0)}")
    print(f"  Volatility       : {agg.get('sentiment_volatility', 0)}")
    print(f"  Positive msgs    : {c(str(agg.get('positive_msgs',0)), 'green')}")
    print(f"  Negative msgs    : {c(str(agg.get('negative_msgs',0)), 'red')}")
    print(f"  Neutral msgs     : {agg.get('neutral_msgs',0)}")
    print(f"  Positivity ratio : {agg.get('positivity_ratio',0)}")

    tone_dist = agg.get("tone_distribution", {})
    if tone_dist:
        print(f"\n  Tone Distribution:")
        for tone, count in sorted(tone_dist.items(), key=lambda x: -x[1]):
            print(f"    {tone:<12} {count}")

    print(f"\n  Per-Sender Sentiment:")
    print(f"  {'Sender':<20} {'Label':<10} {'Score':>7} {'Volatility':>12}")
    sep()
    for s in sentiment.get("per_sender", []):
        lbl   = s.get("overall_label", "?")
        clr   = color_map.get(lbl, "cyan")
        score = s.get("avg_score_0_100", 0)
        vol   = s.get("sentiment_volatility", 0)
        print(f"  {s['sender']:<20} {c(lbl, clr):<10 if lbl else lbl:<10} {score:>7} {vol:>12}")


def print_activity(result: dict):
    section("📅 ACTIVITY — DAY OF WEEK")
    wa = result.get("week_activity", [])
    max_c = max((r["count"] for r in wa), default=1)
    for row in wa:
        bar_len = int(row["count"] / max_c * 40)
        bar     = c("█" * bar_len, "cyan") + "░" * (40 - bar_len)
        print(f"  {row['day']:<12} {row['count']:>5}  {bar}")

    section("🕐 ACTIVITY — HOURLY")
    ha = result.get("hour_activity", [])
    max_h = max((r["count"] for r in ha), default=1)
    for row in sorted(ha, key=lambda x: x["hour"]):
        bar_len = int(row["count"] / max_h * 40)
        bar     = c("█" * bar_len, "yellow")
        print(f"  {row['hour']:02d}:00  {row['count']:>5}  {bar}")


def print_words(words: list):
    section("💬 TOP 30 WORDS")
    max_c = max((w["count"] for w in words[:30]), default=1)
    for row in words[:30]:
        bar_len = int(row["count"] / max_c * 35)
        bar     = c("█" * bar_len, "green")
        print(f"  {row['word']:<20} {row['count']:>5}  {bar}")


def print_emojis(emojis: list):
    section("😂 TOP EMOJIS")
    for row in emojis[:15]:
        print(f"  {row['emoji']}  {row['count']:>5}")


def print_risk(risk: dict):
    section("⚠️  RISK ANALYSIS")
    level = risk.get("risk_level", "none")
    score = risk.get("risk_score_0_100", 0)
    esc   = risk.get("escalation_detected", False)

    color_map = {"high": "red", "medium": "yellow", "low": "green", "none": "cyan"}
    print(f"  Risk Level       : {c(level.upper(), color_map.get(level,'cyan'))}")
    print(f"  Risk Score       : {c(str(score), color_map.get(level,'cyan'))}/100")
    print(f"  Escalation       : {c('⚠ YES', 'red') if esc else c('✓ No', 'green')}")
    print(f"  Categories       : {risk.get('total_categories', 0)}")

    flags = risk.get("risk_flags", {})
    if flags:
        print(f"\n  Flagged keywords:")
        for cat, kws in flags.items():
            print(f"    {c(cat,'yellow'):<20} {', '.join(kws)}")

    fm = risk.get("flagged_messages", [])
    if fm:
        print(f"\n  Sample flagged messages ({len(fm)}):")
        for msg in fm[:5]:
            cats = ", ".join(msg.get("categories", []))
            print(f"    [{msg['date']}] {msg['user']} [{cats}]")
            print(f"      {msg['message'][:100]}")

    print(f"\n  Per-Sender Risk:")
    for s in risk.get("per_sender_risk", []):
        clr = color_map.get(s["risk_level"], "cyan")
        print(f"  {s['sender']:<22} {c(s['risk_level'],''):<8} score={c(str(s['risk_score']), clr)}")


def print_response_time(rt: list):
    section("⏱️  RESPONSE TIMES")
    if not rt:
        print("  No response time data.")
        return
    print(f"  {'Sender':<22} {'Avg (min)':>10} {'Median':>10} {'Fastest':>10}")
    sep()
    for row in rt:
        print(f"  {row['sender']:<22} {row['avg_response_min']:>10} {row['median_response_min']:>10} {row['fastest_response_min']:>10}")


def print_patterns(result: dict):
    section("🔁 CONVERSATION INITIATORS")
    for row in result.get("initiator_stats", []):
        print(f"  {row['sender']:<22} {row['initiations']:>5} starts  {row['percent']:>5}%")

    section("🔇 SILENT PERIODS (>24h)")
    sp = result.get("silent_periods", [])
    if sp:
        for row in sp[:10]:
            print(f"  {row['from']}  →  {row['to']}   ({row['gap_hours']}h)")
    else:
        print(c("  No long silent periods found.", "green"))

    ln = result.get("late_night_stats", {})
    if ln:
        section("🌙 LATE NIGHT MESSAGES (12AM–5AM)")
        print(f"  Total: {ln.get('total_late_night_msgs', 0)}")
        for s in ln.get("per_sender", []):
            print(f"  {s['sender']:<22} {s['count']}")


def print_vocab(vocab: list):
    section("📚 VOCABULARY RICHNESS")
    print(f"  {'Sender':<22} {'Unique Words':>14} {'Total':>10} {'Diversity':>12}")
    sep()
    for row in vocab:
        print(f"  {row['sender']:<22} {row['unique_words']:>14} {row['total_words']:>10} {row['lexical_diversity']:>12}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WhatsApp Chat Analyzer CLI")
    parser.add_argument("file",           help="Path to WhatsApp .txt export")
    parser.add_argument("--user",         default="Overall", help="Filter by sender name")
    parser.add_argument("--full",         action="store_true", help="Show all sections")
    parser.add_argument("--no-sentiment", action="store_true", help="Skip sentiment (faster)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(c(f"❌ File not found: {args.file}", "red"))
        sys.exit(1)

    print()
    header(f"💬 WhatsApp Chat Analyzer  v2.0")
    print(f"  File : {args.file}")
    print(f"  User : {args.user}")

    # ── Parse ──────────────────────────────────────────────────────────────
    print(f"\n  {c('Parsing...', 'yellow')}", end="", flush=True)
    with open(args.file, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    try:
        df = preprocess(raw)
        print(c(f" ✓  ({len(df):,} messages parsed)", "green"))
    except ValueError as e:
        print(c(f"\n❌ Parse error: {e}", "red"))
        sys.exit(1)

    # ── Stats ──────────────────────────────────────────────────────────────
    print(f"  {c('Computing stats...', 'yellow')}", end="", flush=True)
    stats = fetch_stats(df, args.user)
    print(c(" ✓", "green"))

    # ── Full analysis ──────────────────────────────────────────────────────
    if not args.no_sentiment:
        print(f"  {c('Running sentiment (may take a moment)...', 'yellow')}", end="", flush=True)

    result = full_analysis(df, args.user)
    if not args.no_sentiment:
        print(c(" ✓", "green"))

    # ── Print sections ─────────────────────────────────────────────────────
    print_stats(result["stats"])
    print_participants(result["participants"])
    print_most_active(result["most_active"])

    if not args.no_sentiment:
        print_sentiment(result["sentiment"])

    print_activity(result)
    print_words(result["top_words"])
    print_emojis(result["top_emojis"])
    print_vocab(result["vocabulary_richness"])
    print_response_time(result["response_time"])
    print_patterns(result)
    print_risk(result["risk"])

    sep("═")
    print(c("  ✅ Analysis complete!", "green"))
    sep("═")
    print()


if __name__ == "__main__":
    main()
