"""
streamlit_app.py
────────────────────────────────────────────────────────────────
WhatsApp Chat Analyzer — Standalone Streamlit UI
No API server needed. Upload .txt → instant analysis.

Run:
    cd module_1_chat
    streamlit run streamlit_app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"]        { background:#1e1e2e; }
.block-container                 { padding-top: 1.5rem; }
div[data-testid="metric-container"] {
    background:#1e1e2e; border:1px solid #313244;
    border-radius:10px; padding:12px;
}
</style>
""", unsafe_allow_html=True)

PLOT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#cdd6f4",
    margin=dict(l=10, r=10, t=30, b=10),
)


# ── Model loader (cached once) ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load():
    from models.preprocessor    import preprocess
    from models.analysis_model  import full_analysis, sentiment_timeline
    from models.sentiment_model import analyze_message
    return preprocess, full_analysis, sentiment_timeline, analyze_message


# ── Analysis runner (cached per file + user) ──────────────────────────────────
@st.cache_data(show_spinner=False)
def _analyze(raw_bytes: bytes, user: str):
    preprocess, full_analysis, _, _ = _load()
    df     = preprocess(raw_bytes.decode("utf-8", errors="ignore"))
    result = full_analysis(df, user)
    df_json = df.to_json(orient="records", date_format="iso")
    return result, df_json, sorted(df["user"].unique().tolist())


@st.cache_data(show_spinner=False)
def _timeline(df_json: str):
    import pandas as pd
    _, _, sentiment_timeline, _ = _load()
    df = pd.read_json(df_json, orient="records")
    df["is_media"]  = df["is_media"].astype(bool)
    df["only_date"] = pd.to_datetime(df["only_date"]).dt.date
    df["date"]      = pd.to_datetime(df["date"])
    return sentiment_timeline(df)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("💬 WhatsApp Analyzer")
    st.divider()

    uploaded = st.file_uploader(
        "Upload WhatsApp .txt export",
        type=["txt"],
        help="Open chat → ⋮ → More → Export Chat → Without Media",
    )

    # Pre-read participants so the dropdown works before full analysis
    user_filter = "Overall"
    if uploaded:
        try:
            raw_bytes = uploaded.read()
            uploaded.seek(0)
            preprocess, _, _, _ = _load()
            df_preview   = preprocess(raw_bytes.decode("utf-8", errors="ignore"))
            participants = ["Overall"] + sorted(df_preview["user"].unique().tolist())
            user_filter  = st.selectbox("Filter by sender", participants)
        except Exception:
            user_filter = st.text_input("Filter by sender", "Overall")
            raw_bytes   = uploaded.read() if uploaded else b""

        run_btn = st.button("🚀 Run Analysis", use_container_width=True, type="primary")
    else:
        run_btn = False

    st.divider()
    st.markdown("#### 🧪 Quick Message Test")
    msg_test = st.text_area("Type any message", height=80, placeholder="e.g. I'm so happy today!")
    if st.button("Analyze message", use_container_width=True) and msg_test.strip():
        try:
            _, _, _, analyze_message = _load()
            a = analyze_message(msg_test)
            icons = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}
            st.markdown(f"**{icons.get(a['label'],'⚪')} {a['label'].upper()}** · {a['tone']}")
            st.progress(int(a["score_0_100"]), text=f"Score: {a['score_0_100']} / 100")
            st.caption(f"Confidence: {a['confidence']} · Compound: {a['ensemble_compound']}")
        except Exception as e:
            st.error(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# LANDING
# ══════════════════════════════════════════════════════════════════════════════
if not uploaded:
    st.markdown("## 💬 WhatsApp Chat Analyzer")
    st.markdown("""
Upload your WhatsApp `.txt` export using the sidebar to get a complete analysis.

**How to export a WhatsApp chat:**
> Open any chat → tap ⋮ (three dots) → **More** → **Export Chat** → **Without Media**

**What you'll get:**
- 📊 Message stats, media, links, deleted messages
- 😊 Sentiment analysis with tone detection (joy / anger / sadness / fear)
- 📅 Activity heatmaps — hourly, daily, weekly, monthly
- 💬 Top words and emoji frequency
- ⚡ Response times, conversation initiators, silent periods
- ⚠️ Risk scoring with per-sender breakdown and escalation detection
    """)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# RUN / LOAD ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
# Re-read bytes (file_uploader resets seek on re-render)
uploaded.seek(0)
raw_bytes = uploaded.read()

cache_key = f"{uploaded.name}_{len(raw_bytes)}_{user_filter}"

if run_btn or "cache_key" not in st.session_state or st.session_state.cache_key != cache_key:
    with st.spinner("Parsing and analysing your chat… ⏳"):
        try:
            result, df_json, senders = _analyze(raw_bytes, user_filter)
            st.session_state.result    = result
            st.session_state.df_json   = df_json
            st.session_state.cache_key = cache_key
        except ValueError as e:
            st.error(f"❌ Could not parse this file: {e}")
            st.info("Make sure you exported **without media** from WhatsApp.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Analysis error: {e}")
            st.stop()

result  = st.session_state.result
df_json = st.session_state.df_json
stats   = result.get("stats", {})


# ── Top metric bar ─────────────────────────────────────────────────────────────
st.markdown(f"### `{uploaded.name}` — **{user_filter}**")

c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)
c1.metric("💬 Messages",    f"{stats.get('total_messages',0):,}")
c2.metric("📝 Words",       f"{stats.get('total_words',0):,}")
c3.metric("🖼️ Media",       f"{stats.get('media_shared',0):,}")
c4.metric("🔗 Links",       f"{stats.get('links_shared',0):,}")
c5.metric("🗑️ Deleted",     f"{stats.get('deleted_messages',0):,}")
c6.metric("😂 Emojis",      f"{stats.get('total_emojis',0):,}")
c7.metric("📨 Msgs/Day",    stats.get("messages_per_day",0))
c8.metric("📆 Days",        f"{stats.get('date_range_days',0):,}")

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
T = st.tabs(["📊 Overview","😊 Sentiment","📅 Activity","💬 Words & Emojis","⚡ Patterns","⚠️ Risk","🔍 Raw"])


# ─── TAB 1: OVERVIEW ──────────────────────────────────────────────────────────
with T[0]:
    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown("#### Most Active Senders")
        ma = result.get("most_active", [])
        if ma:
            df_ma = pd.DataFrame(ma)
            fig = px.bar(df_ma, x="messages", y="user", orientation="h",
                         color="messages", color_continuous_scale="Blues", text="percent")
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(**PLOT, height=max(280, len(df_ma)*38),
                              coloraxis_showscale=False,
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### Participants")
        for p in result.get("participants", []):
            st.markdown(f"- `{p}`")

    st.markdown("#### Daily Message Count")
    dt = result.get("daily_timeline", [])
    if dt:
        df_dt = pd.DataFrame(dt)
        df_dt["date"] = pd.to_datetime(df_dt["date"])
        fig = px.area(df_dt, x="date", y="count", color_discrete_sequence=["#89b4fa"])
        fig.update_layout(**PLOT, height=220)
        st.plotly_chart(fig, use_container_width=True)


# ─── TAB 2: SENTIMENT ─────────────────────────────────────────────────────────
with T[1]:
    sentiment = result.get("sentiment", {})
    agg       = sentiment.get("aggregate", {})

    if not agg:
        st.info("No sentiment data.")
    else:
        lmap = {"positive":"🟢","negative":"🔴","neutral":"🟡"}
        label = agg.get("overall_label", "neutral")
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Overall",     f"{lmap.get(label,'')} {label.upper()}")
        m2.metric("Score",       f"{agg.get('avg_score_0_100',50)} / 100")
        m3.metric("Confidence",  agg.get("avg_confidence", 0))
        m4.metric("Volatility",  agg.get("sentiment_volatility", 0))

        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Message breakdown**")
            fig = go.Figure(go.Pie(
                labels=["Positive","Negative","Neutral"],
                values=[agg.get("positive_msgs",0), agg.get("negative_msgs",0), agg.get("neutral_msgs",0)],
                hole=0.5, marker_colors=["#a6e3a1","#f38ba8","#fab387"],
            ))
            fig.update_layout(**PLOT, height=290, showlegend=True,
                              legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig, use_container_width=True)

        with d2:
            tone_dist = agg.get("tone_distribution", {})
            if tone_dist:
                st.markdown("**Tone distribution**")
                df_t = pd.DataFrame(list(tone_dist.items()), columns=["Tone","Count"])
                fig  = px.bar(df_t, x="Tone", y="Count", color="Tone",
                              color_discrete_sequence=px.colors.qualitative.Pastel)
                fig.update_layout(**PLOT, height=290, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Per-Sender")
        per_s = sentiment.get("per_sender", [])
        if per_s:
            df_ps = pd.DataFrame(per_s)
            want  = ["sender","overall_label","avg_score_0_100","avg_confidence",
                     "positivity_ratio","negativity_ratio","sentiment_volatility","sample_size"]
            st.dataframe(df_ps[[c for c in want if c in df_ps.columns]],
                         use_container_width=True, hide_index=True)

        st.markdown("#### Daily Sentiment Timeline")
        if st.button("Load timeline", key="tl"):
            with st.spinner("Computing…"):
                tl = _timeline(df_json)
                if tl:
                    df_tl = pd.DataFrame(tl)
                    df_tl["date"] = pd.to_datetime(df_tl["date"])
                    fig = px.line(df_tl, x="date", y="avg_sentiment",
                                  color_discrete_sequence=["#89b4fa"])
                    fig.add_hline(y=50, line_dash="dot", line_color="#585b70",
                                  annotation_text="Neutral")
                    fig.update_layout(**PLOT, height=260, yaxis_range=[0,100])
                    st.plotly_chart(fig, use_container_width=True)


# ─── TAB 3: ACTIVITY ──────────────────────────────────────────────────────────
with T[2]:
    a1, a2 = st.columns(2)

    with a1:
        st.markdown("**Monthly**")
        mt = result.get("monthly_timeline", [])
        if mt:
            df_mt = pd.DataFrame(mt)
            fig   = px.bar(df_mt, x="period", y="count",
                           color_discrete_sequence=["#89dceb"])
            fig.update_layout(**PLOT, height=270, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

    with a2:
        st.markdown("**Day of Week**")
        wa = result.get("week_activity", [])
        if wa:
            df_wa = pd.DataFrame(wa)
            fig   = px.bar(df_wa, x="day", y="count",
                           color_discrete_sequence=["#cba6f7"])
            fig.update_layout(**PLOT, height=270)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Hourly (24h)**")
    ha = result.get("hour_activity", [])
    if ha:
        df_ha = pd.DataFrame(ha)
        fig   = px.bar(df_ha, x="hour", y="count",
                       color_discrete_sequence=["#f9e2af"])
        fig.update_layout(**PLOT, height=230,
                          xaxis=dict(tickmode="linear", dtick=1))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**🔥 Heatmap (Day × Hour)**")
    hm = result.get("heatmap_7x24", [])
    if hm:
        days = [r["day"] for r in hm]
        z    = [[r["hours"].get(str(h), 0) for h in range(24)] for r in hm]
        fig  = go.Figure(go.Heatmap(
            z=z, x=[f"{h:02d}:00" for h in range(24)], y=days,
            colorscale="Blues",
        ))
        fig.update_layout(**PLOT, height=290)
        st.plotly_chart(fig, use_container_width=True)


# ─── TAB 4: WORDS & EMOJIS ────────────────────────────────────────────────────
with T[3]:
    st.markdown("#### Top Words")
    words = result.get("top_words", [])
    if words:
        df_w = pd.DataFrame(words).head(40)
        fig  = px.bar(df_w, x="count", y="word", orientation="h",
                      color="count", color_continuous_scale="Teal")
        fig.update_layout(**PLOT, height=max(380, len(df_w)*16),
                          yaxis=dict(autorange="reversed"),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    w1, w2 = st.columns(2)
    with w1:
        st.markdown("#### Emojis")
        emojis = result.get("top_emojis", [])
        if emojis:
            df_e = pd.DataFrame(emojis).head(15)
            fig  = px.bar(df_e, x="emoji", y="count",
                          color_discrete_sequence=["#f38ba8"])
            fig.update_layout(**PLOT, height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No emojis found.")

    with w2:
        st.markdown("#### Vocabulary Richness")
        vocab = result.get("vocabulary_richness", [])
        if vocab:
            df_v = pd.DataFrame(vocab)
            fig  = px.bar(df_v, x="sender", y="lexical_diversity",
                          color="unique_words", color_continuous_scale="Purples",
                          hover_data=["total_words","unique_words"])
            fig.update_layout(**PLOT, height=300)
            st.plotly_chart(fig, use_container_width=True)


# ─── TAB 5: PATTERNS ──────────────────────────────────────────────────────────
with T[4]:
    p1, p2 = st.columns(2)

    with p1:
        st.markdown("**Conversation Initiators**")
        init = result.get("initiator_stats", [])
        if init:
            df_i = pd.DataFrame(init)
            fig  = px.pie(df_i, names="sender", values="initiations", hole=0.45,
                          color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(**PLOT, height=300)
            st.plotly_chart(fig, use_container_width=True)

    with p2:
        st.markdown("**Avg Response Time (min)**")
        rt = result.get("response_time", [])
        if rt:
            df_rt = pd.DataFrame(rt)
            fig   = px.bar(df_rt, x="sender", y="avg_response_min",
                           color_discrete_sequence=["#89dceb"],
                           hover_data=["median_response_min","fastest_response_min"])
            fig.update_layout(**PLOT, height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for response times.")

    q1, q2 = st.columns(2)
    with q1:
        st.markdown("**Late Night Messages (12AM–5AM)**")
        ln     = result.get("late_night_stats", {})
        ln_tot = ln.get("total_late_night_msgs", 0)
        st.metric("Total", ln_tot)
        if ln_tot and ln.get("per_sender"):
            st.dataframe(pd.DataFrame(ln["per_sender"]),
                         use_container_width=True, hide_index=True)

    with q2:
        st.markdown("**Silent Periods (>24h gap)**")
        sp = result.get("silent_periods", [])
        if sp:
            st.dataframe(pd.DataFrame(sp), use_container_width=True, hide_index=True)
        else:
            st.success("No long silent periods.")


# ─── TAB 6: RISK ──────────────────────────────────────────────────────────────
with T[5]:
    risk  = result.get("risk", {})
    level = risk.get("risk_level", "none")
    score = risk.get("risk_score_0_100", 0)
    esc   = risk.get("escalation_detected", False)

    ico = {"high":"🔴","medium":"🟠","low":"🟡","none":"🟢"}
    r1,r2,r3 = st.columns(3)
    r1.metric("Risk Level", f"{ico.get(level,'⚪')} {level.upper()}")
    r2.metric("Risk Score", f"{score} / 100")
    r3.metric("Escalation", "⚠️ YES" if esc else "✅ No")

    flags = risk.get("risk_flags", {})
    if flags:
        st.markdown("**Flagged keywords:**")
        for cat, kws in flags.items():
            st.markdown(f"- **{cat}**: " + ", ".join(f"`{k}`" for k in kws))
    else:
        st.success("✅ No risk keywords detected.")

    psr = risk.get("per_sender_risk", [])
    if psr:
        st.markdown("#### Per-Sender Risk")
        df_sr = pd.DataFrame(psr)
        fig   = px.bar(df_sr, x="sender", y="risk_score", color="risk_level",
                       color_discrete_map={
                           "high":"#f38ba8","medium":"#fab387",
                           "low":"#a6e3a1","none":"#585b70"
                       },
                       hover_data=["categories"])
        fig.update_layout(**PLOT, height=270)
        st.plotly_chart(fig, use_container_width=True)

    fm = risk.get("flagged_messages", [])
    if fm:
        st.markdown(f"#### Flagged Messages ({len(fm)})")
        for msg in fm:
            cats = ", ".join(msg.get("categories", []))
            with st.expander(f"[{msg['date']}]  {msg['user']}  —  {cats}"):
                st.write(msg["message"])


# ─── TAB 7: RAW ───────────────────────────────────────────────────────────────
with T[6]:
    st.markdown("#### Full Analysis Output")
    st.json(result, expanded=False)
