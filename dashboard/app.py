"""
app.py
------
World Cup Reddit Sentiment Dashboard — Streamlit frontend.

Loads the processed/final CSV (or falls back to raw data with live
preprocessing) and renders:
  - Team selector
  - Sentiment trend chart (line)
  - Sentiment distribution (histogram)
  - Comment volume over time (bar)
  - Top keywords (bar)
  - Team comparison (bar)
  - Recent comments table
"""

import os
import sys
import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from collections import Counter

# Allow importing from src/ whether running from repo root or dashboard/
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_FINAL = os.path.join(_ROOT, "data", "processed", "reddit_final.csv")
DATA_SENTIMENT = os.path.join(_ROOT, "data", "processed", "reddit_sentiment.csv")
DATA_RAW = os.path.join(_ROOT, "data", "raw", "reddit_data.csv")

STOPWORDS_EXTRA = {
    "world", "cup", "game", "match", "team", "played", "play", "playing",
    "tournament", "really", "got", "would", "could", "going", "even",
    "still", "also", "one", "two", "like", "just", "get", "win", "won",
    "great", "good", "bad", "vs", "ever",
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data…")
def load_data() -> pd.DataFrame:
    """
    Load the fully-processed CSV.  Falls back to building it on the fly
    from raw data if the processed file does not exist.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: text, clean_text, timestamp, subreddit,
        score, sentiment_compound, sentiment_label, team, etc.

    Raises
    ------
    FileNotFoundError
        If no source data can be found at all.
    """
    if os.path.exists(DATA_FINAL):
        df = pd.read_csv(DATA_FINAL)
    elif os.path.exists(DATA_SENTIMENT):
        from team_detection import detect_teams_dataframe
        df = pd.read_csv(DATA_SENTIMENT)
        df = detect_teams_dataframe(df)
    elif os.path.exists(DATA_RAW):
        from preprocess import preprocess_dataframe
        from sentiment import analyze_dataframe
        from team_detection import detect_teams_dataframe

        df = pd.read_csv(DATA_RAW)
        df = preprocess_dataframe(df)
        df = analyze_dataframe(df)
        df = detect_teams_dataframe(df)
    else:
        raise FileNotFoundError(
            "No data files found. Run src/collect_data.py first, "
            "or place sample data in data/raw/reddit_data.csv."
        )

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = df["timestamp"].dt.date

    return df


def explode_teams(df: pd.DataFrame) -> pd.DataFrame:
    """Expand rows so each team mention gets its own row."""
    df = df.copy()
    df["team"] = df["team"].fillna("").astype(str)
    df = df[df["team"] != ""]
    df["team"] = df["team"].str.split(",")
    df = df.explode("team")
    df["team"] = df["team"].str.strip()
    return df[df["team"] != ""]


def top_keywords(texts: pd.Series, n: int = 15) -> pd.DataFrame:
    """Return a DataFrame of the top *n* keywords from a Series of strings."""
    all_words: list[str] = []
    for text in texts.dropna():
        all_words.extend(str(text).split())

    filtered = [w for w in all_words if len(w) > 3 and w not in STOPWORDS_EXTRA]
    counts = Counter(filtered).most_common(n)
    return pd.DataFrame(counts, columns=["word", "count"])


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="⚽ World Cup Sentiment Dashboard",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ World Cup Reddit Sentiment Dashboard")
st.caption("Sentiment analysis of Reddit discussions from r/soccer & r/worldcup")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
try:
    df_all = load_data()
except FileNotFoundError as exc:
    st.error(f"**Data not found:** {exc}")
    st.stop()
except Exception as exc:
    st.error(f"**Unexpected error loading data:** {exc}")
    logger.exception("Failed to load data")
    st.stop()

if df_all.empty:
    st.warning("The data file is empty. Please collect data first.")
    st.stop()

df_exploded = explode_teams(df_all)
all_teams = sorted(df_exploded["team"].unique().tolist())

# ---------------------------------------------------------------------------
# Sidebar — filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")

    selected_team = st.selectbox(
        "Select a team",
        options=all_teams,
        index=all_teams.index("Argentina") if "Argentina" in all_teams else 0,
    )

    subreddits = ["All"] + sorted(df_all["subreddit"].dropna().unique().tolist())
    selected_sub = st.selectbox("Subreddit", options=subreddits)

    st.markdown("---")
    st.metric("Total records", len(df_all))
    st.metric("Teams detected", len(all_teams))

# ---------------------------------------------------------------------------
# Filter by team
# ---------------------------------------------------------------------------
df_team = df_exploded[df_exploded["team"] == selected_team].copy()
if selected_sub != "All":
    df_team = df_team[df_team["subreddit"] == selected_sub]

if df_team.empty:
    st.warning(f"No data found for **{selected_team}**. Try a different team or subreddit.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
avg_compound = df_team["sentiment_compound"].mean()
pos_pct = (df_team["sentiment_label"] == "positive").mean() * 100
neg_pct = (df_team["sentiment_label"] == "negative").mean() * 100
comment_count = len(df_team)

col1.metric("Avg Compound Score", f"{avg_compound:.3f}")
col2.metric("Positive %", f"{pos_pct:.1f}%")
col3.metric("Negative %", f"{neg_pct:.1f}%")
col4.metric("Comment Volume", comment_count)

st.markdown("---")

# ---------------------------------------------------------------------------
# Row 1: Sentiment trend + Sentiment distribution
# ---------------------------------------------------------------------------
row1_left, row1_right = st.columns([2, 1])

with row1_left:
    st.subheader(f"📈 Sentiment Trend — {selected_team}")
    trend = (
        df_team.groupby("date")["sentiment_compound"]
        .mean()
        .reset_index()
        .rename(columns={"sentiment_compound": "avg_compound"})
    )
    if not trend.empty:
        fig_trend = px.line(
            trend,
            x="date",
            y="avg_compound",
            markers=True,
            labels={"date": "Date", "avg_compound": "Avg Compound Score"},
            title=f"Daily Average Sentiment — {selected_team}",
        )
        fig_trend.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_trend.update_layout(yaxis_range=[-1, 1])
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("Not enough data to plot trend.")

with row1_right:
    st.subheader("📊 Sentiment Distribution")
    fig_hist = px.histogram(
        df_team,
        x="sentiment_compound",
        nbins=20,
        color_discrete_sequence=["#636EFA"],
        labels={"sentiment_compound": "Compound Score", "count": "Count"},
        title="Compound Score Distribution",
    )
    fig_hist.update_layout(bargap=0.05)
    st.plotly_chart(fig_hist, use_container_width=True)

# ---------------------------------------------------------------------------
# Row 2: Comment volume over time + Top keywords
# ---------------------------------------------------------------------------
row2_left, row2_right = st.columns([2, 1])

with row2_left:
    st.subheader("💬 Comment Volume Over Time")
    volume = df_team.groupby("date").size().reset_index(name="count")
    fig_vol = px.bar(
        volume,
        x="date",
        y="count",
        labels={"date": "Date", "count": "Number of Comments"},
        title=f"Daily Comment Volume — {selected_team}",
        color_discrete_sequence=["#00CC96"],
    )
    st.plotly_chart(fig_vol, use_container_width=True)

with row2_right:
    st.subheader("🔤 Top Keywords")
    text_col = "clean_text" if "clean_text" in df_team.columns else "text"
    kw_df = top_keywords(df_team[text_col])
    if not kw_df.empty:
        fig_kw = px.bar(
            kw_df,
            x="count",
            y="word",
            orientation="h",
            labels={"count": "Frequency", "word": "Keyword"},
            title=f"Top Keywords — {selected_team}",
            color_discrete_sequence=["#EF553B"],
        )
        fig_kw.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_kw, use_container_width=True)
    else:
        st.info("Not enough text to extract keywords.")

# ---------------------------------------------------------------------------
# Row 3: Team comparison (all teams)
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("🌍 Team Sentiment Comparison")

team_agg = (
    df_exploded.groupby("team")
    .agg(
        avg_compound=("sentiment_compound", "mean"),
        comment_volume=("sentiment_compound", "count"),
    )
    .reset_index()
    .sort_values("avg_compound", ascending=False)
)

fig_compare = px.bar(
    team_agg,
    x="team",
    y="avg_compound",
    color="avg_compound",
    color_continuous_scale="RdYlGn",
    labels={"team": "Team", "avg_compound": "Avg Compound Score"},
    title="Average Sentiment Score by Team",
    text_auto=".2f",
)
fig_compare.update_layout(coloraxis_showscale=False, xaxis_tickangle=-30)
st.plotly_chart(fig_compare, use_container_width=True)

# ---------------------------------------------------------------------------
# Row 4: Recent comments
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader(f"🗨️ Recent Comments — {selected_team}")

n_comments = st.slider("Number of comments to show", min_value=5, max_value=50, value=10)

recent = (
    df_team.sort_values("timestamp", ascending=False)
    .head(n_comments)[["timestamp", "subreddit", "text", "sentiment_label", "sentiment_compound", "score"]]
    .rename(
        columns={
            "timestamp": "Date",
            "subreddit": "Subreddit",
            "text": "Comment",
            "sentiment_label": "Sentiment",
            "sentiment_compound": "Score",
            "score": "Upvotes",
        }
    )
)

# Colour-code sentiment
def _colour_sentiment(val: str) -> str:
    colours = {"positive": "background-color: #d4edda", "negative": "background-color: #f8d7da"}
    return colours.get(val, "")

st.dataframe(
    recent.style.applymap(_colour_sentiment, subset=["Sentiment"]),
    use_container_width=True,
    hide_index=True,
)

st.caption("Data sourced from Reddit via PRAW | Sentiment analysis powered by VADER")
