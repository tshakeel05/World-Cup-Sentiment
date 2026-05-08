"""
sentiment.py
------------
Performs VADER sentiment analysis on preprocessed Reddit text.

Outputs per row:
  - sentiment_pos   : positive score (0–1)
  - sentiment_neg   : negative score (0–1)
  - sentiment_neu   : neutral score  (0–1)
  - sentiment_compound : compound score (-1 to +1)
  - sentiment_label : 'positive' | 'negative' | 'neutral'

Aggregated metrics per team (via aggregate_by_team):
  - avg_compound
  - avg_positive
  - avg_negative
  - avg_neutral
  - comment_volume
"""

import logging
import os

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

_analyzer = SentimentIntensityAnalyzer()


def _label(compound: float) -> str:
    """Convert a VADER compound score to a human-readable label."""
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def analyze_text(text: str) -> dict:
    """
    Run VADER on a single string.

    Parameters
    ----------
    text : str
        Cleaned input text.

    Returns
    -------
    dict with keys: pos, neg, neu, compound, label
    """
    if not isinstance(text, str) or not text.strip():
        return {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0, "label": "neutral"}

    scores = _analyzer.polarity_scores(text)
    scores["label"] = _label(scores["compound"])
    return scores


def analyze_dataframe(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """
    Apply VADER sentiment analysis to every row of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a column of cleaned text.
    text_col : str
        Name of the column to analyse.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with four additional score columns and a label column.

    Raises
    ------
    KeyError
        If *text_col* is missing from *df*.
    """
    if text_col not in df.columns:
        raise KeyError(
            f"Column '{text_col}' not found. Available columns: {list(df.columns)}"
        )

    df = df.copy()

    scores = df[text_col].apply(analyze_text)
    df["sentiment_pos"] = scores.apply(lambda s: s["pos"])
    df["sentiment_neg"] = scores.apply(lambda s: s["neg"])
    df["sentiment_neu"] = scores.apply(lambda s: s["neu"])
    df["sentiment_compound"] = scores.apply(lambda s: s["compound"])
    df["sentiment_label"] = scores.apply(lambda s: s["label"])

    logger.info(
        "Sentiment analysis complete. %d rows analysed. "
        "Positive: %d | Negative: %d | Neutral: %d",
        len(df),
        (df["sentiment_label"] == "positive").sum(),
        (df["sentiment_label"] == "negative").sum(),
        (df["sentiment_label"] == "neutral").sum(),
    )
    return df


def aggregate_by_team(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment metrics per team.

    Expects a 'team' column (or 'teams' as a comma-separated string if a row
    maps to multiple teams).  Rows with no team assignment are skipped.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with sentiment scores and a 'team' column.

    Returns
    -------
    pd.DataFrame
        One row per team with average scores and comment volume.
    """
    if "team" not in df.columns:
        raise KeyError("'team' column missing. Run team_detection first.")

    # Explode multi-team rows
    exploded = df.copy()
    exploded["team"] = exploded["team"].str.split(",")
    exploded = exploded.explode("team")
    exploded["team"] = exploded["team"].str.strip()
    exploded = exploded[exploded["team"] != ""]

    agg = (
        exploded.groupby("team")
        .agg(
            avg_compound=("sentiment_compound", "mean"),
            avg_positive=("sentiment_pos", "mean"),
            avg_negative=("sentiment_neg", "mean"),
            avg_neutral=("sentiment_neu", "mean"),
            comment_volume=("sentiment_compound", "count"),
        )
        .reset_index()
    )

    logger.info("Aggregated sentiment for %d teams.", len(agg))
    return agg


def run_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Load a preprocessed CSV, run sentiment analysis, and save.

    Parameters
    ----------
    input_path : str
        Path to the preprocessed CSV (must contain 'clean_text').
    output_path : str
        Destination path for the sentiment-enriched CSV.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If *input_path* does not exist.
    """
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        logger.error("Processed data file not found: %s", input_path)
        raise

    df = analyze_dataframe(df)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Sentiment data saved to %s", output_path)
    return df


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    run_pipeline(
        input_path=os.path.join(base, "../data/processed/reddit_processed.csv"),
        output_path=os.path.join(base, "../data/processed/reddit_sentiment.csv"),
    )
