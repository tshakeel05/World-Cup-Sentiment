"""
preprocess.py
-------------
Cleans raw Reddit text before sentiment analysis.

Steps applied (as per spec):
  1. Lowercase
  2. Remove URLs
  3. Remove punctuation
  4. Remove emojis
  5. Remove stopwords (NLTK English)
  6. Normalize whitespace
"""

import re
import logging

import pandas as pd
import nltk
from nltk.corpus import stopwords

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Download required NLTK resources on first run
def _ensure_nltk_resources() -> None:
    for resource in ("stopwords", "punkt"):
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            logger.info("Downloading NLTK resource: %s", resource)
            nltk.download(resource, quiet=True)


_ensure_nltk_resources()

_STOP_WORDS = set(stopwords.words("english"))

# Regex patterns compiled once for performance
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002700-\U000027BF"  # dingbats
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U00002600-\U000026FF"  # miscellaneous symbols
    "]+",
    flags=re.UNICODE,
)
_PUNCT_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Apply the full cleaning pipeline to a single string.

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    str
        Cleaned text, or empty string if input is not a string.
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()
    # 2. Remove URLs
    text = _URL_RE.sub(" ", text)
    # 3. Remove emojis
    text = _EMOJI_RE.sub(" ", text)
    # 4. Remove punctuation
    text = _PUNCT_RE.sub(" ", text)
    # 5. Remove stopwords
    tokens = text.split()
    tokens = [t for t in tokens if t not in _STOP_WORDS]
    text = " ".join(tokens)
    # 6. Normalize whitespace
    text = _WHITESPACE_RE.sub(" ", text).strip()

    return text


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Apply clean_text to every row of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing raw text.
    text_col : str
        Name of the column that holds raw text.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional 'clean_text' column.

    Raises
    ------
    KeyError
        If *text_col* is not present in *df*.
    """
    if text_col not in df.columns:
        raise KeyError(f"Column '{text_col}' not found in DataFrame. Available: {list(df.columns)}")

    df = df.copy()
    df["clean_text"] = df[text_col].apply(clean_text)
    logger.info("Preprocessing complete. %d rows processed.", len(df))
    return df


def load_and_preprocess(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Load a raw CSV, preprocess it, and save the result.

    Parameters
    ----------
    input_path : str
        Path to the raw CSV file.
    output_path : str
        Path to write the processed CSV file.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame.

    Raises
    ------
    FileNotFoundError
        If *input_path* does not exist.
    """
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        logger.error("Input file not found: %s", input_path)
        raise

    df = preprocess_dataframe(df)
    df.to_csv(output_path, index=False)
    logger.info("Processed data saved to %s", output_path)
    return df


if __name__ == "__main__":
    import os

    base = os.path.dirname(__file__)
    load_and_preprocess(
        input_path=os.path.join(base, "../data/raw/reddit_data.csv"),
        output_path=os.path.join(base, "../data/processed/reddit_processed.csv"),
    )
