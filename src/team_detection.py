"""
team_detection.py
-----------------
Detects World Cup team mentions in Reddit text using keyword matching.

A comment can belong to multiple teams.  The result is stored in a
comma-separated 'team' column (e.g. "Brazil,Argentina").

Teams and their keyword aliases follow the 2022 FIFA World Cup squads.
"""

import logging
import re

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Team keyword mapping
# Each key is the canonical team name; the list contains keywords / aliases
# that indicate a mention of that team (all matched case-insensitively).
# ---------------------------------------------------------------------------
TEAM_KEYWORDS: dict[str, list[str]] = {
    "Brazil": ["brazil", "brasil", "seleção", "selecao", "neymar", "vinicius", "richarlison"],
    "Argentina": ["argentina", "messi", "albiceleste", "di maria", "lautaro"],
    "France": ["france", "les bleus", "mbappe", "giroud", "griezmann"],
    "England": ["england", "three lions", "kane", "rashford", "bellingham"],
    "Spain": ["spain", "españa", "la roja", "pedri", "morata", "busquets"],
    "Germany": ["germany", "deutschland", "die mannschaft", "muller", "neuer", "gnabry"],
    "Portugal": ["portugal", "ronaldo", "cr7", "bernardo silva", "felix"],
    "Netherlands": ["netherlands", "holland", "oranje", "van dijk", "depay", "dumfries"],
    "Belgium": ["belgium", "red devils", "de bruyne", "lukaku", "courtois"],
    "Uruguay": ["uruguay", "celeste", "suarez", "cavani", "valverde"],
    "Croatia": ["croatia", "hrvatska", "modric", "perisic", "gvardiol"],
    "Morocco": ["morocco", "maroc", "atlas lions", "ziyech", "hakimi", "en-nesyri"],
    "Senegal": ["senegal", "lions of teranga", "mane", "koulibaly", "dia"],
    "USA": ["usa", "united states", "usmnt", "pulisic", "weah"],
    "Mexico": ["mexico", "el tri", "lozano", "jimenez"],
    "Japan": ["japan", "samurai blue", "minamino", "doan", "kubo"],
    "South Korea": ["south korea", "korea", "son heung", "hwang"],
    "Australia": ["australia", "socceroos", "leckie", "degenek"],
    "Ghana": ["ghana", "black stars", "ayew", "kudus"],
    "Cameroon": ["cameroon", "indomitable lions", "choupo-moting", "aboubakar"],
    "Ecuador": ["ecuador", "tricolor", "estupinan", "valencia"],
    "Saudi Arabia": ["saudi arabia", "green falcons", "al-dawsari", "al-shahrani"],
    "Iran": ["iran", "team melli", "taremi", "jahanbakhsh"],
    "Qatar": ["qatar", "host", "al annabi"],
    "Serbia": ["serbia", "orlovi", "vlahovic", "mitrovic"],
    "Switzerland": ["switzerland", "nati", "xhaka", "shaqiri"],
    "Poland": ["poland", "bialo-czerwoni", "lewandowski", "szczesny"],
    "Denmark": ["denmark", "danish dynamite", "eriksen", "hojbjerg"],
    "Tunisia": ["tunisia", "eagles of carthage", "khazri", "maaloul"],
    "Costa Rica": ["costa rica", "ticos", "navas"],
    "Canada": ["canada", "davies", "david", "larin"],
    "Wales": ["wales", "cymru", "bale", "ramsey"],
}

# Pre-compile patterns for performance
_TEAM_PATTERNS: dict[str, re.Pattern] = {
    team: re.compile(
        "|".join(re.escape(kw) for kw in keywords),
        flags=re.IGNORECASE,
    )
    for team, keywords in TEAM_KEYWORDS.items()
}


def detect_teams(text: str) -> str:
    """
    Return a comma-separated string of teams mentioned in *text*.

    Parameters
    ----------
    text : str
        Raw or cleaned text to inspect.

    Returns
    -------
    str
        Comma-separated team names, or empty string if none found.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    found = [team for team, pattern in _TEAM_PATTERNS.items() if pattern.search(text)]
    return ",".join(found)


def detect_teams_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Add a 'team' column to *df* based on keyword matching.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a text column.
    text_col : str
        Column to search for team keywords.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with an additional 'team' column.

    Raises
    ------
    KeyError
        If *text_col* is not found in *df*.
    """
    if text_col not in df.columns:
        raise KeyError(
            f"Column '{text_col}' not found. Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df["team"] = df[text_col].apply(detect_teams)

    total = len(df)
    matched = (df["team"] != "").sum()
    logger.info(
        "Team detection complete. %d / %d rows matched at least one team.",
        matched,
        total,
    )
    return df


def run_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Load a CSV, apply team detection, and save.

    Parameters
    ----------
    input_path : str
        Path to the input CSV.
    output_path : str
        Destination path for the team-annotated CSV.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If *input_path* does not exist.
    """
    import os

    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        logger.error("Input file not found: %s", input_path)
        raise

    df = detect_teams_dataframe(df)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Team-annotated data saved to %s", output_path)
    return df


if __name__ == "__main__":
    import os

    base = os.path.dirname(__file__)
    run_pipeline(
        input_path=os.path.join(base, "../data/processed/reddit_sentiment.csv"),
        output_path=os.path.join(base, "../data/processed/reddit_final.csv"),
    )
