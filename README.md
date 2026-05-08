# ⚽ World Cup Reddit Sentiment Dashboard

A lightweight dashboard that tracks fan sentiment for World Cup teams using Reddit discussion data.

---

## Features

- 📥 **Reddit Data Collection** — scrapes posts & comments from `r/soccer` and `r/worldcup` via PRAW
- 🧹 **Text Preprocessing** — URL removal, emoji stripping, punctuation cleaning, stopword removal
- 🤖 **Sentiment Analysis** — VADER compound/positive/negative/neutral scores per comment
- 🏴 **Team Detection** — keyword matching for 32 World Cup teams; a comment can match multiple teams
- 📊 **Interactive Dashboard** — Streamlit + Plotly charts with team selector, trend lines, and more

---

## Project Structure

```
world-cup-sentiment-dashboard/
│
├── data/
│   ├── raw/               # Raw Reddit CSVs / SQLite DB
│   └── processed/         # Cleaned & sentiment-scored CSVs
│
├── src/
│   ├── collect_data.py    # Reddit scraper (PRAW)
│   ├── preprocess.py      # Text cleaning pipeline
│   ├── sentiment.py       # VADER sentiment analysis
│   └── team_detection.py  # Keyword-based team tagging
│
├── dashboard/
│   └── app.py             # Streamlit dashboard
│
├── requirements.txt
├── README.md
└── Spec.md
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/tshakeel05/World-Cup-Sentiment.git
cd World-Cup-Sentiment
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Reddit API credentials

Create a `.env` file in the project root:

```env
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=WorldCupSentimentBot/1.0
```

Get credentials by creating a Reddit app at <https://www.reddit.com/prefs/apps>.

### 5. Collect data

```bash
python src/collect_data.py
```

This saves raw data to `data/raw/reddit_data.csv` and `data/raw/reddit_data.db`.

### 6. Run the preprocessing & sentiment pipeline

```bash
python src/preprocess.py
python src/sentiment.py
python src/team_detection.py
```

### 7. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard opens at `http://localhost:8501`.

---

## Dashboard Screenshots

### Team Selector & KPIs
Displays average compound score, positive %, negative %, and total comment volume for the selected team.

### Sentiment Trend Chart
Line chart showing daily average VADER compound score over time.

### Sentiment Distribution
Histogram of compound scores, revealing the overall tone of discussions.

### Comment Volume Over Time
Bar chart showing daily number of mentions per team.

### Top Keywords
Horizontal bar chart of the most frequent words in team-related comments.

### Team Comparison
Bar chart comparing average sentiment across all 32 World Cup teams.

### Recent Comments
Scrollable table of the most recent Reddit comments, colour-coded by sentiment.

---

## Tech Stack

| Area            | Tool                  |
|-----------------|-----------------------|
| Language        | Python 3.10+          |
| Data Collection | PRAW                  |
| Data Processing | pandas                |
| NLP             | NLTK + VADER          |
| Visualization   | Plotly                |
| Dashboard       | Streamlit             |
| Storage         | CSV / SQLite          |
| Deployment      | Streamlit Cloud       |

---

## Deployment on Streamlit Cloud

1. Push the repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. Set **Main file path** to `dashboard/app.py`.
4. Add secrets (`REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`) via the Streamlit Cloud secrets manager.
5. Click **Deploy**.

---

## Sample Data

A sample dataset (`data/raw/reddit_data.csv`) is included so the dashboard works out of the box without Reddit API credentials.

---

## License

MIT
