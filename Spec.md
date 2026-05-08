# World Cup Reddit Sentiment Dashboard — MVP Spec

## Overview

Build a lightweight dashboard that tracks fan sentiment for World Cup teams using Reddit discussion data.

The MVP should:

* collect Reddit posts and comments related to World Cup teams,
* perform sentiment analysis on the text,
* visualize sentiment trends over time,
* provide a simple interactive dashboard.

---

# MVP Features

## 1. Reddit Data Collection

### Data Sources

* r/soccer
* r/worldcup

### Data to Collect

For each post/comment:

* post title
* comment body
* timestamp
* subreddit
* score/upvotes

### Tools

* Python
* PRAW (Reddit API wrapper)

### Output

Store collected data in:

* CSV files
  or
* SQLite database

---

# 2. Text Preprocessing

### Cleaning Steps

* lowercase text
* remove URLs
* remove punctuation
* remove emojis
* remove stopwords
* normalize whitespace

### Libraries

* pandas
* nltk
* re

---

# 3. Sentiment Analysis

### Model

Use VADER sentiment analysis.

### Sentiment Outputs

For each comment:

* positive score
* negative score
* neutral score
* compound score

### Aggregated Metrics

Per team:

* average sentiment
* sentiment trend over time
* comment volume

---

# 4. Team Detection

### Approach

Use keyword matching for team names.

Example:

* Brazil
* Argentina
* France

A comment can belong to multiple teams.

---

# 5. Dashboard

## Framework

Streamlit

## Dashboard Components

### Team Selector

Allow users to select a World Cup team.

### Visualizations

Display:

* sentiment trend chart
* sentiment distribution
* comment volume over time
* top keywords

### Recent Comments

Show recent Reddit comments related to the selected team.

---

# 6. Visualizations

### Charts

* line chart for sentiment over time
* bar chart for team sentiment comparison
* histogram for sentiment distribution
* word frequency chart

### Libraries

* Plotly

---

# 7. Deployment

### Deployment Platform

* Streamlit Cloud

### Repository Requirements

Include:

* README.md
* requirements.txt
* setup instructions
* screenshots
* sample data

---

# Suggested Project Structure

```text
world-cup-sentiment-dashboard/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── collect_data.py
│   ├── preprocess.py
│   ├── sentiment.py
│   └── team_detection.py
│
├── dashboard/
│   └── app.py
│
├── requirements.txt
├── README.md
└── spec.md
```

---

# Tech Stack

| Area            | Tool            |
| --------------- | --------------- |
| Language        | Python          |
| Data Processing | pandas          |
| NLP             | nltk + VADER    |
| API             | PRAW            |
| Visualization   | Plotly          |
| Dashboard       | Streamlit       |
| Storage         | CSV or SQLite   |
| Deployment      | Streamlit Cloud |

---

# MVP Success Criteria

The MVP is successful if:

* Reddit data is collected successfully,
* sentiment analysis runs correctly,
* sentiment trends are displayed in the dashboard,
* the dashboard is deployed publicly,
* the repository is documented clearly.
