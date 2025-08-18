# 📊 Data Dashboard - Misinformation Analysis Project

This repository contains a comprehensive analysis pipeline for misinformation detection, including data cleaning, topic analysis, and sentiment analysis of news articles.

---

## 📦 Dependencies and Installation

### Core Dependencies
Both scripts require the following core Python packages:

```bash
pip3 install pandas numpy matplotlib seaborn scikit-learn nltk
```

### Topic Analysis Dependencies (`topicScript_simple.py`)
```bash
pip3 install wordcloud
```

### Sentiment Analysis Dependencies (`sentimentScript.py`)
```bash
pip3 install vaderSentiment textblob wordcloud
```

### Complete Installation
To install all dependencies for both scripts at once:

```bash
pip3 install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud vaderSentiment textblob
```

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended for large datasets)
- **Storage**: Sufficient space for output files and visualizations

### NLTK Data Downloads
The scripts automatically download required NLTK data packages:
- `punkt` - For text tokenization
- `stopwords` - For stop word removal
- `wordnet` - For lemmatization

### Troubleshooting
- **SSL Certificate Issues**: The scripts handle macOS SSL certificate problems automatically
- **Encoding Issues**: Multiple encoding formats (UTF-8, Latin-1, CP1252) are supported
- **Memory Issues**: For very large datasets, consider using the sampling approach or processing in batches

---

## 📊 Dataset Overview

| File Name  | Original Rows | Notes                                  |
|------------|----------------|----------------------------------------|
| `True.csv` | ~21,211        | Generally clean with minimal missing values |
| `Fake.csv` | ~23,502        | Contains more missing and malformed data     |

---

## 🧹 Data Cleaning Process

### Overview
The dataset consists of two primary CSV files: `True.csv` and `Fake.csv`, containing news articles that need to be cleaned and prepared for analysis.

### Cleaning Strategy
Both datasets are cleaned using **unified criteria** for consistency and reliability:

- **Essential Fields**: Rows are removed if any of the following critical fields are missing:
  - `title` - Article headline
  - `text` - Article content
  - `subject` - News category
  - `date` - Publication date

- **Quality Assurance**: These four fields are essential for meaningful analysis. Missing data in any of them could skew results or reduce interpretability.

### Dataset-Specific Results

#### ✅ `True.csv`
- **Cleaning Rule:** Drop rows with missing `title`, `text`, `subject`, or `date`
- **Observation:** Fewer rows dropped due to overall higher data quality
- **Outcome:** Majority of data retained for analysis

#### ❌ `Fake.csv`
- **Cleaning Rule:** Drop rows with missing `title`, `text`, `subject`, or `date`
- **Observation:** More rows dropped due to frequent missing or corrupted values
- **Outcome:** Smaller but cleaner dataset ready for comparison

### Character Encoding Cleanup (Mojibake Fix)
Some entries, mostly from `Fake.csv`, contained **mojibake** (corrupted characters) due to encoding issues. These were fixed using formulas in **Excel or Google Sheets**.

#### Example Formula:
```excel
=SUBSTITUTE(
  SUBSTITUTE(
    SUBSTITUTE(
      SUBSTITUTE(
        SUBSTITUTE(
          SUBSTITUTE(
            SUBSTITUTE(
              SUBSTITUTE(
                SUBSTITUTE(
                  SUBSTITUTE(
                    SUBSTITUTE(
                      SUBSTITUTE(
                        A2,
                        "√¢¬Ä¬ú", """),
                      "√¢¬Ä¬ù", """),
                    "√¢¬Ä¬ôs", "'s"),
                  "√¢¬Ä¬ï", "'"),
                "√¢¬Ä¬î", "'"),
              "√¢¬Ä¬ït", "'t"),
            "√¢¬Ä¬í", """),
          "√¢¬Ä¬ì", """),
        "√¢¬Ä¬¶", "…"),
      "√¢¬Ä¬ä", "—"),
    "√¢¬Ä¬", ""),
  "√¢", "")
```

---

## 🔍 Topic Analysis (`topicScript_simple.py`)

### Process Overview
The topic analysis script uses multiple NLP techniques to categorize articles into predefined topic categories, helping identify patterns in misinformation content.

### Analysis Methods
1. **Keyword Matching**: Uses predefined keyword lists for common misinformation topics
2. **LDA (Latent Dirichlet Allocation)**: Machine learning approach to discover hidden topic patterns
3. **Hybrid Classification**: Combines both methods for optimal topic assignment

### Added Columns

| Column Name | Description |
|-------------|-------------|
| `KEYWORD_TOPIC` | Topic assigned using keyword matching |
| `KEYWORD_CONFIDENCE` | Confidence score (0-1) for keyword-based classification |
| `LDA_TOPIC` | Topic discovered using LDA analysis |
| `LDA_CONFIDENCE` | Confidence score (0-1) for LDA classification |
| `FINAL_TOPIC` | Final topic assignment (keyword or LDA based on confidence) |
| `FINAL_CONFIDENCE` | Final confidence score for the assigned topic |

### Topic Categories
- **Politics**: Elections, government, policy, campaigns
- **Health**: Vaccines, COVID, medical treatments
- **Conspiracy**: Secret theories, cover-ups, revelations
- **Science**: Research, studies, scientific evidence
- **Social Media**: Platforms, viral content, online trends
- **Economy**: Financial markets, business, trade
- **Crime**: Criminal activities, law enforcement
- **Technology**: Digital innovations, AI, software
- **Environment**: Climate change, pollution, nature
- **Entertainment**: Celebrities, movies, music

---

## 😊 Sentiment Analysis (`sentimentScript.py`)

### Process Overview
The sentiment analysis script examines the emotional content of articles to identify sentiment patterns and dominant emotions in misinformation.

### Analysis Methods
1. **VADER Sentiment**: Valence Aware Dictionary and sEntiment Reasoner for polarity analysis
2. **TextBlob**: Additional sentiment and subjectivity analysis
3. **Emotion Keyword Analysis**: Identifies specific emotions using predefined keyword lists

### Added Columns

| Column Name | Description |
|-------------|-------------|
| `overall_sentiment` | Overall sentiment classification (positive/negative/neutral) |
| `sentiment_score` | VADER compound sentiment score (-1 to +1) |
| `sentiment_level` | Sentiment intensity (very_negative to very_positive) |
| `dominant_emotion` | Primary emotion detected in the text |
| `emotion_confidence` | Confidence score for emotion classification |
| `subjectivity` | TextBlob subjectivity score (0-1) |
| `vader_compound` | VADER compound polarity score |
| `vader_positive` | VADER positive sentiment score |
| `vader_negative` | VADER negative sentiment score |
| `vader_neutral` | VADER neutral sentiment score |
| `textblob_polarity` | TextBlob polarity score (-1 to +1) |
| `textblob_subjectivity` | TextBlob subjectivity score (0-1) |
| `emotion_fear` | Fear emotion score |
| `emotion_anger` | Anger emotion score |
| `emotion_sadness` | Sadness emotion score |
| `emotion_joy` | Joy emotion score |
| `emotion_surprise` | Surprise emotion score |
| `emotion_disgust` | Disgust emotion score |
| `emotion_trust` | Trust emotion score |
| `emotion_anticipation` | Anticipation emotion score |

### Sentiment Categories
- **Overall Sentiment**: Positive, Negative, Neutral
- **Intensity Levels**: Very Negative, Negative, Slightly Negative, Neutral, Slightly Positive, Positive, Very Positive
- **Emotions**: Fear, Anger, Sadness, Joy, Surprise, Disgust, Trust, Anticipation

### Business Insights
This analysis helps answer key questions such as:
- What sentiment is most common in misinformation per subject?
- Which emotions dominate in different topic categories?
- How do sentiment patterns differ between true and fake news?
- What emotional triggers are most effective in spreading misinformation?

---

## 🔄 Unified Cleaning Rules

Both datasets are cleaned using the **same criteria** for consistency:

- Rows are removed if **any** of the following fields are missing:
  - `title`
  - `text`
  - `subject`
  - `date`

> These four fields are essential for analysis. Missing data in any of them may skew results or reduce interpretability.

