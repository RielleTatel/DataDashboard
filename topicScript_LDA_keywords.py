#!/usr/bin/env python3
"""
LDA Topic Analysis Script with Per-Article Topic Keywords
========================================================

This script analyzes the TEXT column of your dataset and assigns each article to an LDA topic,
outputting both the topic label and the top keywords for that topic in new columns.

Author: Data Science Professional
Date: 2024
"""

import pandas as pd
import numpy as np
import re
import warnings
import logging
from pathlib import Path

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Visualization (optional, not used in this script)
# import matplotlib.pyplot as plt
# import seaborn as sns
# from wordcloud import WordCloud

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LDAKeywordAnalyzer:
    """
    LDA topic analyzer for misinformation articles, outputs per-article topic keywords.
    """
    def __init__(self, language='english'):
        self.language = language
        self.stop_words = set()
        self.lemmatizer = WordNetLemmatizer()
        self._download_nltk_data()
        logger.info("LDAKeywordAnalyzer initialized successfully")

    def _download_nltk_data(self):
        import ssl
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        required_packages = ['punkt', 'stopwords', 'wordnet']
        for package in required_packages:
            try:
                nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
                logger.info(f"NLTK {package} already available")
            except LookupError:
                try:
                    logger.info(f"Downloading NLTK {package}...")
                    nltk.download(package, quiet=True)
                    logger.info(f"Successfully downloaded {package}")
                except Exception as e:
                    logger.warning(f"Failed to download {package}: {e}")
                    if package == 'stopwords':
                        self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs'])
                    continue
        try:
            self.stop_words = set(stopwords.words(self.language))
        except:
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs'])
            logger.info("Using fallback stopwords list")

    def preprocess_text(self, text):
        if pd.isna(text) or text == '':
            return ''
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        try:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words and len(token) > 2]
        except:
            tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)

    def perform_lda_analysis(self, texts, n_topics=8, n_keywords=10):
        logger.info(f"Performing LDA analysis with {n_topics} topics...")
        tfidf = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2)
        )
        tfidf_matrix = tfidf.fit_transform(texts)
        lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=50
        )
        lda_output = lda_model.fit_transform(tfidf_matrix)
        feature_names = tfidf.get_feature_names_out()
        topic_keywords = {}
        for topic_idx, topic in enumerate(lda_model.components_):
            top_keywords = [feature_names[i] for i in topic.argsort()[-n_keywords:][::-1]]
            topic_keywords[topic_idx] = top_keywords
        # For each document, assign topic and keywords
        doc_topics = lda_output.argmax(axis=1)
        doc_topic_conf = lda_output.max(axis=1)
        doc_topic_keywords = [topic_keywords[topic] for topic in doc_topics]
        doc_topic_keywords_str = [', '.join(words) for words in doc_topic_keywords]
        return doc_topics, doc_topic_conf, doc_topic_keywords_str, topic_keywords

    def analyze_dataset(self, file_path, output_path=None):
        logger.info(f"Starting analysis of dataset: {file_path}")
        if file_path.endswith('.csv'):
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file_path, encoding='latin-1')
                except:
                    df = pd.read_csv(file_path, encoding='cp1252')
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please use CSV or Excel files.")
        logger.info(f"Loaded dataset with {len(df)} rows and columns: {list(df.columns)}")
        if 'TEXT' in df.columns:
            text_col = 'TEXT'
        elif 'CONTENT' in df.columns:
            text_col = 'CONTENT'
            df = df.rename(columns={'CONTENT': 'TEXT'})
        else:
            raise ValueError("No TEXT or CONTENT column found in the dataset")
        logger.info("Preprocessing text data...")
        df['TEXT_PREPROCESSED'] = df['TEXT'].apply(self.preprocess_text)
        initial_count = len(df)
        df = df[df['TEXT_PREPROCESSED'].str.strip() != ''].reset_index(drop=True)
        logger.info(f"Removed {initial_count - len(df)} rows with empty text after preprocessing")
        texts = df['TEXT_PREPROCESSED'].tolist()
        # LDA analysis
        doc_topics, doc_topic_conf, doc_topic_keywords_str, topic_keywords = self.perform_lda_analysis(texts, n_topics=8, n_keywords=10)
        df['LDA_TOPIC'] = [f'lda_topic_{t}' for t in doc_topics]
        df['LDA_CONFIDENCE'] = doc_topic_conf
        df['LDA_TOPIC_KEYWORDS'] = doc_topic_keywords_str
        if output_path:
            logger.info(f"Saving results to {output_path}")
            if output_path.endswith('.csv'):
                df.to_csv(output_path, index=False)
            else:
                df.to_excel(output_path, index=False)
        self._generate_summary(df, topic_keywords)
        return df

    def _generate_summary(self, df, topic_keywords):
        logger.info("Generating summary statistics...")
        print("\n" + "="*60)
        print("LDA TOPIC ANALYSIS SUMMARY")
        print("="*60)
        print(f"\nTotal articles analyzed: {len(df)}")
        print(f"\nLDA Topic Distribution:")
        topic_counts = df['LDA_TOPIC'].value_counts()
        for topic, count in topic_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {topic}: {count} articles ({percentage:.1f}%)")
        print(f"\nTop Keywords for Each LDA Topic:")
        for topic_idx, keywords in topic_keywords.items():
            print(f"  lda_topic_{topic_idx}: {', '.join(keywords)}")
        print("="*60)

def main():
    print("="*60)
    print("LDA TOPIC ANALYSIS FOR MISINFORMATION ARTICLES")
    print("="*60)
    analyzer = LDAKeywordAnalyzer()
    input_files = [
        "Source file CSV/SentimentAnlysisSourceFile/UpdatedSourceFileT.csv",
    ]
    existing_files = []
    for file_path in input_files:
        if Path(file_path).exists():
            existing_files.append(file_path)
            print(f"✓ Found: {file_path}")
        else:
            print(f"✗ Missing: {file_path}")
    if not existing_files:
        print("No input files found. Please check file paths.")
        return
    for file_path in existing_files:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {file_path}")
        print(f"{'='*60}")
        try:
            base_name = Path(file_path).stem
            output_csv = f"lda_topic_analysis_{base_name}.csv"
            output_excel = f"lda_topic_analysis_{base_name}.xlsx"
            results_df = analyzer.analyze_dataset(file_path, output_excel)
            results_df.to_csv(output_csv, index=False)
            print(f"\n✓ LDA analysis completed successfully!")
            print(f"  Results saved to: {output_excel}")
            print(f"  Results saved to: {output_csv}")
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            print(f"✗ Error analyzing {file_path}: {e}")
    print(f"\n{'='*60}")
    print("LDA ANALYSIS COMPLETE!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 