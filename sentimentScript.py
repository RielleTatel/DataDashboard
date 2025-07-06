#!/usr/bin/env python3
"""
Sentiment Analysis Script for Misinformation Articles Dataset
============================================================

This script analyzes the sentiment of misinformation articles to identify
the most common emotional patterns per subject and topic category.

Business Question: What sentiment is most common in misinformation per subject?
(e.g., fear, anger, hope, sadness, surprise, disgust, trust, anticipation)

Author: Data Science Professional
Date: 2024
"""

import pandas as pd
import numpy as np
import re
import warnings
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Advanced sentiment analyzer for misinformation articles using multiple techniques.
    """
    
    def __init__(self, language='english'):
        """Initialize the sentiment analyzer."""
        self.language = language
        self.stop_words = set()
        self.lemmatizer = WordNetLemmatizer()
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Define emotion categories and their keywords
        self.emotion_keywords = {
            'fear': ['fear', 'afraid', 'scared', 'terrified', 'panic', 'horror', 'dread', 'anxiety', 'worry', 'threat', 'danger', 'alarm', 'terror', 'frightened', 'nervous', 'concerned', 'disturbed', 'shocked', 'outraged', 'appalled'],
            'anger': ['anger', 'angry', 'furious', 'rage', 'outrage', 'furious', 'mad', 'irritated', 'annoyed', 'frustrated', 'hostile', 'aggressive', 'violent', 'hate', 'hatred', 'disgust', 'contempt', 'resentment', 'bitter', 'enraged'],
            'sadness': ['sad', 'sadness', 'depressed', 'grief', 'sorrow', 'melancholy', 'despair', 'hopeless', 'miserable', 'unhappy', 'disappointed', 'heartbroken', 'devastated', 'crushed', 'defeated', 'lonely', 'isolated', 'abandoned', 'rejected', 'hurt'],
            'joy': ['joy', 'happy', 'happiness', 'excited', 'thrilled', 'delighted', 'pleased', 'content', 'satisfied', 'cheerful', 'jubilant', 'ecstatic', 'elated', 'overjoyed', 'blessed', 'fortunate', 'lucky', 'grateful', 'thankful', 'blessed'],
            'surprise': ['surprise', 'surprised', 'shocked', 'amazed', 'astonished', 'stunned', 'bewildered', 'confused', 'puzzled', 'perplexed', 'startled', 'taken aback', 'unexpected', 'unbelievable', 'incredible', 'remarkable', 'extraordinary', 'unusual', 'strange', 'odd'],
            'disgust': ['disgust', 'disgusted', 'revolted', 'repulsed', 'sickened', 'nauseated', 'appalled', 'horrified', 'offended', 'insulted', 'outraged', 'shocked', 'scandalized', 'disgusting', 'revolting', 'repulsive', 'sickening', 'nauseating', 'appalling', 'horrifying'],
            'trust': ['trust', 'trusted', 'trustworthy', 'reliable', 'dependable', 'faithful', 'loyal', 'honest', 'sincere', 'genuine', 'authentic', 'credible', 'believable', 'confident', 'assured', 'secure', 'safe', 'protected', 'guarded', 'defended'],
            'anticipation': ['anticipation', 'excited', 'eager', 'enthusiastic', 'hopeful', 'optimistic', 'confident', 'assured', 'prepared', 'ready', 'expectant', 'waiting', 'looking forward', 'anticipated', 'expected', 'predicted', 'forecast', 'projected', 'planned', 'scheduled']
        }
        
        # Define sentiment intensity levels
        self.sentiment_levels = {
            'very_negative': (-1.0, -0.6),
            'negative': (-0.6, -0.2),
            'slightly_negative': (-0.2, -0.05),
            'neutral': (-0.05, 0.05),
            'slightly_positive': (0.05, 0.2),
            'positive': (0.2, 0.6),
            'very_positive': (0.6, 1.0)
        }
        
        logger.info("SentimentAnalyzer initialized successfully")
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        import ssl
        
        # Create SSL context that doesn't verify certificates (for macOS SSL issues)
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Download required NLTK data
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
                    # Create a basic fallback for stopwords
                    if package == 'stopwords':
                        self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs'])
                    continue
        
        # Set stopwords
        try:
            self.stop_words = set(stopwords.words(self.language))
        except:
            # Fallback stopwords if download failed
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs'])
            logger.info("Using fallback stopwords list")
    
    def preprocess_text(self, text):
        """Preprocess text for sentiment analysis."""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters but keep punctuation for sentiment analysis
        text = re.sub(r'[^a-zA-Z\s\.\!\?\,\;\:]', '', text)
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            # Fallback tokenization if NLTK punkt is not available
            tokens = text.split()
        
        # Remove stopwords but keep emotional words
        emotional_words = set()
        for emotion_words in self.emotion_keywords.values():
            emotional_words.update(emotion_words)
        
        tokens = [token for token in tokens 
                 if token not in self.stop_words or token in emotional_words]
        
        return ' '.join(tokens)
    
    def analyze_vader_sentiment(self, text):
        """Analyze sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner)."""
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            return {
                'vader_compound': scores['compound'],
                'vader_positive': scores['pos'],
                'vader_negative': scores['neg'],
                'vader_neutral': scores['neu']
            }
        except Exception as e:
            logger.warning(f"VADER analysis failed: {e}")
            return {
                'vader_compound': 0.0,
                'vader_positive': 0.0,
                'vader_negative': 0.0,
                'vader_neutral': 1.0
            }
    
    def analyze_textblob_sentiment(self, text):
        """Analyze sentiment using TextBlob."""
        try:
            blob = TextBlob(text)
            return {
                'textblob_polarity': blob.sentiment.polarity,
                'textblob_subjectivity': blob.sentiment.subjectivity
            }
        except Exception as e:
            logger.warning(f"TextBlob analysis failed: {e}")
            return {
                'textblob_polarity': 0.0,
                'textblob_subjectivity': 0.0
            }
    
    def analyze_emotion_keywords(self, text):
        """Analyze emotions based on keyword presence."""
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in self.emotion_keywords.items():
            # Count keyword occurrences
            count = sum(1 for keyword in keywords if keyword in text_lower)
            # Normalize by text length and keyword list size
            normalized_score = count / (len(keywords) * max(1, len(text.split()) / 100))
            emotion_scores[f'emotion_{emotion}'] = normalized_score
        
        # Find dominant emotion
        if emotion_scores:
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            emotion_scores['dominant_emotion'] = dominant_emotion.replace('emotion_', '')
            emotion_scores['emotion_confidence'] = emotion_scores[dominant_emotion]
        else:
            emotion_scores['dominant_emotion'] = 'neutral'
            emotion_scores['emotion_confidence'] = 0.0
        
        return emotion_scores
    
    def classify_sentiment_level(self, compound_score):
        """Classify sentiment into intensity levels."""
        for level, (min_score, max_score) in self.sentiment_levels.items():
            if min_score <= compound_score < max_score:
                return level
        return 'very_positive'  # Default for scores >= 1.0
    
    def analyze_article_sentiment(self, text):
        """Comprehensive sentiment analysis for a single article."""
        if not text or pd.isna(text):
            return {
                'overall_sentiment': 'neutral',
                'sentiment_score': 0.0,
                'sentiment_level': 'neutral',
                'dominant_emotion': 'neutral',
                'emotion_confidence': 0.0,
                'subjectivity': 0.0
            }
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # VADER sentiment analysis
        vader_results = self.analyze_vader_sentiment(processed_text)
        
        # TextBlob sentiment analysis
        textblob_results = self.analyze_textblob_sentiment(processed_text)
        
        # Emotion keyword analysis
        emotion_results = self.analyze_emotion_keywords(processed_text)
        
        # Combine results
        compound_score = vader_results['vader_compound']
        
        # Determine overall sentiment
        if compound_score > 0.05:
            overall_sentiment = 'positive'
        elif compound_score < -0.05:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_score': compound_score,
            'sentiment_level': self.classify_sentiment_level(compound_score),
            'dominant_emotion': emotion_results['dominant_emotion'],
            'emotion_confidence': emotion_results['emotion_confidence'],
            'subjectivity': textblob_results['textblob_subjectivity'],
            **vader_results,
            **textblob_results,
            **emotion_results
        }
    
    def analyze_dataset_sentiment(self, file_path, output_path=None):
        """Analyze sentiment for entire dataset."""
        logger.info(f"Starting sentiment analysis of dataset: {file_path}")
        
        # Load dataset
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='latin-1')
            except:
                df = pd.read_csv(file_path, encoding='cp1252')
        
        logger.info(f"Loaded dataset with {len(df)} rows and columns: {list(df.columns)}")
        
        # Check required columns
        required_columns = ['TITLE', 'TEXT', 'SUBJECT', 'DATE']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
        
        # Handle different text column names
        if 'TEXT' in df.columns:
            text_col = 'TEXT'
        elif 'CONTENT' in df.columns:
            text_col = 'CONTENT'
            df = df.rename(columns={'CONTENT': 'TEXT'})
        else:
            raise ValueError("No TEXT or CONTENT column found in the dataset")
        
        # Analyze sentiment for each article
        logger.info("Analyzing sentiment for each article...")
        sentiment_results = []
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                logger.info(f"Processed {idx} articles...")
            
            text = row['TEXT']
            sentiment_analysis = self.analyze_article_sentiment(text)
            sentiment_results.append(sentiment_analysis)
        
        # Add sentiment results to dataframe
        sentiment_df = pd.DataFrame(sentiment_results)
        df = pd.concat([df, sentiment_df], axis=1)
        
        # Generate summary statistics
        self._generate_sentiment_summary(df)
        
        # Save results
        if output_path:
            logger.info(f"Saving results to {output_path}")
            df.to_excel(output_path, index=False)
        
        return df
    
    def _generate_sentiment_summary(self, df):
        """Generate and display sentiment summary statistics."""
        logger.info("Generating sentiment summary statistics...")
        
        print("\n" + "="*80)
        print("SENTIMENT ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\nTotal articles analyzed: {len(df)}")
        
        # Overall sentiment distribution
        print(f"\nOverall Sentiment Distribution:")
        sentiment_counts = df['overall_sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {sentiment.capitalize()}: {count} articles ({percentage:.1f}%)")
        
        # Sentiment level distribution
        print(f"\nSentiment Intensity Distribution:")
        level_counts = df['sentiment_level'].value_counts()
        for level, count in level_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {level.replace('_', ' ').title()}: {count} articles ({percentage:.1f}%)")
        
        # Dominant emotions
        print(f"\nDominant Emotions:")
        emotion_counts = df['dominant_emotion'].value_counts()
        for emotion, count in emotion_counts.head(10).items():
            percentage = (count / len(df)) * 100
            print(f"  {emotion.capitalize()}: {count} articles ({percentage:.1f}%)")
        
        # Sentiment by subject
        if 'SUBJECT' in df.columns:
            print(f"\nSentiment by Subject:")
            subject_sentiment = df.groupby('SUBJECT')['overall_sentiment'].value_counts().unstack(fill_value=0)
            for subject in subject_sentiment.index:
                print(f"\n  {subject}:")
                for sentiment in ['positive', 'negative', 'neutral']:
                    if sentiment in subject_sentiment.columns:
                        count = subject_sentiment.loc[subject, sentiment]
                        total = subject_sentiment.loc[subject].sum()
                        percentage = (count / total) * 100 if total > 0 else 0
                        print(f"    {sentiment.capitalize()}: {count} articles ({percentage:.1f}%)")
        
        # Sentiment by topic (if available)
        if 'FINAL_TOPIC' in df.columns:
            print(f"\nSentiment by Topic:")
            topic_sentiment = df.groupby('FINAL_TOPIC')['overall_sentiment'].value_counts().unstack(fill_value=0)
            for topic in topic_sentiment.index:
                print(f"\n  {topic}:")
                for sentiment in ['positive', 'negative', 'neutral']:
                    if sentiment in topic_sentiment.columns:
                        count = topic_sentiment.loc[topic, sentiment]
                        total = topic_sentiment.loc[topic].sum()
                        percentage = (count / total) * 100 if total > 0 else 0
                        print(f"    {sentiment.capitalize()}: {count} articles ({percentage:.1f}%)")
        
        # Average sentiment scores
        print(f"\nAverage Sentiment Scores:")
        print(f"  VADER Compound Score: {df['vader_compound'].mean():.3f}")
        print(f"  TextBlob Polarity: {df['textblob_polarity'].mean():.3f}")
        print(f"  Subjectivity: {df['subjectivity'].mean():.3f}")
        
        print("="*80)
    
    def create_sentiment_visualizations(self, df, output_dir="sentiment_analysis_plots"):
        """Create visualizations for sentiment analysis results."""
        logger.info(f"Creating sentiment visualizations in {output_dir}")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        
        # 1. Overall sentiment distribution
        plt.figure(figsize=(12, 8))
        sentiment_counts = df['overall_sentiment'].value_counts()
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors)
        plt.title('Overall Sentiment Distribution', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/overall_sentiment_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Sentiment intensity distribution
        plt.figure(figsize=(14, 8))
        level_counts = df['sentiment_level'].value_counts()
        plt.bar(range(len(level_counts)), level_counts.values, color='skyblue', edgecolor='black')
        plt.xlabel('Sentiment Intensity Level')
        plt.ylabel('Number of Articles')
        plt.title('Sentiment Intensity Distribution', fontsize=16, fontweight='bold')
        plt.xticks(range(len(level_counts)), [level.replace('_', ' ').title() for level in level_counts.index], rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sentiment_intensity_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Dominant emotions
        plt.figure(figsize=(12, 8))
        emotion_counts = df['dominant_emotion'].value_counts().head(8)
        plt.bar(range(len(emotion_counts)), emotion_counts.values, color='lightcoral', edgecolor='black')
        plt.xlabel('Dominant Emotion')
        plt.ylabel('Number of Articles')
        plt.title('Dominant Emotions in Articles', fontsize=16, fontweight='bold')
        plt.xticks(range(len(emotion_counts)), [emotion.capitalize() for emotion in emotion_counts.index], rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/dominant_emotions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Sentiment by subject (if available)
        if 'SUBJECT' in df.columns:
            plt.figure(figsize=(16, 10))
            subject_sentiment = df.groupby('SUBJECT')['overall_sentiment'].value_counts().unstack(fill_value=0)
            subject_sentiment.plot(kind='bar', stacked=True, figsize=(16, 10), 
                                 color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
            plt.title('Sentiment Distribution by Subject', fontsize=16, fontweight='bold')
            plt.xlabel('Subject')
            plt.ylabel('Number of Articles')
            plt.legend(title='Sentiment')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/sentiment_by_subject.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Sentiment by topic (if available)
        if 'FINAL_TOPIC' in df.columns:
            plt.figure(figsize=(16, 10))
            topic_sentiment = df.groupby('FINAL_TOPIC')['overall_sentiment'].value_counts().unstack(fill_value=0)
            topic_sentiment.plot(kind='bar', stacked=True, figsize=(16, 10), 
                               color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
            plt.title('Sentiment Distribution by Topic', fontsize=16, fontweight='bold')
            plt.xlabel('Topic')
            plt.ylabel('Number of Articles')
            plt.legend(title='Sentiment')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/sentiment_by_topic.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 6. Sentiment score distribution
        plt.figure(figsize=(12, 6))
        plt.hist(df['vader_compound'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.xlabel('VADER Compound Sentiment Score')
        plt.ylabel('Number of Articles')
        plt.title('Distribution of Sentiment Scores', fontsize=16, fontweight='bold')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Neutral (0)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sentiment_score_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Sentiment visualizations saved to {output_dir}")


def main():
    """Main function to run the sentiment analysis."""
    print("="*80)
    print("MISINFORMATION ARTICLES SENTIMENT ANALYSIS")
    print("="*80)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Define input files to analyze
    input_files = [
        "Source file CSV/Fake-InputSourceFile.csv",
        "Source file CSV/True-InputSourceFile.csv",
    ]
    
    # Check which files exist
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
    
    # Analyze each file
    for file_path in existing_files:
        print(f"\n{'='*80}")
        print(f"ANALYZING SENTIMENT: {file_path}")
        print(f"{'='*80}")
        
        try:
            # Generate output filename
            base_name = Path(file_path).stem
            output_excel = f"sentiment_analysis_{base_name}.xlsx"
            
            # Perform sentiment analysis
            results_df = analyzer.analyze_dataset_sentiment(file_path, output_excel)
            
            # Create visualizations
            viz_dir = f"sentiment_analysis_plots_{base_name}"
            analyzer.create_sentiment_visualizations(results_df, viz_dir)
            
            print(f"\n✓ Sentiment analysis completed successfully!")
            print(f"  Results saved to: {output_excel}")
            print(f"  Visualizations saved to: {viz_dir}/")
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {file_path}: {e}")
            print(f"✗ Error analyzing sentiment for {file_path}: {e}")
    
    print(f"\n{'='*80}")
    print("SENTIMENT ANALYSIS COMPLETE!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
