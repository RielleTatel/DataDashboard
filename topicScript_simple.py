#!/usr/bin/env python3
"""
Simplified Topic Analysis Script for Misinformation Articles Dataset
==================================================================

This script analyzes the TEXT column of your dataset and generates topic categories
for each article using multiple NLP techniques.

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
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TopicAnalyzer:
    """
    Simplified topic analyzer for misinformation articles.
    """
    
    def __init__(self, language='english'):
        """Initialize the topic analyzer."""
        self.language = language
        self.stop_words = set()
        self.lemmatizer = WordNetLemmatizer()
        self.topic_keywords = {}
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Predefined topic categories for misinformation detection
        self.misinfo_topics = {
            'politics': ['election', 'politician', 'government', 'policy', 'vote', 'campaign', 'democrat', 'republican'],
            'health': ['vaccine', 'covid', 'virus', 'disease', 'medical', 'health', 'doctor', 'treatment', 'cure'],
            'conspiracy': ['conspiracy', 'secret', 'hidden', 'cover-up', 'truth', 'exposed', 'revealed', 'suppressed'],
            'science': ['research', 'study', 'scientist', 'evidence', 'data', 'experiment', 'discovery'],
            'social_media': ['facebook', 'twitter', 'instagram', 'social', 'media', 'platform', 'viral'],
            'economy': ['economy', 'market', 'financial', 'money', 'business', 'trade', 'stock', 'investment'],
            'crime': ['crime', 'criminal', 'police', 'law', 'justice', 'arrest', 'investigation'],
            'technology': ['technology', 'tech', 'digital', 'computer', 'software', 'internet', 'ai', 'artificial'],
            'environment': ['climate', 'environment', 'global warming', 'pollution', 'nature', 'earth'],
            'entertainment': ['celebrity', 'movie', 'music', 'entertainment', 'star', 'actor', 'artist']
        }
        
        logger.info("TopicAnalyzer initialized successfully")
    
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
        """Preprocess text for analysis."""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            # Fallback tokenization if NLTK punkt is not available
            tokens = text.split()
        
        # Remove stopwords and lemmatize
        try:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words and len(token) > 2]
        except:
            # Fallback if lemmatizer is not available
            tokens = [token for token in tokens 
                     if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def classify_with_keywords(self, text):
        """Classify text using predefined keyword matching."""
        text_lower = text.lower()
        scores = {}
        
        for topic, keywords in self.misinfo_topics.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[topic] = score / len(keywords) if keywords else 0
        
        if not scores or max(scores.values()) == 0:
            return 'general', 0.0
        
        best_topic = max(scores, key=scores.get)
        confidence = scores[best_topic]
        
        return best_topic, confidence
    
    def perform_lda_analysis(self, texts, n_topics=8):
        """Perform Latent Dirichlet Allocation analysis."""
        logger.info(f"Performing LDA analysis with {n_topics} topics...")
        
        # Create TF-IDF matrix for LDA
        tfidf = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = tfidf.fit_transform(texts)
        
        # Fit LDA model
        lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=50
        )
        
        lda_output = lda_model.fit_transform(tfidf_matrix)
        
        # Extract topic keywords
        feature_names = tfidf.get_feature_names_out()
        topic_keywords = {}
        
        for topic_idx, topic in enumerate(lda_model.components_):
            top_keywords = [feature_names[i] for i in topic.argsort()[-10:][::-1]]
            topic_keywords[f'lda_topic_{topic_idx}'] = top_keywords
        
        self.topic_keywords.update(topic_keywords)
        
        # Get dominant topic for each document
        document_topics = {}
        for i, doc_topics in enumerate(lda_output):
            if texts[i].strip():  # Only process non-empty texts
                dominant_topic = doc_topics.argmax()
                confidence = doc_topics[dominant_topic]
                document_topics[i] = {
                    'topic': f'lda_topic_{dominant_topic}',
                    'confidence': confidence,
                    'all_topics': doc_topics.tolist()
                }
        
        return document_topics
    
    def analyze_dataset(self, file_path, output_path=None):
        """Main method to analyze the entire dataset."""
        logger.info(f"Starting analysis of dataset: {file_path}")
        
        # Load dataset with encoding handling
        if file_path.endswith('.csv'):
            try:
                # Try UTF-8 first
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    # Try with different encoding
                    df = pd.read_csv(file_path, encoding='latin-1')
                except:
                    # Last resort
                    df = pd.read_csv(file_path, encoding='cp1252')
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please use CSV or Excel files.")
        
        logger.info(f"Loaded dataset with {len(df)} rows and columns: {list(df.columns)}")
        
        # Handle different text column names
        if 'TEXT' in df.columns:
            text_col = 'TEXT'
        elif 'CONTENT' in df.columns:
            text_col = 'CONTENT'
            # Rename CONTENT to TEXT for consistency
            df = df.rename(columns={'CONTENT': 'TEXT'})
        else:
            raise ValueError("No TEXT or CONTENT column found in the dataset")
        
        # Preprocess TEXT column
        logger.info("Preprocessing text data...")
        df['TEXT_PREPROCESSED'] = df['TEXT'].apply(self.preprocess_text)
        
        # Remove rows with empty preprocessed text
        initial_count = len(df)
        df = df[df['TEXT_PREPROCESSED'].str.strip() != ''].reset_index(drop=True)
        logger.info(f"Removed {initial_count - len(df)} rows with empty text after preprocessing")
        
        texts = df['TEXT_PREPROCESSED'].tolist()
        
        # Perform keyword-based classification
        logger.info("Performing keyword-based classification...")
        keyword_results = []
        for i, text in enumerate(texts):
            topic, confidence = self.classify_with_keywords(text)
            keyword_results.append({'topic': topic, 'confidence': confidence})
        
        # Add keyword analysis results
        df['KEYWORD_TOPIC'] = [r['topic'] for r in keyword_results]
        df['KEYWORD_CONFIDENCE'] = [r['confidence'] for r in keyword_results]
        
        # Perform LDA analysis
        lda_results = self.perform_lda_analysis(texts, n_topics=8)
        
        # Add LDA results
        df['LDA_TOPIC'] = ''
        df['LDA_CONFIDENCE'] = 0.0
        for idx, result in lda_results.items():
            df.loc[idx, 'LDA_TOPIC'] = result['topic']
            df.loc[idx, 'LDA_CONFIDENCE'] = result['confidence']
        
        # Create final topic classification
        df['FINAL_TOPIC'] = df['KEYWORD_TOPIC']
        df['FINAL_CONFIDENCE'] = df['KEYWORD_CONFIDENCE']
        
        # Use LDA if keyword confidence is low
        low_confidence_mask = df['KEYWORD_CONFIDENCE'] < 0.3
        df.loc[low_confidence_mask, 'FINAL_TOPIC'] = df.loc[low_confidence_mask, 'LDA_TOPIC']
        df.loc[low_confidence_mask, 'FINAL_CONFIDENCE'] = df.loc[low_confidence_mask, 'LDA_CONFIDENCE']
        
        # Save results
        if output_path:
            logger.info(f"Saving results to {output_path}")
            if output_path.endswith('.csv'):
                df.to_csv(output_path, index=False)
            else:
                df.to_excel(output_path, index=False)
        
        # Generate summary statistics
        self._generate_summary(df)
        
        return df
    
    def _generate_summary(self, df):
        """Generate and display summary statistics."""
        logger.info("Generating summary statistics...")
        
        print("\n" + "="*60)
        print("TOPIC ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\nTotal articles analyzed: {len(df)}")
        
        # Keyword topic distribution
        print(f"\nKeyword-based Topic Distribution:")
        topic_counts = df['KEYWORD_TOPIC'].value_counts()
        for topic, count in topic_counts.head(10).items():
            percentage = (count / len(df)) * 100
            print(f"  {topic}: {count} articles ({percentage:.1f}%)")
        
        # Confidence statistics
        print(f"\nConfidence Statistics:")
        print(f"  Average keyword confidence: {df['KEYWORD_CONFIDENCE'].mean():.3f}")
        print(f"  Average LDA confidence: {df['LDA_CONFIDENCE'].mean():.3f}")
        
        # Topic keywords summary
        print(f"\nTop Topic Keywords:")
        for topic_name, keywords in self.topic_keywords.items():
            if len(keywords) > 0:
                print(f"  {topic_name}: {', '.join(keywords[:5])}")
        
        print("="*60)
    
    def create_visualizations(self, df, output_dir="topic_analysis_plots"):
        """Create visualizations for the topic analysis results."""
        logger.info(f"Creating visualizations in {output_dir}")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        
        # 1. Topic distribution pie chart
        plt.figure(figsize=(12, 8))
        topic_counts = df['FINAL_TOPIC'].value_counts().head(10)
        plt.pie(topic_counts.values, labels=topic_counts.index, autopct='%1.1f%%')
        plt.title('Distribution of Final Topics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/topic_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confidence distribution histogram
        plt.figure(figsize=(12, 6))
        plt.hist(df['FINAL_CONFIDENCE'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Confidence Score')
        plt.ylabel('Number of Articles')
        plt.title('Distribution of Topic Classification Confidence', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confidence_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")


def main():
    """Main function to run the topic analysis."""
    print("="*60)
    print("MISINFORMATION ARTICLES TOPIC ANALYSIS")
    print("="*60)
    
    # Initialize analyzer
    analyzer = TopicAnalyzer()
    
    # Define input files to analyze
    input_files = [
        "SourceFile.csv",
        "SourceFileTrue.csv",
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
        print(f"\n{'='*60}")
        print(f"ANALYZING: {file_path}")
        print(f"{'='*60}")
        
        try:
            # Generate output filename
            base_name = Path(file_path).stem
            output_csv = f"topic_analysis_{base_name}.csv"
            output_excel = f"topic_analysis_{base_name}.xlsx"
            
            # Perform analysis
            results_df = analyzer.analyze_dataset(file_path, output_excel)
            
            # Save CSV version as well
            results_df.to_csv(output_csv, index=False)
            
            # Create visualizations
            viz_dir = f"topic_analysis_plots_{base_name}"
            analyzer.create_visualizations(results_df, viz_dir)
            
            print(f"\n✓ Analysis completed successfully!")
            print(f"  Results saved to: {output_excel}")
            print(f"  Results saved to: {output_csv}")
            print(f"  Visualizations saved to: {viz_dir}/")
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            print(f"✗ Error analyzing {file_path}: {e}")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main() 