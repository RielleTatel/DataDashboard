# Topic Analysis Script for Misinformation Articles

## Overview

This professional-grade topic analysis script is designed to categorize misinformation articles based on their text content. It uses multiple advanced NLP techniques to provide comprehensive topic classification.

## Features

- **Multiple Analysis Methods**: Keyword-based, LDA, NMF, and BERT-based topic modeling
- **Predefined Misinformation Topics**: Politics, Health, Conspiracy, Science, Social Media, Economy, Crime, Technology, Environment, Entertainment
- **Advanced Text Preprocessing**: Tokenization, lemmatization, stopword removal
- **Comprehensive Output**: Excel/CSV files with all analysis results
- **Visualization**: Automatic generation of charts and word clouds
- **Confidence Scoring**: Each topic classification includes confidence scores

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "import pandas, nltk, sklearn, transformers; print('All packages installed successfully!')"
```

## Usage

### Quick Start

Simply run the script to analyze all available datasets:

```bash
python topicScript.py
```

The script will automatically:
- Find and analyze all CSV and Excel files in your workspace
- Generate topic classifications for each article
- Save results to Excel and CSV files
- Create visualization plots

### Manual Usage

```python
from topicScript import TopicAnalyzer

# Initialize the analyzer
analyzer = TopicAnalyzer()

# Analyze a specific file
results = analyzer.analyze_dataset("path/to/your/file.csv")

# Create visualizations
analyzer.create_visualizations(results, "output_plots/")
```

## Input Data Format

Your dataset should contain these columns:
- `TITLE`: Article title
- `TEXT`: Main article text (this is what gets analyzed)
- `SUBJECT`: Subject category
- `DATE`: Publication date

## Output Files

For each input file, the script generates:

1. **Excel File**: `topic_analysis_[filename].xlsx`
2. **CSV File**: `topic_analysis_[filename].csv`
3. **Visualization Directory**: `topic_analysis_plots_[filename]/`

### Output Columns

- Original columns (TITLE, TEXT, SUBJECT, DATE)
- `TEXT_PREPROCESSED`: Cleaned text for analysis
- `KEYWORD_TOPIC`: Topic based on keyword matching
- `KEYWORD_CONFIDENCE`: Confidence score for keyword classification
- `LDA_TOPIC`: Topic from Latent Dirichlet Allocation
- `LDA_CONFIDENCE`: LDA confidence score
- `NMF_TOPIC`: Topic from Non-negative Matrix Factorization
- `NMF_CONFIDENCE`: NMF confidence score
- `BERT_TOPIC`: Topic from BERT clustering (if available)
- `BERT_CONFIDENCE`: BERT confidence score
- `TOP_KEYWORDS`: Top 5 keywords for each article
- `FINAL_TOPIC`: Best topic classification
- `FINAL_CONFIDENCE`: Best confidence score

## Topic Categories

The script uses these predefined categories optimized for misinformation detection:

1. **Politics**: Elections, politicians, government policies
2. **Health**: Vaccines, COVID, medical treatments
3. **Conspiracy**: Secret theories, cover-ups, revelations
4. **Science**: Research, studies, scientific evidence
5. **Social Media**: Social platforms, viral content
6. **Economy**: Financial markets, business, investments
7. **Crime**: Criminal activities, law enforcement
8. **Technology**: Tech news, digital developments
9. **Environment**: Climate change, environmental issues
10. **Entertainment**: Celebrity news, entertainment industry

## Analysis Methods

### 1. Keyword-Based Classification
- Uses predefined keyword lists for each topic
- Fast and interpretable
- Good for clear topic identification

### 2. LDA (Latent Dirichlet Allocation)
- Probabilistic topic modeling
- Discovers hidden topics in the data
- Good for finding underlying themes

### 3. NMF (Non-negative Matrix Factorization)
- Matrix factorization approach
- Often produces more coherent topics
- Good for document clustering

### 4. BERT-Based Analysis
- Uses state-of-the-art language models
- Captures semantic meaning
- Most accurate but requires more computational resources

## Performance Tips

### For Large Datasets
- The script automatically handles large files
- Consider using a subset for initial testing
- BERT analysis may take longer on large datasets

### Memory Optimization
- The script processes data in chunks
- Close other applications if running on limited RAM
- Consider using the sampled data files for testing

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **NLTK Data Missing**: The script automatically downloads required NLTK data

3. **BERT Not Available**: The script will work without BERT, just with reduced accuracy

4. **Memory Issues**: Try analyzing smaller files first or use sampled data

### Getting Help

If you encounter issues:
1. Check that all dependencies are installed
2. Verify your input files have the correct format
3. Check the console output for specific error messages

## Example Output

```
============================================================
TOPIC ANALYSIS SUMMARY
============================================================

Total articles analyzed: 15,000

Keyword-based Topic Distribution:
  politics: 3,200 articles (21.3%)
  health: 2,800 articles (18.7%)
  conspiracy: 2,100 articles (14.0%)
  science: 1,900 articles (12.7%)
  social_media: 1,500 articles (10.0%)

Confidence Statistics:
  Average keyword confidence: 0.745
  Average LDA confidence: 0.823
  Average NMF confidence: 0.791
```

## Advanced Usage

### Custom Topic Categories

You can modify the predefined topics in the script:

```python
analyzer = TopicAnalyzer()
analyzer.misinfo_topics['custom_topic'] = ['keyword1', 'keyword2', 'keyword3']
```

### Adjusting Analysis Parameters

```python
# Change number of topics for LDA/NMF
lda_results = analyzer.perform_lda_analysis(texts, n_topics=15)

# Adjust TF-IDF parameters
tfidf_keywords = analyzer.extract_keywords_tfidf(texts, max_features=2000)
```

## License

This script is provided for educational and research purposes. Please ensure you have appropriate permissions for the data you're analyzing. 