# Kivy NLP Application

A comprehensive NLP text analysis tool built with Kivy and various NLP libraries.

## Features

### Basic Analysis
- Part of Speech Tagging
- Named Entity Recognition
- Dependency Parsing

### Advanced Analysis
- Sentiment Analysis
  - Overall sentiment polarity
  - Subjectivity analysis
  - Sentence-level sentiment

- Keyword Extraction
  - Word frequency analysis
  - TF-IDF based keywords
  - Top 10 important keywords

- Text Summary
  - Automatic text summarization
  - Sentence importance scoring
  - Original text comparison

- Language Detection
  - Multi-language support
  - Language code and name display
  - Support for major languages

- Text Similarity
  - Text comparison
  - Similarity scoring
  - Long text support

### New Features
- Text Statistics
  - Character, word, and sentence counts
  - Average word and sentence length
  - Vocabulary richness
  - Part of speech distribution

- Readability Analysis
  - Flesch Reading Ease
  - Flesch-Kincaid Grade Level
  - Gunning Fog Index
  - SMOG Index
  - Automated Readability Index
  - Coleman-Liau Index
  - Linsear Write Formula
  - Dale-Chall Readability Score

- Entity Relations
  - Entity pair identification
  - Relationship analysis
  - Entity network statistics
  - Entity type information

- Text Classification
  - Multi-category classification
  - Category scoring
  - Support for:
    - Technology
    - Business
    - Science
    - Health
    - Education

## Installation

1. Make sure you have Python 3.7 or higher installed
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download required models:
   ```bash
   python -m spacy download en_core_web_sm
   python -m nltk.downloader punkt stopwords
   ```

## Running the Application

```bash
python main.py
```

## Usage Instructions

1. Enter text in the input box
2. Select analysis type from the dropdown menu
3. For text similarity analysis, enter a second text in the additional input box
4. Click "Analyze Text" button
5. View the analysis results in the output box

## Requirements

- Python 3.7+
- Kivy 2.2.1
- spaCy 3.7.2
- TextBlob 0.17.1
- NLTK 3.8.1
- scikit-learn 1.3.0
- langdetect 1.0.9
- textstat 0.7.3
- networkx 3.1
- matplotlib 3.7.1

## Notes

- The application currently supports English text analysis
- Some features may require internet connection for model downloads
- For best results, use clear and well-formatted text
- Long texts may take longer to process 