# Kivy NLP Application

A simple NLP text analysis tool built with Kivy and spaCy.

## Features

- Part of Speech Tagging
- Named Entity Recognition
- Dependency Parsing

## Installation

1. Make sure you have Python 3.7 or higher installed
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Running the Application

```bash
python main.py
```

## Usage Instructions

1. Enter English text in the input box
2. Click the "Analyze Text" button
3. View the analysis results, including:
   - Part of Speech Tagging
   - Named Entity Recognition
   - Dependency Parsing 