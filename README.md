# Sentiment-Analysis-for-Restaurant-Reviews

## 📋 Overview

This project conducts sentiment analysis on restaurant reviews, grouping them as either positive or negative. It employs machine learning techniques, specifically the Naive Bayes algorithm, to decide the sentiment of the textual reviews. The main steps are text preprocessing, feature extraction, model training, evaluation, and saving the model for future use.

### Features

- Text Preprocessing: Cleans, tokenizes, and removes stopwords from the review text.
- Model Training: Uses Naive Bayes classification to train the model.
- Model Evaluation: Provides detailed performance metrics including accuracy, precision, recall, and F1-score.
- Confusion Matrix: Visualizes the model's performance in predicting positive vs. negative reviews.
- Model Saving: Saves the trained model and vectorizer for later use and deployment.

### Requirements

Import and install necessary libraries if you are using Google Colab.
Libraries: pandas, numpy, scikit-learn, nltk, matplotlib

### How It Works

1. Load Dataset: The dataset is loaded from a .tsv file that contains restaurant reviews accompanied with their sentiment labels (positive or negative).

2. Text Preprocessing: Clean the review text (remove non-alphabetic characters, convert to lowercase). Also, apply stemming and remove stopwords.

3. Feature Extraction: Convert the processed reviews into numerical features using CountVectorizer or TfidfVectorizer.

4. Model Training:A Naive Bayes classifier is trained with the processed data.

5. Model Evaluation: Evaluate the model performance using accuracy, precision, recall, F1-score, and the confusion matrix.

6. Save Model: The trained model and vectorizer are saved as a .pkl file for future use.



