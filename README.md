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

### Evaluation Metrics

Accuracy: Measures the overall correctness of the model (simply the percentage of correct predictions).
Precision: Shows the accuracy of the positive predictions (how many of the predicted positives were actually positive).
Recall: Shows how well the model distinguishes actual positives (how many of the true positives were correctly identified).
F1-Score: The harmonic mean of precision and recall.
Confusion Matrix: Visualizes the true positive, false positive, true negative, and false negative predictions.

### Methods to Improve Model Performance

Here are some tips to improve the accuracy and performance of the sentiment analysis model:

- **Advanced Text Preprocessing**:
You can try to replace stemming with lemmatization for more accurate word normalization.

- **Use TF-IDF Vectorizer**:
Replace CountVectorizer with TfidfVectorizer to give less common but important words more weight.

- **Model Selection**:
Experiment with models like Logistic Regression, SVM, or Random Forest for possibly better performance.

- **Hyperparameter Tuning**:
Use GridSearchCV or RandomizedSearchCV to tune model parameters for optimal performance.

- **Data Augmentation**:
Increase the diversity of training data by adding more labeled reviews or synthetically augmenting existing data.

- **Deep Learning Models**:
For more advanced methods, you may use Deep Learning models such as LSTM or BERT for state-of-the-art text classification.

### 📝 Example Results
- **Model Accuracy**: 74%

- Precision (Positive): 0.75
- Recall (Positive): 0.74
- F1-Score (Positive): 0.75
- Precision (Negative): 0.72
- Recall (Negative): 0.74
- F1-Score (Negative): 0.73


