# SMS Spam Detection using Naive Bayes

This project builds a machine learning model using the **Naive Bayes algorithm** to classify SMS text messages as **Spam** or **Not Spam (Ham)**. It involves text preprocessing, feature engineering, training a probabilistic classifier, and evaluating the model using various performance metrics.


## Problem Statement

With the increasing volume of unsolicited messages, spam detection has become a crucial task in communication systems. The goal is to create an intelligent classifier that can predict whether an incoming SMS message is spam or not based on its textual content.


## Dataset

**Name:** [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/sahilnbajaj/sms-spam-collection)  
**Labels:**  
- `ham`: Not spam  
- `spam`: Spam messages


## Technologies Used

- Python
- Pandas & NumPy
- scikit-learn
- TfidfVectorizer (for feature extraction)

## Project Workflow

1. **Data Loading & Exploration**
   - Load SMS dataset and inspect distribution of labels.

2. **Text Preprocessing**
   - Convert to lowercase
   - Remove punctuation
   - Remove stopwords

3. **Feature Extraction**
   - TF-IDF vectorization to convert text into numeric features.

4. **Model Building**
   - Train a **Multinomial Naive Bayes** classifier.

5. **Evaluation**
   - Use accuracy, precision, recall, F1-score, and confusion matrix.
   - Compare results before and after tuning.

6. **Result**
   - Final model achieves high accuracy and identifies spam efficiently.


## Results

| Metric       | Score     |
|--------------|-----------|
| Accuracy     | ~98%      |
| Precision    | High      |
| Recall       | High      |
| F1-score     | High      |

- Naive Bayes performs exceptionally well on text classification tasks like spam detection due to the independence assumption and term-frequency modeling.


## Key Learnings

- Importance of **text preprocessing** in machine learning
- Leveraging **TF-IDF** for converting unstructured text into usable features
- Handling class imbalance and improving generalization


## Future Enhancements

- Add **Bigram/Trigram features** for better context
- Integrate **hyperparameter tuning** (e.g., smoothing parameter α)
- Deploy model using **Streamlit or Flask**
- Handle **non-English spam messages**
