# OIBSIP Data Science Task 4 – Email Spam Detection

## Objective
The objective of this project is to build a machine learning model that can classify emails as **spam** or **not spam** (ham) based on their content. This is a real-world application of Natural Language Processing (NLP) and classification algorithms.

## Dataset Details

The dataset used is a collection of labeled SMS/email messages with two columns:

| Column | Description                      |
|--------|----------------------------------|
| label  | Class of the message (ham/spam)  |
| text   | The content of the email/SMS     |

Dataset file: `spam.csv`

**Note:** Some preprocessing is required to drop extra unnamed columns and convert labels to binary format.

## Steps Performed

### 1. Data Loading
- Imported the dataset using pandas
- Cleaned column names and dropped unnecessary columns

### 2. Data Preprocessing
- Converted labels: `ham` to 0 and `spam` to 1
- Cleaned and normalized the text
  - Lowercased all text
  - Removed punctuation and stopwords
  - Tokenized words
- Applied **TF-IDF Vectorization** to convert text into numerical format

### 3. Model Building
- Trained multiple classification models:
  - Multinomial Naive Bayes
  - Logistic Regression
  - Support Vector Machine (SVM)

### 4. Model Evaluation
- Used accuracy score, precision, recall, and F1-score
- Visualized the confusion matrix for each model
- Compared ROC curves for model performance

## Tools and Libraries Used

- Python
- pandas, numpy
- sklearn
- nltk (for text preprocessing)
- matplotlib, seaborn

## Key Insights

- **Naive Bayes** performed exceptionally well due to its probabilistic nature, achieving high precision and recall.
- Most spam emails contain certain recurring patterns and keywords that were effectively captured using TF-IDF.
- The final model was able to **accurately classify over 97%** of messages.



