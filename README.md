# Sentiment Analyzer – Python & Scikit-learn
A command-line sentiment analysis tool built with Python and Scikit-learn, designed to clean review data, train a text classification model, and predict whether custom input is positive or negative — developed as a learning project to practice NLP preprocessing, feature extraction, and model evaluation with real-world data.
 
## ✨ | Features
* Clean raw review text: strip HTML tags and normalize whitespace
* Vectorize text data using TF-IDF (up to 50,000 features)
* Train a Logistic Regression model for binary sentiment classification (positive/negative)
* Evaluate model performance with accuracy, classification report, and confusion matrix
* Save the trained model and vectorizer locally via `joblib` for reuse
* Interactive CLI to test the trained model on custom, self-written sentences
* Displays prediction confidence for each input
## 🛠️ | Technologies & Libraries
* ``pandas`` - Loading and handling CSV review data
* ``scikit-learn`` - TF-IDF vectorization, Logistic Regression, train/test split, and evaluation metrics
* ``joblib`` - Saving and loading the trained model and vectorizer
* ``re`` - Text cleaning (removing HTML tags, extra whitespace)
* ``tkinter`` - Native file save dialog for cleaned data
## 📁 | Project Structure
```
sentiment-analyzer/
│
├── clean_data.py                     # Cleans raw review text (HTML tags, whitespace)
├── model_training.py                 # Trains and evaluates the TF-IDF + Logistic Regression model
├── main.py                           # CLI tool for testing the trained model on custom input
├── vectorizer.joblib                 # Saved TF-IDF vectorizer (generated after training)
├── logistic_regression_model.pkl     # Saved trained model (generated after training)
└── requirements.txt                  # Required packages
```
 
## 🚀 | Installation
 
1. Clone the repository
```bash
   git clone https://github.com/YOUR_USERNAME/sentiment-analyzer.git
   cd sentiment-analyzer
```
 
2. Install required packages
```bash
   pip install -r requirements.txt
```
 
3. Download the training dataset (e.g. the [IMDB 50k Movie Reviews dataset](https://ai.stanford.edu/~amaas/data/sentiment/) from Stanford University) and place it in the project folder.
<br><br/>
4. Clean the raw dataset
```bash
   python clean_data.py
```
   You will be prompted to enter the filepath of the raw CSV file. The cleaned data can then be saved via a native file dialog.
 
5. Train the model
```bash
   python model_training.py
```
   This trains the TF-IDF vectorizer and Logistic Regression model on the cleaned dataset, prints evaluation metrics, and saves `vectorizer.joblib` and `logistic_regression_model.pkl`.
 
6. Run the application
```bash
   python main.py
```
 
## 🗂️ | Usage
 
After launching `main.py`, you can test the trained model interactively:
 
```
Continue (c) or terminate (t): c
Input sentences (for TERMINATION: (1) press ENTER, (2) insert ### and step (1) again):
This movie was absolutely fantastic!
###
  →positive (Confidence: 94.32%)
```
 
Each input line is treated as a separate review. Enter `###` on its own line to run the prediction, or `t` at the main prompt to exit.
 
## 📊 | Model Evaluation
 
`model_training.py` automatically prints the following metrics after training:
* **Accuracy** – overall proportion of correct predictions
* **Classification report** – precision, recall, and F1-score per class
* **Confusion matrix** – breakdown of correct/incorrect predictions by class
## 📕 | Notes
* Developed using Python 3.14 and PyCharm
* The model was trained on the IMDB 50k movie review dataset and currently performs best on similarly styled review text
* This is an ongoing learning project focused on NLP and text classification fundamentals
## 🪲 | Bugs
* Feel free to report any bugs