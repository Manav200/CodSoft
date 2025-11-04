# CodSoft

# Codsoft Machine Learning Internship

This repository contains three machine learning projects completed during the **Codsoft Internship in Machine Learning**.  
Each project demonstrates end-to-end data science workflows — from data preprocessing and feature engineering to model training, evaluation, and performance optimization.

---

## 📁 Projects Overview

### 1. Movie Genre Classification
**Goal:** Predict the genre of a movie based on textual metadata such as title, overview, or keywords.  

**Techniques & Tools:**
- Natural Language Processing (TF-IDF Vectorization)
- Multiclass Classification
- Algorithms: Logistic Regression, Random Forest, Naive Bayes
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score

**Key Steps:**
1. Text preprocessing (tokenization, stopword removal, stemming)
2. Feature extraction with TF-IDF
3. Model training and evaluation
4. Genre prediction for new movie descriptions

**Notebook:** [`Movie_Genre_Classification.ipynb`](./Movie_Genre_Classification.ipynb)

---

### 2. Credit Card Fraud Detection
**Goal:** Identify fraudulent transactions from anonymized credit card data.

**Techniques & Tools:**
- Binary Classification
- Data balancing using SMOTE
- Algorithms: Logistic Regression, Random Forest, XGBoost
- Evaluation Metrics: ROC-AUC, Precision, Recall, F1-score, Confusion Matrix

**Key Steps:**
1. Data exploration and cleaning
2. Handling class imbalance (fraud vs. non-fraud)
3. Feature scaling and model training
4. Evaluation and model comparison

**Notebook:** [`Credit_Card_Fraud_Detection.ipynb`](./Credit_Card_Fraud_Detection.ipynb)

---

### 3. Spam SMS Detection
**Goal:** Classify SMS messages as spam or legitimate (ham).

**Techniques & Tools:**
- NLP and Text Classification
- CountVectorizer / TF-IDF Vectorization
- Algorithms: Naive Bayes, SVM, Logistic Regression
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score

**Key Steps:**
1. Data preprocessing (cleaning and normalization)
2. Text vectorization
3. Model training and evaluation
4. Spam detection prediction on custom messages

**Notebook:** [`Spam_SMS_Detection.ipynb`](./Spam_SMS_Detection.ipynb)

---

## 🧠 Skills Demonstrated
- Machine Learning Workflow Design  
- Data Cleaning & Feature Engineering  
- Supervised Learning Algorithms  
- Model Evaluation & Validation  
- NLP Fundamentals  
- Data Visualization (Matplotlib, Seaborn)

---

## 🛠️ Technologies Used
- **Programming Language:** Python  
- **Libraries:** NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, XGBoost, NLTK  
- **Environment:** Jupyter Notebook

---

## 📊 Results Summary
| Project | Best Model | Accuracy | Key Metric |
|----------|-------------|-----------|-------------|
| Movie Genre Classification | Logistic Regression | ~85% | F1-score |
| Credit Card Fraud Detection | Random Forest | ~99% | ROC-AUC |
| Spam SMS Detection | Multinomial Naive Bayes | ~97% | Precision |

*(Exact values depend on dataset versions and preprocessing.)*

---

## 🚀 How to Run
1. Clone the repository:
   git clone https://github.com/<your-username>/Codsoft-ML-Internship.git
   cd Codsoft-ML-Internship

2. Open the notebooks in Jupyter:

   jupyter notebook
   
4. Run each `.ipynb` file sequentially.

---

## 📚 Acknowledgment

This repository was developed as part of the **Codsoft Machine Learning Internship Program**, focused on applying theoretical ML knowledge to real-world datasets.

---

## 🧾 License

This project is open-source under the [MIT License](LICENSE).

---

### Author

**Name:** *Manav Mahawar*
**Email:** *manavmahawar04@gmail.com*
**LinkedIn:** www.linkedin.com/in/manav-mahawar-a7a92828a

---
