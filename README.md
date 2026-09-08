# Product Review Sentiment Analysis Using Transformer Embeddings and Machine Learning

## 📌 Project Overview

This project develops a **sentiment analysis system** to classify customer product reviews into three categories:

* **Positive**
* **Neutral**
* **Negative**

The project uses a pre-trained **Sentence Transformer (`all-MiniLM-L6-v2`)** to convert product reviews into numerical sentence embeddings. These embeddings are then used as input to machine learning classification models.

## 🎯 Objective

The main objective is to automatically analyze customer reviews and identify their sentiment. This can help businesses understand customer opinions, identify product issues, and support data-driven decisions in product development and customer service.

## 📊 Dataset

The dataset contains **Product ID, Product Review, and Sentiment** columns. The sentiment variable contains three classes: Positive, Neutral, and Negative.

* **Rows:** 1,005
* **Features:** 3
* **Missing values:** None
* **Duplicate records:** Removed during preprocessing
* **Sentiment distribution:** Highly imbalanced, with positive reviews forming the majority.

## 🔄 Project Workflow

```text
Customer Product Reviews
          ↓
     Data Loading
          ↓
 Data Cleaning & Preprocessing
          ↓
Sentence Transformer Embeddings
     (all-MiniLM-L6-v2)
          ↓
   384-Dimensional Vectors
          ↓
     Train-Test Split
          ↓
 ┌───────────────────────┐
 │ Random Forest         │
 │ Gradient Boosting     │
 └───────────────────────┘
          ↓
 Model Evaluation
          ↓
Sentiment Prediction
```

The reviews were converted into **384-dimensional embeddings** using the Sentence Transformer model. The data was then split into **80% training and 20% testing**, while maintaining the sentiment class distribution.

## 🤖 Models Used

### 1. Random Forest + Transformer

Random Forest was trained using the Transformer-generated sentence embeddings.

**Test Performance:**

* Accuracy: **86.57%**
* Weighted F1 Score: **81.80%**

### 2. Gradient Boosting + Transformer

Gradient Boosting was also trained using the same Transformer embeddings.

**Test Performance:**

* Accuracy: **84.08%**
* Weighted F1 Score: **80.33%**

The Random Forest model achieved better test performance than Gradient Boosting.

## 📈 Model Comparison

| Model                           | Training Accuracy | Training F1 | Test Accuracy |    Test F1 |
| ------------------------------- | ----------------: | ----------: | ------------: | ---------: |
| Random Forest + Transformer     |           100.00% |     100.00% |    **86.57%** | **81.80%** |
| Gradient Boosting + Transformer |            99.88% |      99.88% |        84.08% |     80.33% |

The evaluation uses **accuracy** and **weighted F1 score** to assess model performance.

## 🏆 Final Model

**Random Forest + Transformer** was selected as the final model because it achieved higher test accuracy and F1 score than the Gradient Boosting model.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Sentence Transformers
* Matplotlib
* Seaborn
* Google Colab

## 🔮 Future Improvements

The model can be further improved by:

* Adding new customer reviews to the training data.
* Exploring advanced models such as **XGBoost and Support Vector Machines**.
* Experimenting with sequential neural networks.
* Fine-tuning Transformer-based models for the specific sentiment classification task.

## ✅ Conclusion

This project demonstrates how **Transformer embeddings combined with traditional machine learning models** can be used to classify customer product reviews. Among the evaluated models, **Random Forest + Transformer** achieved the best test performance and was selected as the final model.

The system can help businesses analyze customer feedback at scale, identify customer sentiment, and generate insights for improving products and customer experience.

