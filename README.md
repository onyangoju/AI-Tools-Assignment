
# 🛠️ AI Tools and Applications – "Mastering the AI Toolkit"

## 📚 Overview

This project demonstrates practical mastery of core AI tools and frameworks including **TensorFlow**, **PyTorch**, **Scikit-learn**, and **spaCy**. It is structured into three parts: theoretical understanding, hands-on implementation, and ethical reflection with model optimization.

> ⚠️ **Note:** This assignment was originally intended as a **group project**. However, I completed it **individually** to showcase full-stack AI engineering skills and personal commitment to mastering the AI toolkit.

---

## 🔍 Contents

| Section | Description |
|--------|-------------|
| 📘 Part 1 | Theoretical questions on AI frameworks and tools |
| 🧪 Part 2 | Practical implementations: Iris classifier, MNIST CNN, Amazon NLP |
| ⚖️ Part 3 | Ethical considerations & troubleshooting |
| 🚀 Bonus | Streamlit app deployment of MNIST classifier |

---

## 📘 Part 1: Theoretical Understanding

> Located in `/theory/answers.pdf`

- ✅ TensorFlow vs PyTorch comparison
- ✅ spaCy vs basic Python NLP
- ✅ Use cases for Jupyter Notebooks
- ✅ Comparative table: Scikit-learn vs TensorFlow

---

## 🧪 Part 2: Practical Implementation

### Task 1: 🏵️ Classical ML (Scikit-learn + Iris Dataset)
- Dataset: `Iris.csv`
- Model: Decision Tree Classifier using `scikit-learn`
- Preprocessing with `pandas`, evaluation using accuracy, precision, and recall
- Script: [`/iris_classical_ml/iris_decision_tree_classifier.py`](iris_classical_ml/iris_decision_tree_classifier.py)

### Task 2: 🧠 Deep Learning (TensorFlow + MNIST)
- CNN model for handwritten digit recognition
- Achieved **>98%** test accuracy
- Saved trained model in `.keras` format
- Script: [`/mnist_cnn/mnist_cnn_classifier.py`](mnist_cnn/mnist_cnn_classifier.py)  
- Model file: [`/mnist_cnn/mnist_cnn_model.keras`](mnist_cnn/mnist_cnn_model.keras)

### Task 3: 💬 NLP (spaCy + Amazon Reviews)
- Named Entity Recognition (NER) using `spaCy`
- Rule-based sentiment analysis using `TextBlob`
- Exported output to `.csv` and `.json`
- Visualized entity frequency and sentiment distribution
- Script: [`/amazon_reviews_nlp/amazon_ner_sentiment.py`](amazon_reviews_nlp/amazon_ner_sentiment.py)  
- Outputs:
  - [`review_sentiment_entities.csv`](amazon_reviews_nlp/review_sentiment_entities.csv)
  - [`review_sentiment_entities.json`](amazon_reviews_nlp/review_sentiment_entities.json)
  - [`entity_frequency.png`](amazon_reviews_nlp/entity_frequency.png)
  - [`sentiment_distribution.png`](amazon_reviews_nlp/sentiment_distribution.png)

---

## ⚖️ Part 3: Ethics & Optimization

- Bias identified in both MNIST and sentiment analysis models
- Tools like `TensorFlow Fairness Indicators` and spaCy’s rule-based pipelines discussed as mitigation strategies
- Buggy TensorFlow code fixed (loss function and input shape issues)
- Details in: [`/theory/ethical_analysis.pdf`](theory/ethical_analysis.pdf)

---

## 🚀 Bonus: Streamlit Deployment

### 📱 MNIST Digit Classifier Web App
- Draw a digit (0–9) on canvas (28x28 resolution)
- The app predicts the digit and shows prediction confidence
- Inverts image automatically if needed (white-on-black or vice versa)

🔗 **Live Demo**: [https://ai-tools-assignment-8le39fye64cjekzkcqtb7t.streamlit.app/](https://ai-tools-assignment-8le39fye64cjekzkcqtb7t.streamlit.app/)  
📂 Code: [`/streamlit_app/mnist_app.py`](streamlit_app/mnist_app.py)

---

## 📄 Report & Presentation

- 📄 [Final Report PDF](report/AI_Tools_Assignment_Report.pdf)
- 🎥 [3-Minute Video Presentation](https://my-presentation-link.com)

---

## 🛠️ Technologies Used

- `TensorFlow` – CNN on MNIST
- `Scikit-learn` – Decision Trees
- `spaCy` – Named Entity Recognition
- `TextBlob` – Sentiment analysis
- `matplotlib`, `pandas`, `seaborn`
- `Streamlit` – App deployment
- `streamlit-drawable-canvas` – Interactive digit drawing

---

## 🙋 About Me

This project was completed **individually**, although it was intended as a **group assignment** for the **PLP Academy AI Development Software Tools** module. I took this route to fully explore and apply the AI toolkit from theory to deployment.

**Name:** Pauline Onyango  
**Platform:** PLP Academy

---

## ✅ How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/onyangoju/AI-Tools-Assignment.git
cd AI-Tools-Assignment

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit app
streamlit run streamlit_app/mnist_app.py
```
