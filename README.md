# Cardiovascular Disease Prediction

A supervised machine learning project that predicts the presence of cardiovascular disease from patient health metrics. Five classification algorithms are trained, compared, and evaluated on a dataset of 70,000 patient records — with a focus on **recall** as the primary metric, since missing a sick patient is clinically more costly than a false alarm.

---

## 🧠 What It Does

- Loads and cleans a 70,000-record cardiovascular dataset with domain-aware outlier removal
- Performs exploratory data analysis across age, gender, cholesterol, and blood pressure
- Trains and compares **5 ML classifiers** — evaluating accuracy, recall, and F1 score
- Validates the best model with **5-fold cross-validation**
- Visualizes **Random Forest feature importance** to identify key risk factors
- Saves the best model and scaler using `joblib` for deployment

---

## 📊 Models Compared

| Model | Metric Focus |
|---|---|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Interpretable, no scaling needed |
| Random Forest | Ensemble, best feature importance |
| K-Nearest Neighbors | Distance-based, requires scaling |
| Support Vector Machine | Margin-based, requires scaling |

---

## 📈 Key Results

> Run the notebook to generate your own results. Typical performance on this dataset:
- **Best model:** Random Forest
- **Accuracy:** ~73%
- **Recall (Disease class):** ~70%
- **F1 Score:** ~72%
- **5-Fold CV F1:** ~71% ± 1% (stable, no overfitting)

Top predictive features: **systolic blood pressure**, **age**, **cholesterol level**

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.8+ |
| Data Processing | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | scikit-learn |
| Model Persistence | joblib |
| Environment | Jupyter Notebook |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

### Run
```bash
git clone https://github.com/akashcpatil111/cardiovascular-disease-prediction.git
cd cardiovascular-disease-prediction
jupyter notebook cardiovascular_disease_prediction.ipynb
```

> Make sure `cardio_train.csv` is in the same directory as the notebook.

---

## 📁 Project Structure

```
cardiovascular-disease-prediction/
├── cardiovascular_disease_prediction.ipynb   ← Main notebook
├── cardio_train.csv                          ← Dataset (70,000 records)
├── cardio_best_model.pkl                     ← Saved best model (generated on run)
├── cardio_scaler.pkl                         ← Saved StandardScaler (generated on run)
└── README.md
```

---

## 📋 Dataset

**Source:** [Kaggle — Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)

**Features:** Age, gender, height, weight, systolic/diastolic blood pressure, cholesterol, glucose, smoking, alcohol, physical activity

**Target:** `cardio` — 0 (no disease) / 1 (disease present)

**Preprocessing applied:**
- Age converted from days to years
- Blood pressure outliers removed using physiological bounds (systolic > diastolic, realistic ranges)
- Extreme height/weight values filtered

---

## 📄 License

MIT License
