# 🚢 Titanic Survival Prediction — Machine Learning Project

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-red?style=for-the-badge&logo=jupyter&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge"/>
</p>

<p align="center">
  <b>An end-to-end Machine Learning project predicting Titanic passenger survival using 4 classification algorithms.</b>
</p>

---

## 📌 About The Project

This project was built as part of my **Machine Learning Internship**. The goal is to predict whether a passenger survived the Titanic disaster based on features like age, gender, ticket class, and family size.

The project covers the **complete ML pipeline** — from raw data to a tuned, deployment-ready model — and compares multiple algorithms to find the best performer.

---

## 🎯 Project Objectives

- ✅ Perform in-depth **Exploratory Data Analysis (EDA)**
- ✅ Apply **Feature Engineering** to create meaningful new features
- ✅ Train and compare **4 Machine Learning algorithms**
- ✅ Evaluate using **Confusion Matrix, Classification Report & Cross-Validation**
- ✅ Optimize the best model using **GridSearchCV Hyperparameter Tuning**
- ✅ Visualize **Feature Importance** to understand key survival factors

---

## 📂 Project Structure

```
titanic-survival-prediction/
│
├── 📓 titanic_ml_project.ipynb   ← Main Jupyter Notebook (all code)
├── 📄 README.md                  ← Project Documentation
└── 📋 requirements.txt           ← Required Libraries
```

---

## 🗂️ Dataset

| Property | Details |
|----------|---------|
| **Source** | Seaborn Built-in (`sns.load_dataset('titanic')`) |
| **Rows** | 891 passengers |
| **Columns** | 15 features |
| **Target** | `survived` (0 = No, 1 = Yes) |
| **Download** | ❌ Not required — loads automatically! |

### 📊 Key Features Used

| Feature | Description |
|---------|-------------|
| `pclass` | Passenger class (1st, 2nd, 3rd) |
| `sex` | Gender of passenger |
| `age` | Age of passenger |
| `fare` | Ticket fare paid |
| `embarked` | Port of embarkation |
| `family_size` | 🔧 *Engineered* — Total family members aboard |
| `is_alone` | 🔧 *Engineered* — Travelling alone or not |

---

## 🔬 Project Workflow

```
📦 Load Data
     ↓
📊 Exploratory Data Analysis (EDA)
     ↓
🛠️  Feature Engineering
     ↓
🔧 Data Preprocessing (Missing Values, Encoding, Scaling)
     ↓
🤖 Train 4 ML Models
     ↓
📈 Evaluate & Compare Models
     ↓
⚙️  Hyperparameter Tuning (GridSearchCV)
     ↓
🎯 Final Predictions
```

---

## 🤖 ML Models Compared

| # | Model | Type | Scaling Needed |
|---|-------|------|---------------|
| 1 | **Logistic Regression** | Linear | ✅ Yes |
| 2 | **Decision Tree** | Tree-based | ❌ No |
| 3 | **Support Vector Machine (SVM)** | Kernel-based | ✅ Yes |
| 4 | **Random Forest** ⭐ | Ensemble | ❌ No |

---

## 📈 Results

### Model Accuracy Comparison

| Model | Test Accuracy |
|-------|--------------|
| Logistic Regression | ~80% |
| Decision Tree | ~79% |
| Support Vector Machine | ~82% |
| Random Forest | ~83% |
| **Random Forest (Tuned)** ⭐ | **~84%+** |

> ⭐ **Best Model: Tuned Random Forest** (after GridSearchCV optimization)

### 🔑 Key Insights from EDA

- 👩 **Females** had a significantly higher survival rate (~74%) vs males (~19%)
- 🎩 **1st Class** passengers had the highest survival rate (~63%)
- 👨‍👩‍👧 Passengers with **small families (2-4)** survived more than those alone or in large groups
- 💰 Higher **fare** correlated with better survival chances

---

## 🛠️ Feature Engineering Highlights

```python
# Family Size — total people aboard with the passenger
data['family_size'] = data['sibsp'] + data['parch'] + 1

# Is Alone — binary flag for solo travellers
data['is_alone'] = (data['family_size'] == 1).astype(int)
```

---

## ⚙️ Hyperparameter Tuning

Used **GridSearchCV** with 5-fold cross-validation on Random Forest:

```python
param_grid = {
    'n_estimators'      : [50, 100, 200],
    'max_depth'         : [3, 5, 7, None],
    'min_samples_split' : [2, 5, 10],
    'max_features'      : ['sqrt', 'log2']
}
```

---

## 🚀 How to Run This Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/titanic-survival-prediction.git
cd titanic-survival-prediction
```

### 2️⃣ Install Required Libraries
```bash
pip install -r requirements.txt
```

### 3️⃣ Open the Notebook
```bash
jupyter notebook titanic_ml_project.ipynb
```

### 4️⃣ Run All Cells
Click **"Run All"** or execute cells one by one from top to bottom. 

> 💡 **No dataset download needed!** The dataset loads automatically via `sns.load_dataset('titanic')`

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

## 📊 Visualizations Included

- 📊 Survival count plot & pie chart
- 👫 Survival rate by gender
- 🎩 Survival rate by passenger class
- 📅 Age distribution & KDE by survival
- 🔥 Correlation heatmap
- 👨‍👩‍👧 Survival rate by family size
- 📉 Model accuracy comparison bar chart
- 🧩 Confusion matrices for all 4 models
- 🌲 Random Forest feature importance chart

---

## 🧑‍💻 About Me

Hi! I'm a Computer Science student currently doing an **ML Internship** where I'm building hands-on machine learning projects to strengthen my skills.

- 🔗 **LinkedIn:** [https://www.linkedin.com/in/ankit-kumar-singh-862031347/]
- 🐙 **GitHub:** [https://github.com/Ankit5641]
- 📧 **Email:** ankitrajput5641@email.com

---

## 📝 Learnings from This Project

> *"This project taught me that feature engineering and model comparison matter just as much as the algorithm choice itself. A well-tuned Random Forest with meaningful features outperforms a complex model on raw data."*

---

## ⭐ Show Your Support

If you found this project helpful or interesting, please consider giving it a **⭐ Star** — it means a lot and motivates me to build more!

---

<p align="center">
  Made with ❤️ by a passionate ML student
</p>
