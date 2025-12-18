
## Heart-disease-and-health-ml
This repository contains a collection of **foundational Machine Learning projects** implemented using **Python and Jupyter Notebooks**.  
The goal of this project is to build a **strong practical understanding of the complete Machine Learning pipeline**, starting from raw data and ending with trained and evaluated models.

The repository focuses on **Supervised Learning**, covering both:
- **Classification**
- **Regression**

using **real-world datasets** rather than synthetic examples.

---

## 📖 Motivation

Machine Learning is best learned by working on real data and solving practical problems.  
This project was created to:

- Understand how raw datasets are handled in practice
- Learn data preprocessing and feature engineering
- Apply exploratory data analysis (EDA) to discover patterns
- Train and evaluate basic Machine Learning models
- Build intuition around model performance and limitations

This repository represents **Part 1** of my Machine Learning journey and focuses on **core concepts that form the foundation for advanced ML and AI topics**.

---

## 📂 Repository Structure

```

heart-disease-and-health-ml/
│
├── Heart.ipynb
├── heart.csv
│
├── insurance.ipynb
├── insurance.csv
│
└── anaconda_projects/

````

### File Description
- **`.ipynb` files** contain the full Machine Learning workflow with explanations and outputs
- **`.csv` files** are the datasets used for training and testing models
- **`anaconda_projects/`** contains environment-related project data

---

## 📊 Projects Overview

### 1️⃣ Heart Disease Prediction (Classification)

**Notebook:** `Heart.ipynb`  
**Dataset:** `heart.csv`  
**Problem Type:** Binary Classification  

#### 📌 Problem Statement
The objective of this project is to predict whether a person is likely to have **heart disease** based on multiple medical attributes.

The dataset includes features such as:
- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol levels
- Maximum heart rate
- Exercise-induced angina
- Other clinical indicators

The target variable indicates the **presence or absence of heart disease**.

---

#### 🧠 Approach & Methodology

The project follows a structured Machine Learning pipeline:

1. **Data Loading**
   - Load dataset using Pandas
   - Inspect shape, data types, and missing values

2. **Data Preprocessing**
   - Handle categorical and numerical features
   - Check for null or inconsistent values
   - Prepare data for model training

3. **Exploratory Data Analysis (EDA)**
   - Analyze feature distributions
   - Study relationships between features and target variable
   - Identify important predictors for heart disease

4. **Feature Selection**
   - Select relevant features based on data understanding
   - Reduce noise and improve model learning

5. **Model Training**
   - Apply supervised classification algorithms
   - Split data into training and testing sets

6. **Model Evaluation**
   - Evaluate performance using accuracy and other metrics
   - Analyze predictions and misclassifications

---

#### 🎯 Learning Outcomes (Heart Disease Project)

- Understanding classification problems
- Working with medical datasets
- Interpreting model predictions
- Evaluating classification performance
- Learning the importance of data quality in healthcare ML problems

---

### 2️⃣ Insurance Cost Prediction (Regression)

**Notebook:** `insurance.ipynb`  
**Dataset:** `insurance.csv`  
**Problem Type:** Regression  

#### 📌 Problem Statement
The objective of this project is to predict **medical insurance charges** for individuals based on personal and lifestyle attributes.

The dataset includes:
- Age
- Gender
- Body Mass Index (BMI)
- Number of children
- Smoking habits
- Region
- Insurance charges (target variable)

This is a **regression problem**, where the output is a continuous numerical value.

---

#### 🧠 Approach & Methodology

1. **Data Exploration**
   - Understand dataset structure
   - Identify numerical and categorical variables

2. **Data Preprocessing**
   - Encode categorical variables
   - Normalize or scale numerical features if required

3. **Exploratory Data Analysis (EDA)**
   - Analyze how factors like smoking and BMI affect insurance costs
   - Visualize relationships between features and target variable

4. **Model Building**
   - Apply regression algorithms (e.g., Linear Regression)
   - Train the model using processed data

5. **Model Evaluation**
   - Measure performance using regression metrics
   - Analyze prediction errors and limitations

---

#### 🎯 Learning Outcomes (Insurance Project)

- Understanding regression problems
- Handling categorical data in ML
- Interpreting continuous predictions
- Learning how real-world economic factors influence outcomes
- Understanding model assumptions and error patterns

---

## 🛠️ Technologies & Libraries Used

- **Python 3**
- **Jupyter Notebook**
- **Pandas** – data manipulation
- **NumPy** – numerical computing
- **Matplotlib & Seaborn** – data visualization
- **Scikit-learn** – Machine Learning models and evaluation

---

## 🚀 How to Run the Project

1. Clone this repository:
   ```bash
   git clone https://github.com/wittyswayam/heart-disease-and-health-ml.git
````

2. Navigate to the project directory:

   ```bash
   cd heart-disease-and-health-ml
   ```

3. Install required libraries (if not already installed):

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

4. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open and run:

   * `Heart.ipynb`
   * `insurance.ipynb`

---

## 📌 Key Skills Demonstrated

* End-to-end Machine Learning workflow
* Data preprocessing and cleaning
* Exploratory Data Analysis (EDA)
* Supervised learning (classification & regression)
* Model evaluation and interpretation
* Working with real-world datasets

---

## 🔮 Future Improvements

* Add more advanced models (Decision Trees, Random Forests)
* Perform hyperparameter tuning
* Improve feature engineering
* Add cross-validation
---
