# Smartphone Battery & Academic Performance Analysis
### EDA Assignment 2 – Team 6

---

## 1. Project Overview

This project studies whether **smartphone usage and battery habits** are related to the **academic performance (CGPA)** of university students.

We take a survey-style dataset, clean it, explore it with visualizations, test relationships statistically, and finally build regression models to find out which factors matter most for CGPA.

### Research questions
1. Does higher daily screen time go with a lower CGPA?
2. Does daily study time have a stronger effect on CGPA than phone-related factors?
3. Do battery health, charge cycles or overnight charging have any relation with CGPA?
4. Which features are the most important predictors of CGPA?
5. Which model predicts CGPA best?

---

## 2. Files in this Submission

| File | What it contains |
|------|------------------|
| `EDA_Assignment_2_Team6.ipynb` | Main notebook: cleaning, EDA, charts, statistical tests, modelling |
| `colab_battery_academic_analysis.py` | Single-cell Colab script version (generates data, cleans, plots, tests) |
| `Team_6_-_original_data__cleaned_data.xlsx` | Excel workbook with 2 sheets: **Original Data** and **Cleaned Data** |
| `model_comparison.csv` | Cross-validation and test results of all models |
| `feature_importance.csv` | Feature importance from Random Forest, permutation importance and linear coefficients |
| `README.md` | This file |

---

## 3. Dataset Description

The data is **synthetic / sample survey data** created for this assignment, so results show the method, not real-world proof.

| Column | Meaning | Type |
|--------|---------|------|
| `student_id` | Unique student ID (S001, S002, ...) | text |
| `age` | Age of the student (years) | integer |
| `gender` | Female / Male | category |
| `year_of_study` | Year of study (1 to 4) | integer |
| `daily_screen_time_hrs` | Average phone screen time per day (hours) | float |
| `daily_charge_cycles` | Number of times phone is charged per day | integer |
| `battery_health_pct` | Battery health of the phone (%) | integer |
| `charges_overnight` | Does the student charge the phone overnight? (Yes / No) | category |
| `daily_study_hrs` | Average study hours per day | float |
| `attendance_pct` | Class attendance (%) | integer |
| `cgpa` | Cumulative GPA (target variable) | float |

**Target variable:** `cgpa`

---

## 4. Data Cleaning

### Raw data problems found

| Problem | Details |
|---------|---------|
| Size | 63 rows, 11 columns in the original data |
| Duplicate rows | 3 duplicate records |
| Missing values | 18 in total (age 2, screen time 4, battery health 3, study hours 3, attendance 3, CGPA 3) |
| Inconsistent gender text | 8 different spellings: `Male`, `male`, `MALE `, `M`, `Female`, `female`, `FEMALE `, `F` |
| Inconsistent overnight-charging text | 6 different spellings: `Yes`, `yes`, `Y`, `No`, `no`, `N` |
| Wrong data types | `daily_screen_time_hrs` and `battery_health_pct` were stored as text (for example `87%`) |
| Impossible values | Age as low as 2 and CGPA as high as 14.2 (CGPA scale is out of 10) |

### Cleaning steps applied
1. Removed duplicate rows
2. Standardized text: trimmed spaces, fixed case, mapped `M/F` to `Male/Female` and `Y/N` to `Yes/No`
3. Removed symbols such as `%` and converted columns to numbers
4. Handled impossible values (age and CGPA outside valid range)
5. Filled missing numeric values (median imputation)
6. Sorted by `student_id` and fixed correct data types

### Result
| | Original | Cleaned |
|---|---|---|
| Rows | 63 | **60** |
| Missing values | 18 | **0** |
| Gender categories | 8 | **2** |
| Overnight-charging categories | 6 | **2** |
| CGPA range | up to 14.2 | **6.32 to 9.85** |
| Age range | from 2 | **17 to 24** |

---

## 5. Exploratory Data Analysis (Charts)

The notebook contains these visualizations:

1. **Missing values chart** – shows where data was missing in the raw file
2. **Outlier / box plots** – shows impossible values before cleaning
3. **Histograms** – distribution of age, screen time, study hours, battery health, attendance, CGPA
4. **Category counts** – gender, year of study, overnight charging
5. **Correlation heatmap** – relation between all numeric columns
6. **Scatter plots with regression line** – screen time vs CGPA, study hours vs CGPA
7. **Battery & charging habits** – box and bar plots of CGPA by overnight charging and charge cycles
8. **Screen-time groups** – Low / Medium / High screen time compared with CGPA
9. **Pairplot** – all main variables together

---

## 6. Key Findings

### Correlation of each variable with CGPA (cleaned data)

| Variable | Correlation with CGPA | Meaning |
|----------|----------------------|---------|
| `daily_study_hrs` | **+0.76** | Strong positive: more study, higher CGPA |
| `daily_screen_time_hrs` | **-0.34** | Moderate negative: more screen time, lower CGPA |
| `attendance_pct` | +0.16 | Weak positive |
| `battery_health_pct` | +0.05 | Almost no relation |

### Main takeaways
- **Study time is the strongest factor** for CGPA by a big margin.
- **Screen time has a moderate negative link** with CGPA.
- **Battery health is almost unrelated** to CGPA. It shows up only as a minor factor, not a driver.
- Correlation is not causation. Students who study more may simply use their phones less.

---

## 7. Statistical Tests

| Test | Used for |
|------|----------|
| Welch's t-test | Compare mean CGPA of students who charge overnight vs those who do not |
| One-way ANOVA | Compare mean CGPA across Low / Medium / High screen-time groups |
| Multiple linear regression | Find the effect of each variable on CGPA while holding others constant |

Use a 5% significance level (p < 0.05) to decide if a result is statistically significant.

---

## 8. Machine Learning Models

Models compared (5-fold cross-validation plus a held-out test set):

| Model | CV R² (mean) | CV RMSE | Test R² | Test RMSE |
|-------|-------------|---------|---------|-----------|
| **Ridge Regression** | 0.467 | 0.455 | **0.631** | **0.406** |
| Linear Regression | 0.451 | 0.460 | 0.622 | 0.411 |
| Random Forest | 0.434 | 0.490 | 0.474 | 0.485 |
| Gradient Boosting | 0.358 | 0.490 | 0.441 | 0.500 |
| Dummy (mean baseline) | -0.228 | 0.748 | -0.019 | 0.674 |

**Best model: Ridge Regression** (highest test R² of 0.631 and lowest test RMSE of 0.406). It also clearly beats the baseline, so the features do carry real information.

**Why simple models win:** the dataset is small (60 rows) and the relationships are mostly linear, so complex models such as Random Forest and Gradient Boosting overfit and do not help.

### Feature importance

| Feature | RF importance | Permutation importance | Linear std. coefficient |
|---------|---------------|------------------------|-------------------------|
| `daily_study_hrs` | **0.726** | **1.254** | **+0.532** |
| `daily_screen_time_hrs` | 0.101 | 0.107 | -0.175 |
| `battery_health_pct` | 0.052 | 0.044 | +0.081 |
| `attendance_pct` | 0.046 | 0.040 | +0.119 |
| `age` | 0.031 | 0.021 | -0.090 |
| `daily_charge_cycles` | 0.017 | 0.012 | +0.002 |
| `charges_overnight` | 0.016 | 0.020 | -0.128 |
| `gender` | 0.007 | 0.003 | -0.011 |
| `year_of_study` | 0.004 | 0.002 | +0.065 |

All three methods agree: **study hours is number one, screen time is number two**.

---

## 9. Conclusion

1. Daily study hours is by far the best predictor of CGPA.
2. Higher screen time is linked with lower CGPA, but the effect is smaller than study time.
3. Battery health, charge cycles and overnight charging have little to no effect on CGPA.
4. Ridge Regression is the best model for this data.
5. Because the dataset is small and synthetic, these results should be treated as a demonstration of the analysis method.

### Limitations
- Only 60 students after cleaning
- Synthetic data, so no real-world conclusion can be drawn
- Correlation does not prove cause and effect
- Model results vary a lot between cross-validation folds (high CV standard deviation)

---

## 10. How to Run

### Option A: Notebook in Google Colab
1. Go to https://colab.research.google.com
2. Choose **File > Upload notebook** and select `EDA_Assignment_2_Team6.ipynb`
3. If the notebook reads the Excel file, upload `Team_6_-_original_data__cleaned_data.xlsx` in the Files panel (left side)
4. Choose **Runtime > Run all**

### Option B: Single script
1. Open a new Colab notebook
2. Paste all of `colab_battery_academic_analysis.py` into one cell
3. Press **Shift + Enter**. No upload is needed because the script generates its own data

### Libraries used
`pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`, `scikit-learn`, `openpyxl`

(All of these are pre-installed in Google Colab.)

---

## 11. Team Members

| Name | Register Number |
|------|-----------------|
| (add name) | (add number) |
| (add name) | (add number) |
| (add name) | (add number) |

**Course:** (add course name)
**Institution:** (add university name)
**Submitted on:** 28 October 2026
