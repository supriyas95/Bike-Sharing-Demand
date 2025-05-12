# 🚲 Bike Sharing Demand Prediction

**Kaggle Competition Project**
*Forecasting hourly bike rental demand using ensemble machine learning models*

---

## 🧠 Problem Statement

Bike-sharing systems need accurate demand forecasts to efficiently allocate bikes across locations and time. This project addresses the challenge of **predicting hourly rental demand** using a dataset from the Kaggle competition “Bike Sharing Demand.”

By leveraging machine learning techniques, we aim to build models that learn from historical data to accurately predict the total number of bikes rented per hour. This improves fleet management, reduces stockouts, and enhances user experience.

---

## 📦 Dataset Overview

* **Source**: [Kaggle Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand/data)
* **Observations**: 10,886 hourly entries (2011–2012)
* **Target Variable**: `count` – total number of bike rentals per hour

### Key Features:

| Feature      | Description                              | Type        |
| ------------ | ---------------------------------------- | ----------- |
| `hour`       | Hour of the day (0–23)                   | Categorical |
| `season`     | Season (1: Spring, 2: Summer, ...)       | Categorical |
| `holiday`    | Whether the day is a holiday             | Boolean     |
| `workingday` | Whether the day is a working day         | Boolean     |
| `weather`    | Weather situation (1: Clear to 4: Storm) | Categorical |
| `temp`       | Temperature (°C)                         | Numerical   |
| `atemp`      | "Feels-like" temperature (°C)            | Numerical   |
| `humidity`   | Humidity (%)                             | Numerical   |
| `windspeed`  | Wind speed                               | Numerical   |

---

## 🧹 Data Preparation

* Removed columns not usable at prediction time (`casual`, `registered`, `datetime`)
* Converted categorical variables to factors (e.g., season, weather)
* Confirmed no missing values
* Feature engineered new variables (e.g., extracting hour from datetime)

---

## 📊 Exploratory Data Analysis

* **Bike rentals peak during rush hours (8 AM and 5 PM)**
* **Higher rentals in spring/summer and during working days**
* Rentals drop under poor weather conditions (rain/snow)


---

## 🔧 Modeling Approach

### Individual Models Trained:

1. **Linear Regression**

   * Simple, interpretable, but limited in capturing nonlinear relationships
   * **MAPE**: 294% | **RMSE**: 102

2. **Regression Tree**

   * Visual and interpretable
   * Pruned version improved generalization
   * **MAPE**: 73.5% | **RMSE**: 66.9

3. **Random Forest**

   * Ensemble of trees, captures non-linear patterns
   * **MAPE**: **58.0%** | **RMSE**: **55.3**
   * **Top single model**

4. **Gradient Boosting (GBM)**

   * Builds sequential trees, but underperformed in this case
   * **MAPE**: 217.8% | **RMSE**: 110.8

---

## 🧪 Ensemble Learning (Stacked Models)

To enhance prediction accuracy, we combined multiple models using **stacked generalization**:

* Base models: Linear Regression, Regression Tree (pruned), Random Forest
* Meta-models:

  * **Neural Network** (best results):

    * **MAPE**: **98.8%**, **RMSE**: **59.1**
  * Gradient Boosting (GBM): Higher error than Neural Net

📌 **Why Neural Net Performed Best**:

* Captured complex non-linear interactions (e.g., time + temperature + working day)
* Handled edge cases where individual models failed

---

## 📈 Model Performance Summary

| Model                | MAPE (%) | RMSE (Rentals) | Strengths                       | Weaknesses                                |
| -------------------- | -------- | -------------- | ------------------------------- | ----------------------------------------- |
| Linear Regression    | 294.6    | 102.2          | Simple, interpretable           | Misses complex patterns                   |
| Regression Tree      | 73.5     | 66.9           | Visual, easy to explain         | Prone to overfitting                      |
| Random Forest        | **58.0** | **55.3**       | Accurate, handles interactions  | Less interpretable, computationally heavy |
| GBM (Stacked)        | 217.8    | 110.8          | Learns from mistakes            | Underperformed                            |
| Neural Net (Stacked) | 98.8     | 59.1           | Best ensemble performer, robust | Hard to interpret                         |

---

## 🔍 Key Insights

* **Hour of day** was the strongest predictor of bike demand
* **Random Forest** outperformed all individual models
* **Stacking models** provided an ensemble boost, though not always better
* **Weather, temperature, and working day** status also heavily influenced demand

---

## 🛠️ Future Work

* Integrate **real-time data** (e.g., weather, traffic) for dynamic predictions
* Test generalizability in **different cities/seasons**
* Deploy model in a **dashboard** for operations teams
* Explore **deep learning** models on larger datasets

---

## ✅ Conclusion

This project demonstrated the power of machine learning for real-world forecasting. Random Forest provided the most accurate single model, while Neural Net stacking offered a strong ensemble boost. With these models, bike-sharing systems can significantly improve resource planning and customer satisfaction.

---
