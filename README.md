# 🌱 Smart Crop Yield Predictor (SCYP)

Smart Crop Yield Predictor is an AI-based Machine Learning project that predicts crop yield using historical agricultural data and environmental factors.

The system analyzes different agricultural parameters such as crop type, country, year, rainfall, temperature, pesticide usage, and production data to estimate crop yield.

---

## 🚀 Features

* 🌾 Crop yield prediction using Machine Learning
* 🌍 Country and crop based prediction
* 🌧️ Rainfall and temperature analysis
* 🧪 Pesticide usage impact analysis
* 📊 Data preprocessing and feature engineering
* 🤖 Machine Learning based prediction model
* 🌐 Interactive web interface
* ⚡ FastAPI backend integration

---

## 🧠 Machine Learning Approach

The project uses Machine Learning algorithms to learn patterns from historical agricultural data and predict crop yield.

### Algorithms Used

* Random Forest Regressor
* Linear Regression (Baseline Model)

### Input Parameters

| Parameter       | Description                       |
| --------------- | --------------------------------- |
| Country         | Region/Area of cultivation        |
| Crop            | Selected crop type                |
| Year            | Prediction year                   |
| Pesticides      | Amount of pesticide usage         |
| Rainfall        | Average rainfall                  |
| Temperature     | Average temperature               |
| Production Data | Historical production information |

### Output

The model predicts:

**Crop Yield (hg/ha)**

---

## 📂 Project Structure

```text
Smart-Crop-Yield-Predictor/

│
├── Background.jpg
├── Crop_Yield_Prediction.html
├── Crop_dropdown.html
├── Logo.png
├── Production_Crops_Livestock_E_All_Data.py
├── Smart_Crop_Yield_Predictor.csv
├── app.py
├── basemodelcls.py
├── crop_yield_predictor_model.pkl
├── encoders.pkl
├── index.html
├── pesticides.py
├── rainfall.py
├── style.css
├── temp.py
├── tempCodeRunnerFile.py
├── test.py
├── yield.py
└── yield_df.py
```

---

## ⚙️ Technologies Used

### Programming Language

* Python

### Machine Learning & Data Processing

* Scikit-learn
* Pandas
* NumPy

### Backend

* FastAPI
* Uvicorn

### Frontend

* HTML
* CSS
* JavaScript

### Model Serialization

* Joblib

---

## 🔄 Project Workflow

```text
Agricultural Datasets
          |
          ↓
Data Cleaning & Preprocessing
          |
          ↓
Feature Engineering
          |
          ↓
Machine Learning Model Training
          |
          ↓
Model Serialization
          |
          ↓
FastAPI Prediction API
          |
          ↓
Web Interface
          |
          ↓
Crop Yield Prediction
```

---

## 📊 Data Processing

The dataset is processed through multiple steps:

* Removing unnecessary columns
* Handling missing values
* Data transformation
* Feature encoding
* Merging agricultural and environmental datasets
* Preparing data for Machine Learning models

---

## 💾 Model Files

The trained model and encoders are stored as:

* `crop_yield_predictor_model.pkl`
* `encoders.pkl`

These files are loaded by the backend to generate predictions.

---

## ▶️ How To Run

### 1. Clone Repository

```bash
git clone https://github.com/Kartavya-Jain/Smart-Crop-Yield-Predictor.git
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start FastAPI Server

```bash
uvicorn app:app --reload
```

### 4. Run Frontend

Open:

```text
index.html
```

in your browser.

---

## 🔮 Future Improvements

* Real-time weather API integration
* Satellite based crop monitoring
* Advanced Deep Learning models
* Explainable AI integration
* Improved regional farming recommendations
* Real-time agricultural insights

---

## 👨‍💻 Author

**Kartavya Jain**

Computer Science Engineering (AI & ML)

GitHub: Kartavya-Jain

---

## ⭐ Project Objective

The objective of Smart Crop Yield Predictor is to combine Machine Learning and agricultural data analysis to build an intelligent system that can estimate crop yield and support data-driven farming decisions.
