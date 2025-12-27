# ⛽ Fuel Consumption Predictor

A professional Streamlit web application that predicts vehicle fuel consumption (City & Highway L/100km) using multiple machine learning models.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🚀 Features

- **5 ML Models** for ensemble predictions:
  - 🌲 Random Forest
  - 🚀 XGBoost
  - ⚡ LightGBM
  - 🌳 Decision Tree
  - 🐱 CatBoost

- **Dual Output Prediction**:
  - 🛣️ Highway fuel consumption (L/100km)
  - 🏙️ City fuel consumption (L/100km)

- **Professional UI** with:
  - Modern gradient design
  - Interactive sidebar inputs
  - Real-time predictions
  - Model comparison analysis
  - Ensemble averaging

## 📋 Input Features

| Feature | Description |
|---------|-------------|
| Combined (L/100 km) | Combined fuel consumption |
| Cylinders | Number of engine cylinders |
| Engine size (L) | Engine displacement in liters |
| Model year | Vehicle model year |
| Fuel type | Diesel, E85, Midgrade/Premium/Regular Gasoline, Natural Gas |

## 🛠️ Installation

1. **Clone or download the repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model files are present**:
   - `random_forest_model.pkl`
   - `xgb_model.pkl`
   - `lgbm_model.pkl`
   - `decision_tree_model.pkl`
   - `catboost_model.pkl`

## 🚀 Running the App

```bash
python -m streamlit run app.py
```

Or:
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 🧾 Generate the PDF Report

This repository includes an automated report generator that reproduces the core notebook workflow (EDA + evaluation) and outputs a professional PDF.

```bash
python generate_pdf_report.py --data path\\to\\vehicles.csv --out Fuel_Consumption_Report.pdf
```

Outputs:
- `Fuel_Consumption_Report.pdf` (or the path you pass via `--out`)
- `report_assets/` (figures embedded into the PDF)

## 📁 Project Structure

```
Models/
├── app.py                      # Main Streamlit application
├── generate_pdf_report.py       # PDF report generator
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── random_forest_model.pkl     # Random Forest model
├── xgb_model.pkl              # XGBoost model
├── lgbm_model.pkl             # LightGBM model
├── decision_tree_model.pkl    # Decision Tree model
└── catboost_model.pkl         # CatBoost model
```

## 📊 How It Works

1. **Input** your vehicle specifications in the sidebar
2. **Click** "🔮 Generate Predictions"
3. **View** predictions from all 5 models
4. **Compare** results in the detailed analysis section

## 🔧 Dependencies

- streamlit >= 1.28.0
- pandas >= 1.5.0
- numpy >= 1.24.0
- joblib >= 1.3.0
- scikit-learn == 1.6.1
- matplotlib >= 3.8.0
- seaborn >= 0.13.0
- reportlab >= 4.0.0
- lightgbm >= 4.0.0
- xgboost >= 2.0.0
- catboost >= 1.2.0

## 📈 Example Usage

**Compact Car (Toyota Corolla)**:
- Model Year: 2022
- Engine Size: 1.8 L
- Cylinders: 4
- Combined: 7.4 L/100km
- Fuel Type: Regular Gasoline

**SUV (Ford Explorer)**:
- Model Year: 2023
- Engine Size: 3.0 L
- Cylinders: 6
- Combined: 10.9 L/100km
- Fuel Type: Premium Gasoline

## 📝 Notes

- Models were trained with scikit-learn 1.6.1 - ensure compatibility
- Lower fuel consumption values indicate better efficiency
- Ensemble average provides a balanced prediction from all models

## 📄 License

MIT License - Feel free to use and modify for your projects.

---

**Built with ❤️ using Streamlit and Machine Learning**
