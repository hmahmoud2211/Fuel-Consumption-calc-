import streamlit as st
import pickle
import numpy as np
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Fuel Consumption Predictor",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        color: #e0e0e0;
        font-size: 1.1rem;
    }
    
    /* Card styling */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        margin: 0.5rem 0;
    }
    
    .prediction-card-lr {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    .prediction-card-rf {
        background: linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%);
    }
    
    .prediction-card-svr {
        background: linear-gradient(135deg, #8E2DE2 0%, #4A00E0 100%);
    }
    
    .prediction-card-lgbm {
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
    }
    
    .model-name {
        color: white;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        opacity: 0.9;
    }
    
    .prediction-value {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
    }
    
    .prediction-unit {
        color: white;
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Input section styling */
    .input-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #1e3c72;
    }
    
    .section-title {
        color: #1e3c72;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.9rem;
    }
    
    /* Metric card */
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin: 0.5rem 0;
    }
    
    /* Average prediction card */
    .avg-card {
        background: linear-gradient(135deg, #232526 0%, #414345 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    
    .avg-title {
        color: #ffd700;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .avg-value {
        color: white;
        font-size: 3rem;
        font-weight: 700;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(30, 60, 114, 0.4);
    }
    
    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #1e3c72, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'Random Forest': 'random_forest_model.pkl',
        'XGBoost': 'xgb_model.pkl',
        'LightGBM': 'lgbm_model.pkl',
        'Decision Tree': 'decision_tree_model.pkl',
        'CatBoost': 'catboost_model.pkl'
    }
    
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    for name, file in model_files.items():
        file_path = os.path.join(script_dir, file)
        try:
            # Try joblib first (most common for sklearn models)
            models[name] = joblib.load(file_path)
        except Exception as e1:
            try:
                # Try pickle as fallback
                with open(file_path, 'rb') as f:
                    models[name] = pickle.load(f)
            except Exception as e2:
                st.error(f"Error loading {name}: joblib error: {e1}, pickle error: {e2}")
    
    return models

models = load_models()

# Header
st.markdown("""
<div class="header-container">
    <div class="header-title">⛽ Fuel Consumption Predictor</div>
    <div class="header-subtitle">Advanced ML-Powered Vehicle Efficiency Analysis</div>
</div>
""", unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.markdown("## 🚗 Vehicle Specifications")
    st.markdown("---")
    
    # Vehicle Information Section
    st.markdown("### 📋 Basic Information")
    
    model_year = st.slider(
        "Model Year",
        min_value=1990,
        max_value=2025,
        value=2020,
        help="Select the vehicle's model year"
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Engine Specifications")
    
    engine_size = st.number_input(
        "Engine Size (L)",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.1,
        help="Engine displacement in liters"
    )
    
    cylinders = st.selectbox(
        "Number of Cylinders",
        options=[2, 3, 4, 5, 6, 8, 10, 12],
        index=2,
        help="Number of engine cylinders"
    )
    
    st.markdown("---")
    st.markdown("### ⛽ Fuel Consumption")
    
    combined_consumption = st.number_input(
        "Combined (L/100 km)",
        min_value=1.0,
        max_value=28.0,
        value=8.5,
        step=0.1,
        help="Combined fuel consumption"
    )
    
    st.markdown("---")
    st.markdown("### 🔥 Fuel Type")
    
    fuel_type = st.selectbox(
        "Select Fuel Type",
        options=[
            "Regular Gasoline",
            "Premium Gasoline",
            "Midgrade Gasoline",
            "Diesel",
            "E85",
            "Natural Gas"
        ],
        index=0,
        help="Select the vehicle's fuel type"
    )

# Create feature encoding for fuel type
def encode_fuel_type(fuel_type):
    fuel_encoding = {
        'Diesel': [1, 0, 0, 0, 0, 0],
        'E85': [0, 1, 0, 0, 0, 0],
        'Midgrade Gasoline': [0, 0, 1, 0, 0, 0],
        'Natural Gas': [0, 0, 0, 1, 0, 0],
        'Premium Gasoline': [0, 0, 0, 0, 1, 0],
        'Regular Gasoline': [0, 0, 0, 0, 0, 1]
    }
    return fuel_encoding.get(fuel_type, [0, 0, 0, 0, 0, 1])

# Prepare input features
def prepare_features():
    fuel_encoding = encode_fuel_type(fuel_type)
    
    # Feature order based on provided schema:
    # 0: Combined (L/100 km)
    # 1: Cylinders
    # 2: Engine size (L)
    # 3: Model year
    # 4: Fuel type_diesel
    # 5: Fuel type_e85
    # 6: Fuel type_midgrade gasoline
    # 7: Fuel type_natural gas
    # 8: Fuel type_premium gasoline
    # 9: Fuel type_regular gasoline
    
    features = [
        combined_consumption,    # 0: Combined (L/100 km)
        cylinders,               # 1: Cylinders
        engine_size,             # 2: Engine size (L)
        model_year,              # 3: Model year
        fuel_encoding[0],        # 4: Fuel type_diesel
        fuel_encoding[1],        # 5: Fuel type_e85
        fuel_encoding[2],        # 6: Fuel type_midgrade gasoline
        fuel_encoding[3],        # 7: Fuel type_natural gas
        fuel_encoding[4],        # 8: Fuel type_premium gasoline
        fuel_encoding[5],        # 9: Fuel type_regular gasoline
    ]
    
    return np.array(features).reshape(1, -1)

# Main content area
st.markdown("### 🤖 Models Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("""
    <div class="metric-container">
        <h4 style="color: #fc4a1a; margin-bottom: 0.5rem;">🌲 Random Forest</h4>
        <p style="margin: 0; font-size: 0.8rem; color: #666;">Ensemble of decision trees reducing overfitting.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-container">
        <h4 style="color: #8E2DE2; margin-bottom: 0.5rem;">🚀 XGBoost</h4>
        <p style="margin: 0; font-size: 0.8rem; color: #666;">Gradient boosting with regularization.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-container">
        <h4 style="color: #0072ff; margin-bottom: 0.5rem;">⚡ LightGBM</h4>
        <p style="margin: 0; font-size: 0.8rem; color: #666;">Fast leaf-wise tree growth framework.</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-container">
        <h4 style="color: #11998e; margin-bottom: 0.5rem;">🌳 Decision Tree</h4>
        <p style="margin: 0; font-size: 0.8rem; color: #666;">Simple tree-based splitting rules.</p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="metric-container">
        <h4 style="color: #e91e63; margin-bottom: 0.5rem;">🐱 CatBoost</h4>
        <p style="margin: 0; font-size: 0.8rem; color: #666;">Gradient boosting for categorical features.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Prediction button
predict_button = st.button("🔮 Generate Predictions", use_container_width=True)

if predict_button:
    features = prepare_features()
    
    st.markdown("### 📈 Model Predictions")
    st.markdown("*Fuel Consumption Predictions (City & Highway) from Multiple ML Models*")
    
    predictions_city = {}
    predictions_highway = {}
    
    # Make predictions with each model - Row 1
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'Random Forest' in models:
            try:
                pred = models['Random Forest'].predict(features)[0]
                city_pred = pred[0] if hasattr(pred, '__len__') else pred
                highway_pred = pred[1] if hasattr(pred, '__len__') and len(pred) > 1 else pred
                predictions_city['Random Forest'] = city_pred
                predictions_highway['Random Forest'] = highway_pred
                st.markdown(f"""
                <div class="prediction-card prediction-card-rf">
                    <div class="model-name">🌲 Random Forest</div>
                    <div class="prediction-value">{city_pred:.2f}</div>
                    <div class="prediction-unit">🛣️ Highway L/100km</div>
                    <div class="prediction-value" style="margin-top: 0.5rem;">{highway_pred:.2f}</div>
                    <div class="prediction-unit">🏙️ City L/100km</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Random Forest Error: {e}")
    
    with col2:
        if 'XGBoost' in models:
            try:
                pred = models['XGBoost'].predict(features)[0]
                city_pred = pred[0] if hasattr(pred, '__len__') else pred
                highway_pred = pred[1] if hasattr(pred, '__len__') and len(pred) > 1 else pred
                predictions_city['XGBoost'] = city_pred
                predictions_highway['XGBoost'] = highway_pred
                st.markdown(f"""
                <div class="prediction-card prediction-card-svr">
                    <div class="model-name">🚀 XGBoost</div>
                    <div class="prediction-value">{city_pred:.2f}</div>
                    <div class="prediction-unit">🛣️ Highway L/100km</div>
                    <div class="prediction-value" style="margin-top: 0.5rem;">{highway_pred:.2f}</div>
                    <div class="prediction-unit">🏙️ City L/100km</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"XGBoost Error: {e}")
    
    with col3:
        if 'LightGBM' in models:
            try:
                pred = models['LightGBM'].predict(features)[0]
                city_pred = pred[0] if hasattr(pred, '__len__') else pred
                highway_pred = pred[1] if hasattr(pred, '__len__') and len(pred) > 1 else pred
                predictions_city['LightGBM'] = city_pred
                predictions_highway['LightGBM'] = highway_pred
                st.markdown(f"""
                <div class="prediction-card prediction-card-lgbm">
                    <div class="model-name">⚡ LightGBM</div>
                    <div class="prediction-value">{city_pred:.2f}</div>
                    <div class="prediction-unit">🛣️ Highway L/100km</div>
                    <div class="prediction-value" style="margin-top: 0.5rem;">{highway_pred:.2f}</div>
                    <div class="prediction-unit">🏙️ City L/100km</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"LightGBM Error: {e}")
    
    # Row 2 for Decision Tree and ANN
    col4, col5, col6 = st.columns(3)
    
    with col4:
        if 'Decision Tree' in models:
            try:
                pred = models['Decision Tree'].predict(features)[0]
                city_pred = pred[0] if hasattr(pred, '__len__') else pred
                highway_pred = pred[1] if hasattr(pred, '__len__') and len(pred) > 1 else pred
                predictions_city['Decision Tree'] = city_pred
                predictions_highway['Decision Tree'] = highway_pred
                st.markdown(f"""
                <div class="prediction-card prediction-card-lr">
                    <div class="model-name">🌳 Decision Tree</div>
                    <div class="prediction-value">{city_pred:.2f}</div>
                    <div class="prediction-unit">🛣️ Highway L/100km</div>
                    <div class="prediction-value" style="margin-top: 0.5rem;">{highway_pred:.2f}</div>
                    <div class="prediction-unit">🏙️ City L/100km</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Decision Tree Error: {e}")
    
    with col5:
        if 'CatBoost' in models:
            try:
                pred = models['CatBoost'].predict(features)[0]
                city_pred = pred[0] if hasattr(pred, '__len__') else pred
                highway_pred = pred[1] if hasattr(pred, '__len__') and len(pred) > 1 else pred
                predictions_city['CatBoost'] = city_pred
                predictions_highway['CatBoost'] = highway_pred
                st.markdown(f"""
                <div class="prediction-card" style="background: linear-gradient(135deg, #e91e63 0%, #f06292 100%);">
                    <div class="model-name">🐱 CatBoost</div>
                    <div class="prediction-value">{city_pred:.2f}</div>
                    <div class="prediction-unit">🛣️ Highway L/100km</div>
                    <div class="prediction-value" style="margin-top: 0.5rem;">{highway_pred:.2f}</div>
                    <div class="prediction-unit">🏙️ City L/100km</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"CatBoost Error: {e}")
    
    # Average prediction
    if predictions_city and predictions_highway:
        avg_city = np.mean(list(predictions_city.values()))
        avg_highway = np.mean(list(predictions_highway.values()))
        
        st.markdown(f"""
        <div class="avg-card">
            <div class="avg-title">🏆 ENSEMBLE AVERAGE PREDICTION</div>
            <div style="display: flex; justify-content: center; gap: 3rem; margin-top: 1rem;">
                <div>
                    <div class="avg-value">{avg_city:.2f}</div>
                    <div style="color: #ffd700; font-size: 1rem;">🛣️ Highway L/100km</div>
                </div>
                <div>
                    <div class="avg-value">{avg_highway:.2f}</div>
                    <div style="color: #ffd700; font-size: 1rem;">🏙️ City L/100km</div>
                </div>
            </div>
            <div style="color: #aaa; font-size: 0.9rem; margin-top: 1rem;">
                Based on {len(predictions_city)} ML models
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional insights
        st.markdown("---")
        st.markdown("### 📋 Detailed Analysis")
        
        col_insight1, col_insight2 = st.columns(2)
        
        with col_insight1:
            # Create a comparison dataframe
            df_predictions = pd.DataFrame({
                'Model': list(predictions_city.keys()),
                'Highway (L/100km)': [round(v, 2) for v in predictions_city.values()],
                'City (L/100km)': [round(v, 2) for v in predictions_highway.values()]
            })
            
            st.dataframe(df_predictions, use_container_width=True, hide_index=True)
        
        with col_insight2:
            min_city = min(predictions_city.values())
            max_city = max(predictions_city.values())
            min_highway = min(predictions_highway.values())
            max_highway = max(predictions_highway.values())
            
            st.markdown(f"""
            **📊 Highway Prediction Statistics:**
            - **Min:** {min_city:.2f} | **Max:** {max_city:.2f} | **Avg:** {avg_city:.2f} L/100km
            
            **📊 City Prediction Statistics:**
            - **Min:** {min_highway:.2f} | **Max:** {max_highway:.2f} | **Avg:** {avg_highway:.2f} L/100km
            
            **🔍 Model Agreement:** {'High' if (max_city - min_city) < 1 and (max_highway - min_highway) < 1 else 'Medium' if (max_city - min_city) < 3 else 'Low'}
            """)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🚀 <strong>Fuel Consumption Predictor</strong> | Powered by Machine Learning</p>
    <p style="font-size: 0.8rem; color: #888;">
        Using Random Forest, XGBoost, LightGBM, Decision Tree, and CatBoost models
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar footer
with st.sidebar:
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        <p>💡 <strong>Tips:</strong></p>
        <p>• Adjust all parameters for accurate predictions</p>
        <p>• Compare results across different models</p>
        <p>• Lower values = better fuel efficiency</p>
    </div>
    """, unsafe_allow_html=True)
