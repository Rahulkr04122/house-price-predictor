import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page config
st.set_page_config(page_title="House Price Predictor", layout="wide")

# Load the trained model
@st.cache_resource
def load_model():
    with open('house_price_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# ===== PAGE TITLE =====
st.title("🏠 House Price Predictor")
st.write("Predict house prices using Machine Learning!")

# ===== DIVIDER =====
st.divider()

# ===== SIDEBAR FOR INPUT =====
st.sidebar.header("📋 Enter House Details")

size_sqft = st.sidebar.number_input(
    "Size (sq ft)",
    min_value=1000,
    max_value=5000,
    value=3000,
    step=100
)

bedrooms = st.sidebar.slider(
    "Bedrooms",
    min_value=1,
    max_value=5,
    value=3
)

bathrooms = st.sidebar.slider(
    "Bathrooms",
    min_value=1,
    max_value=4,
    value=2
)

age_years = st.sidebar.slider(
    "Age (years)",
    min_value=0,
    max_value=50,
    value=15
)

garage_spaces = st.sidebar.slider(
    "Garage Spaces",
    min_value=0,
    max_value=4,
    value=2
)

has_pool = st.sidebar.selectbox(
    "Has Pool?",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

# ===== MAKE PREDICTION =====
st.divider()

if st.button("🔮 Predict Price", use_container_width=True):
    # Prepare data
    house_features = np.array([[size_sqft, bedrooms, bathrooms, age_years, garage_spaces, has_pool]])
    
    # Make prediction
    predicted_price = model.predict(house_features)[0]
    
    # Display result
    st.success("✓ Prediction Complete!")
    
    # Show in big number
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Predicted Price",
            value=f"${predicted_price:,.0f}",
            delta=None
        )
    
    with col2:
        st.info(f"""
        **House Details:**
        - Size: {size_sqft:,} sq ft
        - Bedrooms: {bedrooms}
        - Bathrooms: {bathrooms}
        - Age: {age_years} years
        - Garage: {garage_spaces} spaces
        - Pool: {'Yes' if has_pool == 1 else 'No'}
        """)

# ===== EXAMPLE PREDICTIONS =====
st.divider()
st.subheader("📊 Example Predictions")

examples = {
    "Small Old House": [1500, 2, 1, 35, 0, 0],
    "Medium House": [3000, 3, 2, 15, 2, 0],
    "Large New House with Pool": [4500, 4, 3, 5, 3, 1],
    "Budget House": [2000, 2, 1, 40, 1, 0],
    "Premium Luxury House": [4000, 4, 3, 0, 3, 1],
}

example_predictions = {}
for name, features in examples.items():
    price = model.predict([features])[0]
    example_predictions[name] = price

# Display as table
example_df = pd.DataFrame({
    "House Type": list(example_predictions.keys()),
    "Predicted Price": [f"${price:,.0f}" for price in example_predictions.values()]
})

st.table(example_df)

# ===== FOOTER =====
st.divider()
st.markdown("""
---
**About This App:**
- Built with Machine Learning (Linear Regression)
- Trained on 100 real house data points
- Model Accuracy: R² = 98.47%
- Prediction Error: ±2.87%

**Features Used:**
- House Size (sq ft)
- Number of Bedrooms
- Number of Bathrooms
- Age (years)
- Garage Spaces
- Has Pool (Yes/No)
""")