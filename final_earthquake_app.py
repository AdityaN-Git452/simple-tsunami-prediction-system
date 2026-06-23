# app.py
import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Earthquake & Tsunami Risk Prediction",
    page_icon="🌋",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

model = load_model()

# Title and description
st.title("🌋 Earthquake & Tsunami Risk Prediction System")
st.markdown("### Enter seismic parameters to predict tsunami risk")
st.markdown("---")

# Create three columns for organized input
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📊 Seismic Measurements")
    magnitude = st.number_input(
        "Magnitude", 
        min_value=0.0, 
        max_value=10.0, 
        value=6.5, 
        step=0.1,
        help="Earthquake magnitude (Richter scale)"
    )
    
    cdi = st.number_input(
        "CDI (Community Decimal Intensity)", 
        min_value=0, 
        max_value=12, 
        value=5, 
        step=1,
        help="Maximum reported intensity"
    )
    
    mmi = st.number_input(
        "MMI (Modified Mercalli Intensity)", 
        min_value=0, 
        max_value=12, 
        value=5, 
        step=1,
        help="Instrumental intensity"
    )
    
    sig = st.number_input(
        "Significance", 
        min_value=0, 
        max_value=1000, 
        value=600, 
        step=10,
        help="Event significance (0-1000)"
    )

with col2:
    st.subheader("📍 Location & Depth")
    
    nst = st.number_input(
        "NST (Number of Stations)", 
        min_value=0, 
        max_value=500, 
        value=100, 
        step=1,
        help="Number of seismic stations used"
    )
    
    dmin = st.number_input(
        "DMIN (Distance)", 
        min_value=0.0, 
        max_value=20.0, 
        value=2.0, 
        step=0.1,
        help="Horizontal distance from epicenter to nearest station (degrees)"
    )
    
    gap = st.number_input(
        "Azimuthal Gap", 
        min_value=0.0, 
        max_value=360.0, 
        value=30.0, 
        step=1.0,
        help="Largest azimuthal gap between stations (degrees)"
    )
    
    depth = st.number_input(
        "Depth (km)", 
        min_value=0.0, 
        max_value=700.0, 
        value=25.0, 
        step=1.0,
        help="Depth of the earthquake"
    )

with col3:
    st.subheader("🌍 Geographic Data")
    
    latitude = st.number_input(
        "Latitude", 
        min_value=-90.0, 
        max_value=90.0, 
        value=0.0, 
        step=0.1,
        help="Geographic latitude"
    )
    
    longitude = st.number_input(
        "Longitude", 
        min_value=-180.0, 
        max_value=180.0, 
        value=0.0, 
        step=0.1,
        help="Geographic longitude"
    )
    
    year = st.number_input(
        "Year", 
        min_value=2000, 
        max_value=2025, 
        value=2024, 
        step=1,
        help="Year of the event"
    )
    
    month = st.number_input(
        "Month", 
        min_value=1, 
        max_value=12, 
        value=1, 
        step=1,
        help="Month of the event (1-12)"
    )

st.markdown("---")

# Collect features in EXACT order as training data
features = np.array([[
    magnitude,   # 1
    cdi,        # 2
    mmi,        # 3
    sig,        # 4
    nst,        # 5
    dmin,       # 6
    gap,        # 7
    depth,      # 8
    latitude,   # 9
    longitude,  # 10
    year,       # 11
    month       # 12
]])

# Debug info (expandable)
with st.expander("🔍 Debug Information"):
    st.write(f"**Model expects:** {model.n_features_in_} features")
    st.write(f"**Provided:** {features.shape[1]} features")
    st.write("**Feature order:**")
    feature_names = [
        "magnitude", "cdi", "mmi", "sig", "nst", "dmin", 
        "gap", "depth", "latitude", "longitude", "Year", "Month"
    ]
    for i, (name, val) in enumerate(zip(feature_names, features[0]), 1):
        st.write(f"{i}. {name}: {val}")

# Prediction button
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

with col_btn2:
    predict_button = st.button("🔮 Predict Tsunami Risk", type="primary", use_container_width=True)

if predict_button:
    try:
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        st.markdown("---")
        st.subheader("📋 Prediction Results")
        
        # Display results in columns
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            if prediction == 1:
                st.error("### ⚠️ HIGH TSUNAMI RISK!")
                st.write("**Status:** Tsunami likely to occur")
                st.write(f"**Confidence:** {probability[1]*100:.1f}%")
                st.warning("⚠️ **Recommendation:** Immediate evacuation of coastal areas advised!")
            else:
                st.success("### ✅ LOW TSUNAMI RISK")
                st.write("**Status:** Tsunami unlikely")
                st.write(f"**Confidence:** {probability[0]*100:.1f}%")
                st.info("ℹ️ **Recommendation:** Continue normal monitoring procedures.")
        
        with result_col2:
            # Probability bar chart
            st.write("**Probability Distribution:**")
            st.progress(probability[1], text=f"Tsunami Risk: {probability[1]*100:.1f}%")
            st.progress(probability[0], text=f"No Tsunami: {probability[0]*100:.1f}%")
            
    except Exception as e:
        st.error(f"❌ **Error during prediction:** {str(e)}")
        st.info("Please check that all input values are valid and try again.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🌊 Earthquake & Tsunami Prediction System | Built with Streamlit & Machine Learning</p>
        <p><small>Note: Predictions are based on historical data and should be used alongside official monitoring systems.</small></p>
    </div>
    """, 
    unsafe_allow_html=True
)
