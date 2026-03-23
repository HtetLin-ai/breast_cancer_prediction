import joblib
import streamlit as st
import numpy as np

# Load model and scaler
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

# st.title("Breast Cancer Prediction")

# Input sliders (with ranges)
# radius_mean = st.slider("radius_mean", 0.0, 30.0, 10.0)
# texture_mean = st.slider("texture_mean", 0.0, 40.0, 15.0)
# perimeter_mean = st.slider("perimeter_mean", 0.0, 200.0, 50.0)
# area_mean = st.slider("area_mean", 0.0, 2500.0, 500.0)
# smoothness_mean = st.slider("smoothness_mean", 0.0, 1.0, 0.1)
# compactness_mean = st.slider("compactness_mean", 0.0, 1.0, 0.1)
# concavity_mean = st.slider("concavity_mean", 0.0, 1.0, 0.1)
# concave_points_mean = st.slider("concave_points_mean", 0.0, 1.0, 0.1)
# symmetry_mean = st.slider("symmetry_mean", 0.0, 1.0, 0.2)
# fractal_dimension_mean = st.slider("fractal_dimension_mean", 0.0, 1.0, 0.05)

# radius_se = st.slider("radius_se", 0.0, 10.0, 1.0)
# texture_se = st.slider("texture_se", 0.0, 10.0, 1.0)
# perimeter_se = st.slider("perimeter_se", 0.0, 100.0, 5.0)
# area_se = st.slider("area_se", 0.0, 1000.0, 50.0)
# smoothness_se = st.slider("smoothness_se", 0.0, 1.0, 0.01)
# compactness_se = st.slider("compactness_se", 0.0, 1.0, 0.01)
# concavity_se = st.slider("concavity_se", 0.0, 1.0, 0.01)
# concave_points_se = st.slider("concave_points_se", 0.0, 1.0, 0.01)
# symmetry_se = st.slider("symmetry_se", 0.0, 1.0, 0.01)
# fractal_dimension_se = st.slider("fractal_dimension_se", 0.0, 1.0, 0.01)

# radius_worst = st.slider("radius_worst", 0.0, 50.0, 15.0)
# texture_worst = st.slider("texture_worst", 0.0, 50.0, 20.0)
# perimeter_worst = st.slider("perimeter_worst", 0.0, 300.0, 100.0)
# area_worst = st.slider("area_worst", 0.0, 5000.0, 800.0)
# smoothness_worst = st.slider("smoothness_worst", 0.0, 1.0, 0.2)
# compactness_worst = st.slider("compactness_worst", 0.0, 1.0, 0.2)
# concavity_worst = st.slider("concavity_worst", 0.0, 1.0, 0.2)
# concave_points_worst = st.slider("concave_points_worst", 0.0, 1.0, 0.2)
# symmetry_worst = st.slider("symmetry_worst", 0.0, 1.0, 0.3)
# fractal_dimension_worst = st.slider("fractal_dimension_worst", 0.0, 1.0, 0.1)

# Prediction
# if st.button("Predict"):
#     input_data = np.array([[
#         radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
#         compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
#         radius_se, texture_se, perimeter_se, area_se, smoothness_se,
#         compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
#         radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
#         compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
#     ]])
#     print("input_data:", input_data)
#     input_scaled = scaler.transform(input_data)
#     print("input_scaled:",input_scaled)
#     prediction = model.predict(input_scaled)
#     print("Prediction",prediction)
#     st.write("Prediction:", prediction)

#     if prediction[0] == 1:
#         st.write("🔴 Malignant (Cancer)")
#     else:
#         st.write("🟢 Benign (No Cancer)")

import pandas as pd
import numpy as np

# if st.button("Predict"):
#     # 1. Collect all your slider variables in the EXACT order they appear in the UI
#     slider_values = [
#         radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
#         compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
#         radius_se, texture_se, perimeter_se, area_se, smoothness_se,
#         compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
#         radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
#         compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
#     ]

#     # 2. Automatically grab the "Correct" names from the scaler itself
#     expected_names = scaler.feature_names_in_

#     # 3. Create the DataFrame using the scaler's own preferred names
#     input_df = pd.DataFrame([slider_values], columns=expected_names)

#     # 4. Scale and Predict
#     input_scaled = scaler.transform(input_df)
#     prediction = model.predict(input_scaled)

#     # UI Output
#     if prediction[0] == 1:
#         st.error("🔴 Result: Malignant (Cancer)")
#     else:
#         st.success("🟢 Result: Benign (No Cancer)")


st.set_page_config(layout="wide") # Use the full width of the screen
st.title("Breast Cancer Diagnostic Dashboard")
st.markdown("---")

# Create three columns
col1, col2, col3 = st.columns(3)

with col1:
    st.header("Mean Values")
    radius_mean = st.slider("Radius", 0.0, 30.0, 14.0)
    texture_mean = st.slider("Texture", 0.0, 40.0, 19.0)
    perimeter_mean = st.slider("Perimeter", 0.0, 200.0, 92.0)
    area_mean = st.slider("Area", 0.0, 2500.0, 650.0)
    smoothness_mean = st.slider("Smoothness", 0.0, 0.2, 0.1)
    compactness_mean = st.slider("Compactness", 0.0, 0.3, 0.1)
    concavity_mean = st.slider("Concavity", 0.0, 0.5, 0.1)
    concave_points_mean = st.slider("Concave Points", 0.0, 0.2, 0.05)
    symmetry_mean = st.slider("Symmetry", 0.0, 0.3, 0.18)
    fractal_dimension_mean = st.slider("Fractal Dim.", 0.0, 0.1, 0.06)

with col2:
    st.header("Standard Error")
    radius_se = st.slider("Radius SE", 0.0, 3.0, 0.4)
    texture_se = st.slider("Texture SE", 0.0, 5.0, 1.2)
    perimeter_se = st.slider("Perimeter SE", 0.0, 25.0, 3.0)
    area_se = st.slider("Area SE", 0.0, 500.0, 40.0)
    smoothness_se = st.slider("Smoothness SE", 0.0, 0.03, 0.01)
    compactness_se = st.slider("Compactness SE", 0.0, 0.15, 0.02)
    concavity_se = st.slider("Concavity SE", 0.0, 0.4, 0.03)
    concave_points_se = st.slider("Concave Points SE", 0.0, 0.05, 0.01)
    symmetry_se = st.slider("Symmetry SE", 0.0, 0.1, 0.02)
    fractal_dimension_se = st.slider("Fractal Dim. SE", 0.0, 0.03, 0.004)

with col3:
    st.header("Worst Values")
    radius_worst = st.slider("Radius Worst", 0.0, 40.0, 16.0)
    texture_worst = st.slider("Texture Worst", 0.0, 50.0, 25.0)
    perimeter_worst = st.slider("Perimeter Worst", 0.0, 250.0, 107.0)
    area_worst = st.slider("Area Worst", 0.0, 4000.0, 880.0)
    smoothness_worst = st.slider("Smoothness Worst", 0.0, 0.3, 0.13)
    compactness_worst = st.slider("Compactness Worst", 0.0, 1.0, 0.25)
    concavity_worst = st.slider("Concavity Worst", 0.0, 1.3, 0.3)
    concave_points_worst = st.slider("Concave Points Worst", 0.0, 0.3, 0.1)
    symmetry_worst = st.slider("Symmetry Worst", 0.0, 0.7, 0.3)
    fractal_dimension_worst = st.slider("Fractal Dim. Worst", 0.0, 0.2, 0.08)

st.markdown("---")

# Prediction logic
if st.button("Analyze Data", use_container_width=True):
    # This list MUST be in the same order as your training data columns
    slider_values = [
        radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
        compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
        radius_se, texture_se, perimeter_se, area_se, smoothness_se,
        compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
        radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
        compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
    ]

    # Map values to the scaler's expected feature names
    input_df = pd.DataFrame([slider_values], columns=scaler.feature_names_in_)
    
    # Scale and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled) # Shows confidence level
    print(probability)
    # Display results nicely
    if prediction[0] == 1:
        st.error(f"### Result: Malignant")
        st.write(f"Confidence: {probability[0][1]*100:.2f}%")
    else:
        st.success(f"### Result: Benign")
        st.write(f"Confidence: {probability[0][0]*100:.2f}%")