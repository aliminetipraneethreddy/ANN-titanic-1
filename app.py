import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# -----------------------------------------
# LOAD DATASET
# -----------------------------------------
df = pd.read_csv("titanic_ann_model.csv")

# Select Features
X = df[['Pclass', 'Age', 'Fare']]

# Fill missing Age values
X['Age'] = X['Age'].fillna(X['Age'].mean())

# Target
y = df['Survived']

# -----------------------------------------
# NORMALIZATION
# -----------------------------------------
scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

# -----------------------------------------
# HEADER SECTION
# -----------------------------------------
st.markdown("""
<h1 style='text-align:center; color:#1E90FF;'>
🚢 Titanic Survival Prediction System
</h1>

<h4 style='text-align:center; color:gray;'>
Deep Learning Based Passenger Survival Prediction
</h4>
""", unsafe_allow_html=True)

st.divider()

# -----------------------------------------
# PROJECT DESCRIPTION
# -----------------------------------------
st.subheader("📘 Project Description")

st.write("""
This application predicts whether a passenger would survive during the Titanic disaster using an Artificial Neural Network (ANN).

The system uses:
- TensorFlow
- Deep Learning
- Forward Propagation
- Sigmoid Activation Function

The model predicts passenger survival probability based on:
- Passenger Class
- Age
- Fare
""")

st.divider()

# -----------------------------------------
# INPUT SECTION
# -----------------------------------------
st.subheader("🧾 Passenger Input Form")

col1, col2, col3 = st.columns(3)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

with col2:
    age = st.slider(
        "Age",
        1,
        80,
        25
    )

with col3:
    fare = st.number_input(
        "Fare",
        min_value=0.0,
        max_value=600.0,
        value=50.0
    )

# -----------------------------------------
# PREDICTION BUTTON
# -----------------------------------------
if st.button("🔍 Predict Survival"):

    # -----------------------------------------
    # NORMALIZE INPUT
    # -----------------------------------------
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Age': [age],
        'Fare': [fare]
    })

    input_scaled = scaler.transform(input_data)

    x1 = input_scaled[0][0]
    x2 = input_scaled[0][1]
    x3 = input_scaled[0][2]

    # -----------------------------------------
    # FORWARD PROPAGATION
    # -----------------------------------------

    # Hidden Layer Weights
    w1 = 0.11
    w2 = 0.14
    w3 = 0.17

    w4 = 0.21
    w5 = 0.24
    w6 = 0.27

    # Biases
    bh1 = 0.1
    bh2 = 0.1

    # Hidden Layer Calculation
    net_h1 = (x1*w1) + (x2*w2) + (x3*w3) + bh1
    net_h2 = (x1*w4) + (x2*w5) + (x3*w6) + bh2

    # Activation
    out_h1 = tf.sigmoid(net_h1).numpy()
    out_h2 = tf.sigmoid(net_h2).numpy()

    # Output Layer Weights
    w7 = 0.31
    w8 = 0.34

    # Output Bias
    bo = 0.1

    # Output Layer
    net_o1 = (out_h1*w7) + (out_h2*w8) + bo

    predicted_output = tf.sigmoid(net_o1).numpy()

    probability = float(predicted_output)

    # -----------------------------------------
    # PREDICTION LOGIC
    # -----------------------------------------
    if probability > 0.5:
        result = "✅ Survived"
    else:
        result = "❌ Not Survived"

    confidence = probability * 100

    # -----------------------------------------
    # OUTPUT SECTION
    # -----------------------------------------
    st.divider()

    st.subheader("📊 Prediction Output")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            label="Prediction Result",
            value=result
        )

    with c2:
        st.metric(
            label="Survival Probability",
            value=f"{probability:.4f}"
        )

    with c3:
        st.metric(
            label="Confidence Score",
            value=f"{confidence:.2f}%"
        )

    # -----------------------------------------
    # MSE CALCULATION
    # -----------------------------------------
    target = 1.0

    mse = 0.5 * tf.square(target - predicted_output)

    st.write("### Mean Squared Error")
    st.write(float(mse.numpy()))

    # -----------------------------------------
    # VISUALIZATION SECTION
    # -----------------------------------------
    st.divider()

    st.subheader("📈 Probability Visualization")

    labels = ['Survival', 'Non-Survival']
    values = [probability, 1 - probability]

    # BAR CHART
    fig, ax = plt.subplots()

    ax.bar(labels, values)

    ax.set_ylabel("Probability")

    st.pyplot(fig)

    # PIE CHART
    fig2, ax2 = plt.subplots()

    ax2.pie(
        values,
        labels=labels,
        autopct='%1.1f%%'
    )

    st.pyplot(fig2)

# -----------------------------------------
# FOOTER
# -----------------------------------------
st.divider()

st.markdown("""
<center>
Developed using TensorFlow + Streamlit 🚀
</center>
""", unsafe_allow_html=True)