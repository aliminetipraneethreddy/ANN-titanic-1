import streamlit as st
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

# Features
X = df[['Pclass', 'Age', 'Fare']]

# Fill missing age values
X['Age'] = X['Age'].fillna(X['Age'].mean())

# Normalize
scaler = MinMaxScaler()
scaler.fit(X)

# -----------------------------------------
# SIGMOID FUNCTION
# -----------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# -----------------------------------------
# HEADER
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
# DESCRIPTION
# -----------------------------------------
st.subheader("📘 Project Description")

st.write("""
This application predicts Titanic passenger survival using:
- Artificial Neural Network (ANN)
- Forward Propagation
- Sigmoid Activation Function

The prediction is based on:
- Passenger Class
- Age
- Fare
""")

st.divider()

# -----------------------------------------
# INPUT FORM
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
# PREDICT BUTTON
# -----------------------------------------
if st.button("🔍 Predict Survival"):

    # Input dataframe
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Age': [age],
        'Fare': [fare]
    })

    # Normalize
    input_scaled = scaler.transform(input_data)

    x1 = input_scaled[0][0]
    x2 = input_scaled[0][1]
    x3 = input_scaled[0][2]

    # -----------------------------------------
    # FORWARD PROPAGATION
    # -----------------------------------------

    # Hidden Layer Weights
    w1, w2, w3 = 0.11, 0.14, 0.17
    w4, w5, w6 = 0.21, 0.24, 0.27

    # Biases
    bh1, bh2 = 0.1, 0.1

    # Hidden Layer
    net_h1 = (x1*w1) + (x2*w2) + (x3*w3) + bh1
    net_h2 = (x1*w4) + (x2*w5) + (x3*w6) + bh2

    out_h1 = sigmoid(net_h1)
    out_h2 = sigmoid(net_h2)

    # Output Layer
    w7, w8 = 0.31, 0.34
    bo = 0.1

    net_o1 = (out_h1*w7) + (out_h2*w8) + bo

    predicted_output = sigmoid(net_o1)

    probability = float(predicted_output)

    # -----------------------------------------
    # PREDICTION
    # -----------------------------------------
    if probability > 0.5:
        result = "✅ Survived"
    else:
        result = "❌ Not Survived"

    confidence = probability * 100

    # -----------------------------------------
    # OUTPUT
    # -----------------------------------------
    st.divider()

    st.subheader("📊 Prediction Output")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Prediction Result",
            result
        )

    with c2:
        st.metric(
            "Survival Probability",
            f"{probability:.4f}"
        )

    with c3:
        st.metric(
            "Confidence Score",
            f"{confidence:.2f}%"
        )

    # -----------------------------------------
    # MSE
    # -----------------------------------------
    target = 1.0

    mse = 0.5 * ((target - predicted_output) ** 2)

    st.write("### Mean Squared Error")
    st.write(mse)

    # -----------------------------------------
    # VISUALIZATION
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
Developed using Streamlit + ANN 🚀
</center>
""", unsafe_allow_html=True)
