import pandas as pd
df = pd.read_csv("titanic_ann_model.csv")
df.head()

import tensorflow as tf
import math

X = df[['Pclass', 'Age', 'Fare']]
y = df['Survived']

print(X.head())
print(y.head())

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

print(X_scaled[:5])

x1 = X_scaled[0][0]
x2 = X_scaled[0][1]
x3 = X_scaled[0][2]

target = y[0]

print("Normalized Inputs:")
print(x1, x2, x3)

print("Target:", target)

import tensorflow as tf

inputs = tf.constant([x1, x2, x3], dtype=tf.float32)

print(inputs)

# Inputs
x1 = 0.2     # Pclass
x2 = 0.24    # Age
x3 = 0.80    # Fare

# Convert to tensor
inputs = tf.constant([x1, x2, x3], dtype=tf.float32)

print("Inputs =", inputs)

# Hidden neuron h1 weights
w1 = 0.11
w2 = 0.14
w3 = 0.17

# Hidden neuron h2 weights
w4 = 0.21
w5 = 0.24
w6 = 0.27

bh1 = 0.1
bh2 = 0.1
net_h1 = (x1*w1) + (x2*w2) + (x3*w3) + bh1

print("Net Input h1 =", net_h1)
net_h2 = (x1*w4) + (x2*w5) + (x3*w6) + bh2

print("Net Input h2 =", net_h2)
out_h1 = tf.sigmoid(net_h1)
out_h2 = tf.sigmoid(net_h2)

print("Output h1 =", out_h1.numpy())
print("Output h2 =", out_h2.numpy())

w7 = 0.31
w8 = 0.34
bo = 0.1
net_o1 = (out_h1*w7) + (out_h2*w8) + bo

print("Net Input Output Neuron =", net_o1.numpy())
out_o1 = tf.sigmoid(net_o1)

print("Output =", out_o1.numpy())

import tensorflow as tf

# Target Output
target = 1.0

# Predicted Output from forward propagation
predicted_output = 0.645   # example value

# Mean Squared Error
mse = 0.5 * tf.square(target - predicted_output)

print("Mean Squared Error =", mse.numpy())

