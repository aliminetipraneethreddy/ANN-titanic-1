import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("titanic_ann_model.csv")

# Features and target
X = df[['Pclass', 'Age', 'Fare']]
y = df['Survived']

# Normalize data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Select one passenger record
index = 3     # change index to test different passengers

x1 = X_scaled[index][0]
x2 = X_scaled[index][1]
x3 = X_scaled[index][2]

target = y[index]

print("Normalized Inputs:")
print(x1, x2, x3)

print("Actual Target =", target)

# ---------------- FORWARD PROPAGATION ----------------

# Hidden layer weights
w1, w2, w3 = 0.11, -0.14, 0.17
w4, w5, w6 = -0.21, 0.24, -0.27

# Bias
bh1 = 0.1
bh2 = 0.1

# Hidden neuron calculations
net_h1 = (x1*w1) + (x2*w2) + (x3*w3) + bh1
net_h2 = (x1*w4) + (x2*w5) + (x3*w6) + bh2

out_h1 = tf.sigmoid(net_h1)
out_h2 = tf.sigmoid(net_h2)

print("\nHidden Layer Outputs")
print("h1 =", out_h1.numpy())
print("h2 =", out_h2.numpy())

# Output layer weights
w7 = 0.31
w8 = -0.34
bo = 0.1

# Output neuron
net_o1 = (out_h1*w7) + (out_h2*w8) + bo

out_o1 = tf.sigmoid(net_o1)

prediction = out_o1.numpy()

print("\nPredicted Output =", prediction)

# Final prediction
if prediction >= 0.5:
    print("Passenger Survived")
else:
    print("Passenger Did NOT Survive")

# ---------------- ERROR CALCULATION ----------------

mse = 0.5 * tf.square(target - prediction)

print("\nMean Squared Error =", mse.numpy())
