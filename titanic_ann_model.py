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
w1, w2, w3 = -0.8, 0.2, -0.6
w4, w5, w6 = 0.5, -0.7, 0.3

bh1 = -0.3
bh2 = 0.2

# Hidden Layer
net_h1 = (x1*w1) + (x2*w2) + (x3*w3) + bh1
net_h2 = (x1*w4) + (x2*w5) + (x3*w6) + bh2

out_h1 = tf.sigmoid(net_h1)
out_h2 = tf.sigmoid(net_h2)

print("Hidden Output h1 =", out_h1.numpy())
print("Hidden Output h2 =", out_h2.numpy())

# Output layer
w7 = -1.2
w8 = 1.1
bo = -0.4

net_o1 = (out_h1*w7) + (out_h2*w8) + bo

out_o1 = tf.sigmoid(net_o1)

prediction = float(out_o1.numpy())

print("\nPredicted Output =", prediction)

if prediction >= 0.5:
    print("Prediction : Survived")
else:
    print("Prediction : Not Survived")
