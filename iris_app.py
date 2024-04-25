# Import necessary packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# *********** MODEL TRAINING ******************
# Set random seed
seed = 42

# Read original dataset
iris_df = pd.read_csv("Iris.csv")
iris_df.sample(frac=1, random_state=seed)

# Select features and target
X = iris_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Species']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed, stratify=y
)

# Create and train classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train.values.ravel())  # Use ravel() to create a 1D array

# Save model
filename = "rf_model.sav"
pickle.dump(clf, open(filename, 'wb'))

# Load model
clf_loaded = pickle.load(open(filename, 'rb'))

# Predict on test set and calculate accuracy (optional, for validation only)
y_pred = clf_loaded.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# *********** STREAMLIT WEB APP ************************
st.title('Classifying Iris Flowers')

# Input sliders for the plant features
st.header("Plant Features")
sepal_l = st.slider('Sepal length (cm)', 4.3, 7.9, 5.0)
sepal_w = st.slider('Sepal width (cm)', 2.0, 4.4, 3.0)
petal_l = st.slider('Petal length (cm)', 1.0, 6.9, 1.3)
petal_w = st.slider('Petal width (cm)', 0.1, 2.5, 0.2)

# Prediction button
if st.button("Predict type of Iris"):
    result = clf_loaded.predict(np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
    st.success(f"The predicted species is: {result[0]}")

st.text('')
st.markdown('`Initial code was developed by` [VarunKumar G K]')
