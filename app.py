import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
data = pd.read_csv(url, names=columns)

# Preprocessing
X = data.drop("Outcome", axis=1)
y = data["Outcome"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a RandomForest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions for the test set
y_pred = model.predict(X_test)

# Streamlit interface
st.title('Diabetes Prediction App')
st.write("Enter the following data to predict the diabetes status:")

# Input sliders
pregnancies = st.slider('Number of Pregnancies', 0, 20, 1)
glucose = st.slider('Glucose Level', 50, 200, 120)
blood_pressure = st.slider('Blood Pressure', 30, 140, 80)
skin_thickness = st.slider('Skin Thickness', 0, 99, 20)
insulin = st.slider('Insulin Level', 0, 846, 80)
bmi = st.slider('BMI', 10.0, 70.0, 21.0)
dpf = st.slider('Diabetes Pedigree Function', 0.1, 2.5, 0.5)
age = st.slider('Age', 18, 100, 30)

# Button to predict
if st.button('Predict Diabetes'):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)[0]

    if prediction[0] == 1:
        st.success(f'The prediction result for diabetes is: Positive (Probability {prediction_proba[1]:.2f})')
    else:
        st.error(f'The prediction result for diabetes is: Negative (Probability {prediction_proba[0]:.2f})')

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Generate classification report
report = classification_report(y_test, y_pred, target_names=['Non-Diabetic', 'Diabetic'])
print('Classification Report:\n', report)
