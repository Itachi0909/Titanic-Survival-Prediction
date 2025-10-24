import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('model.pkl')

st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to predict survival:")

# Input fields
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 29)
sibsp = st.number_input("Number of Siblings/Spouses aboard (SibSp)", 0, 10, 0)
parch = st.number_input("Number of Parents/Children aboard (Parch)", 0, 10, 0)
fare = st.slider("Ticket Fare", 0.0, 500.0, 32.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Encode categorical features same as in training
sex = 1 if sex == "female" else 0  # (male=0, female=1)

# Encode Embarked: use the same mapping you used during training
embarked_mapping = {'C': 0, 'Q': 1, 'S': 2}
embarked = embarked_mapping[embarked]

# Create DataFrame exactly matching model’s training format
data = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked": [embarked]
})

# Prediction
if st.button("Predict Survival"):
    try:
        prediction = model.predict(data)
        proba = model.predict_proba(data)[0][1]
        if prediction[0] == 1:
            st.success(f"✅ Passenger Survived (Probability: {proba:.2f})")
        else:
            st.error(f"❌ Passenger Did Not Survive (Probability: {proba:.2f})")
    except Exception as e:
        st.error(f"Error: {e}")
