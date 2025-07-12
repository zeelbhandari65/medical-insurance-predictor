import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv")

df = load_data()

# Encode categorical features
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

df["sex"] = le_sex.fit_transform(df["sex"])
df["smoker"] = le_smoker.fit_transform(df["smoker"])
df["region"] = le_region.fit_transform(df["region"])

# Split data
X = df.drop("charges", axis=1)
y = df["charges"]

# Train model
model = LinearRegression()
model.fit(X, y)

# App layout
st.set_page_config(page_title="Medical Insurance Cost Predictor")
st.title("🏥 Medical Insurance Cost Predictor")

st.sidebar.header("Enter Patient Information")

age = st.sidebar.slider("Age", 18, 100, 30)
sex = st.sidebar.selectbox("Sex", le_sex.classes_)
bmi = st.sidebar.number_input("BMI", 10.0, 50.0, 25.0)
children = st.sidebar.slider("Children", 0, 5, 0)
smoker = st.sidebar.selectbox("Smoker", le_smoker.classes_)
region = st.sidebar.selectbox("Region", le_region.classes_)

# Prepare input
input_data = pd.DataFrame({
    "age": [age],
    "sex": le_sex.transform([sex]),
    "bmi": [bmi],
    "children": [children],
    "smoker": le_smoker.transform([smoker]),
    "region": le_region.transform([region])
})

# Prediction
if st.button("Predict Insurance Cost 💰"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Insurance Charges: **${prediction:,.2f}**")
