import streamlit as st
import pandas as pd
import pickle

with open("Financial_inclusion_full.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
label_encoders = saved["label_encoders"]

st.title("✅ Prédiction : Possession d’un compte bancaire")
st.write(
    "Remplissez les informations ci-dessous pour prédire si une personne possède un compte bancaire."
)

# 2) Création des champs Streamlit
household_size = st.number_input(
    "Household size :", min_value=1, max_value=50, value=1, step=1
)

age_of_respondent = st.number_input(
    "Age of respondent :", min_value=0, max_value=120, value=30, step=1
)

country = st.selectbox("Country :", ["Kenya", "Tanzania", "Rwanda", "Uganda"])

location_type = st.selectbox("Location type :", ["Rural", "Urban"])

cellphone_access = st.selectbox("Cellphone access :", ["Yes", "No"])

gender_of_respondent = st.selectbox("Gender of respondent :", ["Male", "Female"])

relationship_with_head = st.selectbox(
    "Relationship with head :",
    [
        "Head of Household",
        "Spouse",
        "Child",
        "Parent",
        "Other relative",
        "Other non-relatives",
    ],
)

marital_status = st.selectbox(
    "Marital status :",
    [
        "Married/Living together",
        "Single/Never Married",
        "Widowed",
        "Divorced/Seperated",
        "Dont know",
    ],
)

education_level = st.selectbox(
    "Education level :",
    [
        "Primary education",
        "No formal education",
        "Secondary education",
        "Tertiary education",
        "Vocational/Specialised training",
        "Other/Dont know/RTA",
    ],
)

job_type = st.selectbox(
    "Job type :",
    [
        "Self employed",
        "Informally employed",
        "Farming and Fishing",
        "Remittance Dependent",
        "Other Income",
        "Formally employed Private",
        "Formally employed Government",
        "Government Dependent",
        "Dont Know/Refuse to answer",
    ],
)

year = st.selectbox("Year :", [2016, 2017, 2018])

if st.button("Prédire"):

    country_code = label_encoders["country"].transform([country])[0]
    location_code = label_encoders["location_type"].transform([location_type])[0]
    cellphone_code = label_encoders["cellphone_access"].transform([cellphone_access])[0]
    gender_code = label_encoders["gender_of_respondent"].transform(
        [gender_of_respondent]
    )[0]
    relationship_code = label_encoders["relationship_with_head"].transform(
        [relationship_with_head]
    )[0]
    marital_code = label_encoders["marital_status"].transform([marital_status])[0]
    education_code = label_encoders["education_level"].transform([education_level])[0]
    job_code = label_encoders["job_type"].transform([job_type])[0]

    data = {
        "country": country_code,
        "year": year,
        "location_type": location_code,
        "cellphone_access": cellphone_code,
        "household_size": household_size,
        "age_of_respondent": age_of_respondent,
        "gender_of_respondent": gender_code,
        "relationship_with_head": relationship_code,
        "marital_status": marital_code,
        "education_level": education_code,
        "job_type": job_code,
    }

    df_input = pd.DataFrame([data])

    df_input = df_input[model.feature_names_in_]

    pred = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0]

    if pred == 0:
        st.success(f"🔵 La personne **possède** un compte bancaire ")
    else:
        st.error(f"🔴 La personne **n’a pas** de compte bancaire ")
