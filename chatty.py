# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 09:49:13 2024

@author: swaro
"""
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import re
from datetime import datetime
import random

# Dictionary of diseases and corresponding reference links
disease_links = {
    "Flu": "https://www.healthline.com/health/flu-causes",
    "Common Cold": "https://www.healthline.com/health/common-cold",
    "Migraine": "https://www.mayoclinic.org/diseases-conditions/migraine-headache/symptoms-causes/syc-20360201",
    "Pneumonia": "https://www.cdc.gov/pneumonia/index.html",
    "COVID-19": "https://www.cdc.gov/coronavirus/2019-ncov/index.html",
    "Asthma": "https://www.webmd.com/asthma/guide/asthma-treatment-care",
    "Malaria": "https://www.who.int/news-room/fact-sheets/detail/malaria",
    "Tuberculosis": "https://www.cdc.gov/tb/topic/basics/default.htm",
    "Heart Attack": "https://www.heart.org/en/health-topics/heart-attack",
    "Gastroenteritis": "https://www.healthline.com/health/gastroenteritis",
    "Dengue Fever": "https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue",
    "Measles": "https://www.who.int/news-room/fact-sheets/detail/measles",
    "Bronchitis": "https://www.webmd.com/lung/understanding-bronchitis-treatment",
    "RSV": "https://www.cdc.gov/rsv/index.html",
    "Tonsillitis": "https://www.healthline.com/health/tonsillitis",
    "Lung Cancer": "https://www.cancer.org/cancer/lung-cancer.html",
    "Ebola": "https://www.who.int/health-topics/ebola",
    "Strep Throat": "https://www.healthline.com/health/strep-throat",
    "Food Poisoning": "https://www.cdc.gov/foodsafety/food-poisoning.html",
    "Mononucleosis": "https://www.healthline.com/health/mononucleosis",
    "Pulmonary Embolism": "https://www.healthline.com/health/pulmonary-embolism",
    "Lung Fibrosis": "https://www.healthline.com/health/lung-fibrosis",
    "COPD": "https://www.cdc.gov/copd/index.html",
    "Bacterial Pneumonia": "https://www.webmd.com/lung/what-is-bacterial-pneumonia",
    "Whooping Cough": "https://www.cdc.gov/pertussis/index.html",
    "Norovirus": "https://www.cdc.gov/norovirus/index.html",
    "Hepatitis A": "https://www.cdc.gov/hepatitis/hav/index.html",
    "Rheumatic Fever": "https://www.mayoclinic.org/diseases-conditions/rheumatic-fever/symptoms-causes/syc-20383554",
    "SARS": "https://www.who.int/health-topics/sars",
    "Rheumatoid Arthritis": "https://www.rheumatology.org/Portals/0/Files/Rheumatoid-Arthritis-2021-Patient-Education-Brochure-ENG.pdf",
    "Varicella": "https://www.cdc.gov/chickenpox/index.html",
    "Cholera": "https://www.who.int/health-topics/cholera",
    "Rubella": "https://www.cdc.gov/rubella/index.html",
    "Legionnaires' Disease": "https://www.cdc.gov/legionella/index.html",
    "Irritable Bowel Syndrome": "https://www.ibsinfo.org/",
    "Peptic Ulcer": "https://www.healthline.com/health/peptic-ulcer",
    "Gout": "https://www.healthline.com/health/gout",
    "Psoriasis": "https://www.psoriasis.org/",
    "Lyme Disease": "https://www.cdc.gov/lyme/index.html",
    "Zika Virus": "https://www.cdc.gov/zika/index.html",
    "Hantavirus": "https://www.cdc.gov/hantavirus/index.html",
    "Chronic Fatigue Syndrome": "https://www.cdc.gov/cfs/index.html",
    "Endocarditis": "https://www.mayoclinic.org/diseases-conditions/endocarditis/symptoms-causes/syc-20366388",
    "Schistosomiasis": "https://www.cdc.gov/parasites/schistosomiasis/index.html",
    "Cytomegalovirus": "https://www.cdc.gov/cmv/index.html",
    "Hepatitis C": "https://www.cdc.gov/hepatitis/hcv/index.html",
    "Toxic Shock Syndrome": "https://www.mayoclinic.org/diseases-conditions/toxic-shock-syndrome/symptoms-causes/syc-20384396",
    "Histoplasmosis": "https://www.cdc.gov/fungal/diseases/histoplasmosis/index.html",
    "Ehrlichiosis": "https://www.cdc.gov/ehrlichiosis/index.html",
    "Meningococcal Disease": "https://www.cdc.gov/meningococcal/index.html",
    "Rabies": "https://www.cdc.gov/rabies/index.html",
    "Chagas Disease": "https://www.cdc.gov/parasites/chagas/index.html",
    "Bubonic Plague": "https://www.cdc.gov/plague/index.html",
    "Coronary Artery Disease": "https://www.heart.org/en/health-topics/heart-attack/about-heart-attacks/coronary-artery-disease",
    "Dysentery": "https://www.healthline.com/health/dysentery",
    "Cryptosporidiosis": "https://www.cdc.gov/parasites/crypto/index.html",
    "Melioidosis": "https://www.cdc.gov/melioidosis/index.html",
    "Middle East Respiratory Syndrome (MERS)": "https://www.cdc.gov/coronavirus/mers/index.html",
    "Nipah Virus": "https://www.cdc.gov/nipah/index.html",
    "Cystic Fibrosis": "https://www.cff.org/",
    "Fibromyalgia": "https://www.healthline.com/health/fibromyalgia",
    "Toxoplasmosis": "https://www.cdc.gov/parasites/toxoplasmosis/index.html",
    "Viral Meningitis": "https://www.cdc.gov/meningitis/viral.html",
    "Chronic Bronchitis": "https://www.healthline.com/health/chronic-bronchitis",
    "Acute Respiratory Distress Syndrome": "https://www.healthline.com/health/acute-respiratory-distress-syndrome",
    "Epiglottitis": "https://www.healthline.com/health/epiglottitis",
    "Fibroids": "https://www.healthline.com/health/uterine-fibroids",
    "Sjogren's Syndrome": "https://www.sjogrens.org/",
    "Anemia": "https://www.healthline.com/health/anemia",
    "Angina": "https://www.healthline.com/health/angina",
    "Amoebiasis": "https://www.healthline.com/health/amoebiasis",
    "Bacterial Meningitis": "https://www.cdc.gov/meningitis/bacterial.html",
    "RSV": "https://www.cdc.gov/rsv/index.html",
    "Tension Headache": "https://medlineplus.gov/ency/article/000797.htm#:~:text=Tension%20headaches%20occur%20when%20neck,tends%20to%20run%20in%20families.",
    "Ebola": "https://www.cdc.gov/ebola/about/index.html#:~:text=Ebola%20disease%20is%20caused%20by%20a%20group%20of%20viruses%2C%20known,primarily%20in%20sub%2DSaharan%20Africa.",
    "Typhoid": "https://www.nhs.uk/conditions/typhoid-fever/#:~:text=Typhoid%20fever%20is%20a%20bacterial,that%20cause%20salmonella%20food%20poisoning.",
    "Rubella": "https://www.who.int/news-room/fact-sheets/detail/rubella#:~:text=Rubella%20is%20a%20contagious%20viral,(CRS)%20each%20year%20worldwide."
}


def load_data():
    data = pd.read_csv('symptom_disease.csv') 
    return data

def train_model(data):
    X = data.iloc[:, :-1]  
    y = data.iloc[:, -1]   
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    rf_model = RandomForestClassifier()
    rf_model.fit(X, y_encoded)
    
    return rf_model, label_encoder

def predict_disease(symptoms, model, label_encoder, feature_columns):
    input_vector = np.zeros(len(feature_columns))
    
    for symptom in symptoms:
        if symptom in feature_columns:
            symptom_index = feature_columns.index(symptom)
            input_vector[symptom_index] = 1
    
    prediction = model.predict([input_vector])
    disease_name = label_encoder.inverse_transform(prediction)
    
    return disease_name[0]

def extract_symptoms(user_input, feature_columns):
    cleaned_input = re.sub(r'[^a-zA-Z\s]', '', user_input.lower())
    
    detected_symptoms = [symptom for symptom in feature_columns if symptom.lower() in cleaned_input]
    return detected_symptoms

def handle_greeting(user_input):
    greetings = ["hello", "hi", "hey"]
    if any(greeting in user_input.lower() for greeting in greetings):
        return random.choice(["Hello! How can I assist you today?", "Hi there! How can I help you?", "Hey! What can I do for you?"])

def handle_nlp_questions(user_input):
    user_input = user_input.lower()
    if "time" in user_input:
        current_time = datetime.now().strftime("%H:%M")
        return f"The current time is {current_time}."
    elif "date" in user_input:
        current_date = datetime.now().strftime("%B %d, %Y")
        return f"Today's date is {current_date}."
    elif "how are you" in user_input:
        return random.choice(["I'm just a bunch of code, but thanks for asking! How are you?", "I'm doing great, thanks for asking! How about you?", "I'm here to help you, so I'm doing well!"])
    elif "your name" in user_input:
        return "I don't have a name, but you can call me your virtual health assistant!"
    elif "gender" in user_input:
        return "I'm just an AI, so I don't have a gender!"
    else:
        return None

def get_disease_link(disease_name):
    return disease_links.get(disease_name, "Sorry, I don't have a link for this disease.")

def main():
    st.set_page_config(page_title="Medical Diagnosis Chatbot", page_icon="💬", layout="wide")

    st.markdown("""
        <style>
        body {
            background-color: #e0f7fa;
        }
        h1, h2, h3 {
            color: #004d40;
        }
        .stButton>button {
            background-color: #00796b;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("Medical Diagnosis Chatbot")
    st.subheader("Describe your symptoms to get a possible diagnosis.")

    data = load_data()
    rf_model, label_encoder = train_model(data)

    user_input = st.text_input("Enter your symptoms (e.g., 'I have a headache and fever'):")

    if user_input:
        greeting_response = handle_greeting(user_input)
        nlp_response = handle_nlp_questions(user_input)

        if greeting_response:
            st.write(greeting_response)
        elif nlp_response:
            st.write(nlp_response)
        else:
            symptoms = extract_symptoms(user_input, list(data.columns[:-1]))
            
            if symptoms:
                diagnosis = predict_disease(symptoms, rf_model, label_encoder, list(data.columns[:-1]))
                disease_link = get_disease_link(diagnosis)
                st.success(f"Based on the symptoms you mentioned, you may have: {diagnosis}. It's always best to consult a physician for an accurate diagnosis.")
                st.markdown(f"**Learn more about {diagnosis}:** [Click here]({disease_link})")
            else:
                st.error("Sorry, I couldn't detect any known symptoms. Please try again with more details.")

if __name__ == "__main__":
    main()
