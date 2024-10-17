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
import csv
import json
import os

disease_links = {
    "Miliary Tuberculosis": "https://www.cdc.gov/tb/publications/factsheets/general/tb.htm",
    "Ehrlichiosis": "https://www.cdc.gov/ehrlichiosis/",
    "Bunyavirus Infection": "https://www.cdc.gov/vhf/bunyavirus/index.html",
    "HIV/AIDS": "https://www.cdc.gov/hiv/default.html",
    "Whipple's Disease": "https://rarediseases.info.nih.gov/diseases/7444/whipples-disease",
    "Tularemia": "https://www.cdc.gov/tularemia/index.html",
    "Shigella Infection": "https://www.cdc.gov/shigella/index.html",
    "Bacterial Pneumonia": "https://www.cdc.gov/pneumonia/causes.html",
    "Lassa Fever": "https://www.cdc.gov/vhf/lassa/index.html",
    "Severe Acute Respiratory Syndrome": "https://www.who.int/health-topics/severe-acute-respiratory-syndrome",
    "Listeriosis": "https://www.cdc.gov/listeria/index.html",
    "Varicella-Zoster Virus": "https://www.cdc.gov/vaccines/vpd/varicella/index.html",
    "Cluster Headache": "https://www.mayoclinic.org/diseases-conditions/cluster-headache/symptoms-causes/syc-20352080",
    "West Nile Virus": "https://www.cdc.gov/westnile/index.html",
    "Helicobacter Pylori Infection": "https://www.cdc.gov/ulcer/h-pylori.html",
    "Primary Amebic Meningoencephalitis": "https://www.cdc.gov/parasites/naegleria/",
    "Avian Influenza": "https://www.cdc.gov/flu/avianflu/",
    "Rocky Mountain Spotted Fever": "https://www.cdc.gov/rmsf/index.html",
    "Fascioliasis": "https://www.cdc.gov/parasites/fasciola/",
    "Infectious Mononucleosis": "https://www.cdc.gov/epstein-barr/about-mono.html",
    "Ebola": "https://www.cdc.gov/vhf/ebola/index.html",
    "Hantavirus Pulmonary Syndrome": "https://www.cdc.gov/hantavirus/hps/index.html",
    "Hepatitis B": "https://www.cdc.gov/hepatitis/hbv/index.htm",
    "Onchocerciasis": "https://www.cdc.gov/parasites/onchocerciasis/",
    "Strongyloidiasis": "https://www.cdc.gov/parasites/strongyloides/index.html",
    "Anthrax": "https://www.cdc.gov/anthrax/index.html",
    "Zoonotic Influenza": "https://www.cdc.gov/flu/avianflu/zoonotic.htm",
    "Epidemic Typhus": "https://www.cdc.gov/typhus/epidemic/index.html",
    "Bubonic Plague": "https://www.cdc.gov/plague/index.html",
    "Hand-Foot-Mouth Disease": "https://www.cdc.gov/hand-foot-mouth/index.html",
    "Guillain-Barré Syndrome": "https://www.cdc.gov/campylobacter/guillain-barre.html",
    "Human Metapneumovirus": "https://www.cdc.gov/surveillance/nrevss/hmpv/index.html",
    "Pneumocystis Pneumonia": "https://www.cdc.gov/fungal/diseases/pneumocystis-pneumonia/index.html",
    "Middle East Respiratory Syndrome": "https://www.who.int/emergencies/mers-cov/en/",
    "Influenza A": "https://www.cdc.gov/flu/avianflu/influenza-a-virus-subtypes.htm",
    "Eastern Equine Encephalitis Virus": "https://www.cdc.gov/easternequineencephalitis/index.html",
    "Marburg Virus": "https://www.cdc.gov/vhf/marburg/index.html",
    "Amebiasis": "https://www.cdc.gov/parasites/amebiasis/",
    "Shingles": "https://www.cdc.gov/shingles/about/index.html",
    "Severe Malaria": "https://www.cdc.gov/malaria/about/index.html",
    "Influenza": "https://www.cdc.gov/flu/index.html",
    "Conjunctivitis": "https://www.cdc.gov/conjunctivitis/",
    "Leishmaniasis": "https://www.cdc.gov/parasites/leishmaniasis/",
    "Campylobacter Infection": "https://www.cdc.gov/campylobacter/index.html",
    "Coccidioidomycosis": "https://www.cdc.gov/fungal/diseases/coccidioidomycosis/index.html",
    "Herpes Zoster Virus": "https://www.cdc.gov/shingles/about/herpes-zoster.html",
    "Asthma": "https://www.cdc.gov/asthma/",
    "Rotavirus Infection": "https://www.cdc.gov/rotavirus/",
    "Rheumatic Fever": "https://www.cdc.gov/groupastrep/diseases-public/rheumatic-fever.html",
    "Acute Lymphoblastic Leukemia": "https://www.cancer.org/cancer/acute-lymphocytic-leukemia.html",
    "Meningococcal Infection": "https://www.cdc.gov/meningococcal/index.html",
    "Norovirus": "https://www.cdc.gov/norovirus/index.html",
    "Schistosomiasis": "https://www.cdc.gov/parasites/schistosomiasis/",
    "Tuberculosis": "https://www.cdc.gov/tb/default.htm",
    "Syphilis": "https://www.cdc.gov/std/syphilis/",
    "Zoster Virus": "https://www.cdc.gov/shingles/about/herpes-zoster.html",
    "Encephalomyelitis": "https://www.mayoclinic.org/diseases-conditions/encephalomyelitis/symptoms-causes/syc-20374560",
    "Babesia Infection": "https://www.cdc.gov/parasites/babesiosis/",
    "Malnutrition": "https://www.who.int/news-room/fact-sheets/detail/malnutrition",
    "Salmonella Infection": "https://www.cdc.gov/salmonella/index.html",
    "SARS": "https://www.who.int/health-topics/severe-acute-respiratory-syndrome",
    "Campylobacteriosis": "https://www.cdc.gov/campylobacter/index.html",
    "Lung Cancer": "https://www.cdc.gov/cancer/lung/index.htm",
    "Histoplasmosis": "https://www.cdc.gov/fungal/diseases/histoplasmosis/index.html",
    "Hemorrhagic Fever": "https://www.cdc.gov/vhf/index.html",
    "Rubella": "https://www.cdc.gov/rubella/index.html",
    "Hand-Foot-Mouth Syndrome": "https://www.cdc.gov/hand-foot-mouth/index.html",
    "Rift Valley Fever Virus": "https://www.cdc.gov/vhf/rvf/index.html",
    "Acute Gastroenteritis": "https://www.cdc.gov/norovirus/about/index.html",
    "Respiratory Adenovirus": "https://www.cdc.gov/adenovirus/index.html",
    "Pneumonia": "https://www.cdc.gov/pneumonia/index.html",
    "Respiratory Syncytial Virus (RSV)": "https://www.cdc.gov/rsv/index.html",
    "Orthomyxovirus Infection": "https://www.cdc.gov/flu/index.html",
    "Scarlet Fever": "https://www.cdc.gov/groupastrep/diseases-public/scarlet-fever.html",
    "Abnormal Menstrual Bleeding": "https://www.mayoclinic.org/diseases-conditions/menorrhagia/symptoms-causes/syc-20352829",
    "Taeniasis": "https://www.cdc.gov/parasites/taeniasis/",
    "Bronchitis": "https://www.cdc.gov/bronchitis/",
    "RSV": "https://www.cdc.gov/rsv/index.html",
    "Chikungunya": "https://www.cdc.gov/chikungunya/index.html",
    "Measles Virus": "https://www.cdc.gov/measles/index.html",
    "Monkeypox Virus": "https://www.cdc.gov/poxvirus/monkeypox/index.html",
    "Shiga Toxin-Producing E. coli": "https://www.cdc.gov/ecoli/index.html",
    "Infectious Polyneuritis": "https://rarediseases.info.nih.gov/diseases/7405/infectious-polyneuritis",
    "Streptococcal Infection": "https://www.cdc.gov/groupastrep/diseases-public/strep-throat.html",
    "Influenza B": "https://www.cdc.gov/flu/index.html",
    "Chikungunya Fever": "https://www.cdc.gov/chikungunya/index.html",
    "Hemorrhagic Cystitis": "https://rarediseases.org/rare-diseases/hemorrhagic-cystitis/",
    "Salmonellosis": "https://www.cdc.gov/salmonella/index.html",
    "Hemolytic Uremic Syndrome": "https://www.cdc.gov/foodsafety/hemolytic-uremic-syndrome.html",
    "Rift Valley Fever": "https://www.cdc.gov/vhf/rvf/index.html",
    "Cholera": "https://www.cdc.gov/cholera/index.html",
    "Paratyphoid Fever": "https://www.cdc.gov/typhoid-fever/paratyphoid.html",
    "Malaria Hemorrhagic Fever": "https://www.cdc.gov/malaria/about/index.html",
    "Whooping Cough": "https://www.cdc.gov/pertussis/index.html",
    "Cryptosporidiosis": "https://www.cdc.gov/parasites/crypto/index.html",
    "Tonsillitis": "https://www.mayoclinic.org/diseases-conditions/tonsillitis/symptoms-causes/syc-20378479",
    "Typhoid Fever": "https://www.cdc.gov/typhoid-fever/index.html",
    "Toxocariasis": "https://www.cdc.gov/parasites/toxocariasis/",
    "Poliomyelitis": "https://www.cdc.gov/polio/index.html",
    "Hepatitis A": "https://www.cdc.gov/hepatitis/hav/index.htm",
    "Bacillus Anthracis": "https://www.cdc.gov/anthrax/index.html",
    "Ebola Virus": "https://www.cdc.gov/vhf/ebola/index.html",
    "Leptospirosis": "https://www.cdc.gov/leptospirosis/index.html",
    "Human T-Lymphotropic Virus": "https://www.cdc.gov/htlv/",
    "Lassa Virus": "https://www.cdc.gov/vhf/lassa/index.html",
    "Gastroenteritis": "https://www.cdc.gov/norovirus/about/index.html",
    "Alveolar Echinococcosis": "https://www.cdc.gov/parasites/echinococcosis/gen_info/ae-faqs.html",
    "Gout": "https://www.cdc.gov/arthritis/basics/gout.html",
    "Pulmonary Embolism": "https://www.cdc.gov/ncbddd/dvt/index.html",
    "Plague": "https://www.cdc.gov/plague/index.html",
    "Respiratory Anthrax": "https://www.cdc.gov/anthrax/basics/index.html",
    "Eye Infection": "https://www.cdc.gov/conjunctivitis/",
    "Borna Virus Disease": "https://www.cdc.gov/ncezid/dvbd/bornavirus/index.html",
    "Migraine": "https://www.mayoclinic.org/diseases-conditions/migraine-headache/symptoms-causes/syc-20360201",
    "Japanese B Encephalitis Virus": "https://www.cdc.gov/japaneseencephalitis/index.html",
    "Yellow Fever": "https://www.cdc.gov/yellowfever/index.html",
    "Bloating": "https://www.healthline.com/health/bloating",
    "Lyme Disease": "https://www.cdc.gov/lyme/index.html",
    "Bovine Spongiform Encephalopathy": "https://www.cdc.gov/prions/bse/index.html",
    "RSV Pneumonia": "https://www.cdc.gov/rsv/index.html",
    "Acute Viral Hepatitis": "https://www.cdc.gov/hepatitis/abc/index.htm",
    "Sepsis": "https://www.cdc.gov/sepsis/index.html",
    "Venezuelan Equine Encephalitis": "https://www.cdc.gov/easternequineencephalitis/index.html",
    "Zika Virus": "https://www.cdc.gov/zika/index.html",
    "E. Coli Infection": "https://www.cdc.gov/ecoli/index.html",
    "Leprosy": "https://www.cdc.gov/leprosy/index.html",
    "Smallpox": "https://www.cdc.gov/smallpox/index.html",
    "Zika Fever": "https://www.cdc.gov/zika/index.html",
    "HIV Infection": "https://www.cdc.gov/hiv/default.html",
    "Vision Problems": "https://www.cdc.gov/ncbddd/visionloss/index.html",
    "COVID-19": "https://www.cdc.gov/coronavirus/2019-ncov/index.html",
    "Measles": "https://www.cdc.gov/measles/index.html",
    "Mononucleosis": "https://www.cdc.gov/epstein-barr/about-mono.html",
    "Viral Hepatitis E": "https://www.cdc.gov/hepatitis/hev/index.htm",
    "Rubeola Virus": "https://www.cdc.gov/measles/index.html",
    "Hemorrhagic Septicemia": "https://www.merckvetmanual.com/generalized-conditions/hemorrhagic-septicemia",
    "Respiratory Syncytial Virus Infection": "https://www.cdc.gov/rsv/index.html",
    "Dengue Fever": "https://www.cdc.gov/dengue/index.html",
    "Neonatal Sepsis": "https://www.cdc.gov/groupbstrep/about/prevention.html",
    "Malaria": "https://www.cdc.gov/malaria/index.html",
    "Varicella": "https://www.cdc.gov/vaccines/vpd/varicella/index.html",
    "Strep Throat": "https://www.cdc.gov/groupastrep/diseases-public/strep-throat.html",
    "Rheumatoid Arthritis": "https://www.cdc.gov/arthritis/basics/rheumatoid-arthritis.html",
    "Enteric Fever": "https://www.cdc.gov/typhoid-fever/index.html",
    "Human Papillomavirus": "https://www.cdc.gov/std/hpv/stdfact-hpv.htm",
    "Acinetobacter Infection": "https://www.cdc.gov/acinetobacter/index.html",
    "Legionnaires' Disease": "https://www.cdc.gov/legionella/index.html",
    "Japanese Encephalitis": "https://www.cdc.gov/japaneseencephalitis/index.html",
    "Rabies": "https://www.cdc.gov/rabies/index.html",
    "Melioidosis": "https://www.cdc.gov/melioidosis/index.html",
    "Psoriatic Arthritis": "https://www.cdc.gov/arthritis/basics/psoriatic-arthritis.html",
    "Tick-Borne Encephalitis": "https://www.cdc.gov/tickborne/index.html",
    "Nipah Virus": "https://www.cdc.gov/vhf/nipah/index.html",
    "Chlamydia Psittaci": "https://www.cdc.gov/pneumonia/atypical/chlamydia-psittaci.html",
    "Dengue Hemorrhagic Fever": "https://www.cdc.gov/dengue/index.html",
    "Mumps": "https://www.cdc.gov/mumps/index.html",
    "Shigellosis": "https://www.cdc.gov/shigella/index.html",
    "Tension Headache": "https://www.mayoclinic.org/diseases-conditions/tension-headache/symptoms-causes/syc-20373224",
    "Rabies Hemorrhagic Fever": "https://www.cdc.gov/rabies/index.html",
    "Vibrio Cholerae": "https://www.cdc.gov/cholera/index.html",
    "Lyme Borreliosis": "https://www.cdc.gov/lyme/index.html",
    "Bacterial Meningitis": "https://www.cdc.gov/meningitis/bacterial.html",
    "Hantavirus Infection": "https://www.cdc.gov/hantavirus/index.html",
    "Brucellosis": "https://www.cdc.gov/brucellosis/index.html",
    "MERS": "https://www.who.int/emergencies/mers-cov/en/",
    "Enterovirus Infection": "https://www.cdc.gov/non-polio-enterovirus/index.html",
    "Clostridium Difficile Infection": "https://www.cdc.gov/cdiff/index.html",
    "Q Fever": "https://www.cdc.gov/qfever/index.html",
    "Flu": "https://www.healthline.com/health/flu-causes",
    "Giardiasis": "https://www.cdc.gov/parasites/giardia/index.html",
    "SARS-CoV-2 Variant": "https://www.cdc.gov/coronavirus/2019-ncov/variants/variant.html",
    "Lymphatic Filariasis": "https://www.cdc.gov/parasites/lymphaticfilariasis/index.html",
    "Coronavirus HKU1": "https://www.cdc.gov/coronavirus/2019-ncov/variants/variant.html",
    "Hepatitis C Virus": "https://www.cdc.gov/hepatitis/hcv/index.htm",
    "Kaposi's Sarcoma": "https://www.cdc.gov/cancer/kaposi/index.htm",
    "Non-Hodgkin Lymphoma": "https://www.cdc.gov/cancer/lymphoma/index.htm",
    "Marburg Hemorrhagic Fever": "https://www.cdc.gov/vhf/marburg/index.html",
    "Influenza Hemorrhagic Fever": "https://www.cdc.gov/flu/index.html",
    "Cutaneous Anthrax": "https://www.cdc.gov/anthrax/basics/index.html",
    "Toxoplasmosis": "https://www.cdc.gov/parasites/toxoplasmosis/index.html",
    "Hepatitis E": "https://www.cdc.gov/hepatitis/hev/index.htm",
    "Pneumococcal Infection": "https://www.cdc.gov/pneumococcal/index.html",
    "Rabies Virus": "https://www.cdc.gov/rabies/index.html",
    "Hepatitis D Virus": "https://www.cdc.gov/hepatitis/hdv/index.htm",
    "Stomach Cancer": "https://www.cdc.gov/cancer/stomach/index.htm",
    "Prostate Cancer": "https://www.cdc.gov/cancer/prostate/index.htm",
    "Rotavirus": "https://www.cdc.gov/rotavirus/index.html",
    "Thyroid Cancer": "https://www.cdc.gov/cancer/thyroid/index.htm",
    "Gonorrhea": "https://www.cdc.gov/std/gonorrhea/default.htm",
    "Smallpox Virus": "https://www.cdc.gov/smallpox/index.html",
    "HIV Virus": "https://www.cdc.gov/hiv/default.html",
    "Burkholderia Infection": "https://www.cdc.gov/burkholderia/index.html",
    "Respiratory Syncytial Virus": "https://www.cdc.gov/rsv/index.html",
    "Shiga Toxin-Producing E. coli Infection": "https://www.cdc.gov/ecoli/index.html",
    "Epstein-Barr Virus": "https://www.cdc.gov/epstein-barr/index.html",
    "Clostridium Perfringens Infection": "https://www.cdc.gov/clostridium/index.html",
    "Coronary Heart Disease": "https://www.cdc.gov/heartdisease/coronary_ad.htm",
    "Middle East Respiratory Syndrome Coronavirus": "https://www.who.int/emergencies/mers-cov/en/",
    "Tick-Borne Encephalitis Virus": "https://www.cdc.gov/tickborne/encephalitis/index.html",
    "Heart Disease": "https://www.cdc.gov/heartdisease/coronary_ad.htm",
    "Severe Acute Respiratory Syndrome Coronavirus": "https://www.who.int/health-topics/severe-acute-respiratory-syndrome",
    "Crimean-Congo Hemorrhagic Fever Virus": "https://www.cdc.gov/vhf/crimean-congo/index.html",
    "Shigella Bacterium": "https://www.cdc.gov/shigella/index.html",
    "Crimean-Congo Hemorrhagic Fever": "https://www.cdc.gov/vhf/crimean-congo/index.html",
    "Bubonic Plague Virus": "https://www.cdc.gov/plague/index.html",
    "Dengue": "https://www.cdc.gov/dengue/index.html",
    "Lymphoma": "https://www.cdc.gov/cancer/lymphoma/index.htm",
    "Headache": "https://www.mayoclinic.org/diseases-conditions/headaches/symptoms-causes/syc-20353039",
    "Hepatitis C": "https://www.cdc.gov/hepatitis/hcv/index.htm",
    "HIV": "https://www.cdc.gov/hiv/default.html",
    "Hepatitis D": "https://www.cdc.gov/hepatitis/hdv/index.htm",
    "Shigellosis Virus": "https://www.cdc.gov/shigella/index.html",
    "E. Coli Virus": "https://www.cdc.gov/ecoli/index.html",
    "Flu Virus": "https://www.cdc.gov/flu/index.html",
    "Human Papillomavirus (HPV)": "https://www.cdc.gov/std/hpv/stdfact-hpv.htm",
    "Infectious Disease": "https://www.cdc.gov/ncezid/",
    "Adenovirus": "https://www.cdc.gov/adenovirus/index.html",
    "Poliovirus": "https://www.cdc.gov/polio/index.html",
    "Crimean-Congo Hemorrhagic Fever Bunyavirus": "https://www.cdc.gov/vhf/crimean-congo/index.html",
    "Leprosy Bacterium": "https://www.cdc.gov/leprosy/index.html",
    "Herpes Simplex Virus": "https://www.cdc.gov/herpes/index.html",
    "Hepatitis B Virus": "https://www.cdc.gov/hepatitis/hbv/index.htm",
    "Varicella-Zoster Virus Infection": "https://www.cdc.gov/vaccines/vpd/varicella/index.html",
    "Hantavirus": "https://www.cdc.gov/hantavirus/index.html",
    "Pertussis": "https://www.cdc.gov/pertussis/index.html",
    "Marburg Virus Hemorrhagic Fever": "https://www.cdc.gov/vhf/marburg/index.html",
    "Burkholderia Virus": "https://www.cdc.gov/burkholderia/index.html",
    "Cholera Virus": "https://www.cdc.gov/cholera/index.html",
    "Middle East Respiratory Syndrome Virus": "https://www.who.int/emergencies/mers-cov/en/",
    "Zika": "https://www.cdc.gov/zika/index.html",
    "Leishmania": "https://www.cdc.gov/parasites/leishmaniasis/",
    "Japanese Encephalitis Virus Infection": "https://www.cdc.gov/japaneseencephalitis/index.html",
    "Yellow Fever Virus": "https://www.cdc.gov/yellowfever/index.html"
}

def load_data():
    data = pd.read_json('symptom_disease.json')
    return data

def train_model(data):
    X = data.iloc[:, :-2]  
    y = data.iloc[:, -2]  

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    rf_model = RandomForestClassifier()
    rf_model.fit(X, y_encoded)

    return rf_model, label_encoder

def predict_diseases(symptoms, model, label_encoder, feature_columns):
    input_vector = np.zeros(len(feature_columns))

    for symptom in symptoms:
        if symptom in feature_columns:
            symptom_index = feature_columns.index(symptom)
            input_vector[symptom_index] = 1

    probabilities = model.predict_proba([input_vector])[0]
    top_indices = np.argsort(probabilities)[-3:]  # Top 3 possible diseases
    possible_diseases = label_encoder.inverse_transform(top_indices)

    return possible_diseases

def extract_symptoms(user_input, feature_columns):
    cleaned_input = re.sub(r'[^a-zA-Z\s]', '', user_input.lower())
    detected_symptoms = [symptom for symptom in feature_columns if symptom.lower() in cleaned_input]
    return detected_symptoms

def get_category(disease, data):
    category_row = data[data['disease'] == disease]
    if not category_row.empty:
        return category_row['category'].values[0]
    return None

def suggest_nutrition(category):
    nutrition = ""

    if category == 'A':  # Cardiovascular Health
        nutrition = ("Heart-Healthy Fats (avocados, nuts, olive oil), Omega-3s (salmon, flaxseeds), "
                     "Fiber (whole grains, legumes), Limit Sodium")
    elif category == 'B':  # Bone Health
        nutrition = ("Calcium (dairy, leafy greens), Vitamin D (sun, fatty fish), Magnesium (nuts, seeds)")
    elif category == 'C':  # Immune Support
        nutrition = ("Vitamin C (citrus, bell peppers), Zinc (meat, legumes), Vitamin A (carrots, spinach)")
    elif category == 'D':  # Blood Sugar Management
        nutrition = ("Complex Carbs (whole grains, legumes), Fiber (vegetables), Limit Added Sugars")
    elif category == 'E':  # Brain Health
        nutrition = ("Antioxidants (berries, dark chocolate), Healthy Fats (Omega-3), B Vitamins (whole grains)")
    elif category == 'F':  # Digestive Health
        nutrition = ("Fiber (promotes regular bowel movements), Probiotics (yogurt, kefir), Hydration")
    elif category == 'G':  # Weight Management
        nutrition = ("Caloric Balance, Lean Proteins (chicken, fish), Whole Foods (minimize processed foods)")
    elif category == 'H':  # Eye Health
        nutrition = ("Lutein and Zeaxanthin (leafy greens, eggs), Vitamin A (carrots, sweet potatoes)")
    elif category == 'I':  # Skip
        nutrition = ("Data to be uploaded soon")
    
    return nutrition if nutrition else "No specific nutrition advice available for this condition."

import json
import os
import numpy as np  
from datetime import datetime



def log_interaction(user_input, predictions, category, feedback=None):
    if isinstance(predictions, np.ndarray):
        predictions = predictions.tolist()
    
    interaction = {
        "user_input": user_input,
        "predictions": predictions,  
        "category": category,
        "feedback": feedback,
        "timestamp": datetime.now().isoformat()
    }

    json_file_path = 'user_interactions.json'

    if os.path.exists(json_file_path):
        
        if os.stat(json_file_path).st_size > 0:
            try:
                with open(json_file_path, 'r') as file:
                    data = json.load(file)
            except json.JSONDecodeError:
                print("Error: JSON file is malformed. Creating a new file.")
                data = []  
        else:
            data = []  
        data = []  

    data.append(interaction)

    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=4)



def handle_greeting(user_input):
    greetings = ["hello", "hi", "hey", "yo","hola"]
    if any(greeting in user_input.lower() for greeting in greetings):
        return random.choice(["Hello! How can I assist you today?", "Hi there! How can I help you?", "Hey! What can I do for you?", "Yo! How can I help you?","Hola! How can I help you?"])

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
    return disease_links.get(disease_name, "Link will be updated soon.")

def main():
    st.set_page_config(page_title="MAI - MedAI", page_icon="💬", layout="wide")

    st.markdown("""
        <style>
        body {
            background-color: #a1dbe3;
        }
        h1, h2, h3 {
            color: #177538;
        }
        .stButton>button {
            background-color: #00796b;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("Medical Diagnosis Chatbot")
    st.subheader("Describe your symptoms to get a possible diagnosis and nutritional suggestions.")

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
            symptoms = extract_symptoms(user_input, list(data.columns[:-2]))

            if symptoms:
                possible_diseases = predict_diseases(symptoms, rf_model, label_encoder, list(data.columns[:-2]))
                st.write("Based on the symptoms you mentioned, these are some possible conditions:")

                # Create columns for displaying diseases in a grid
                num_diseases = len(possible_diseases)
                cols = st.columns(3)  # Create 3 columns in a row

                for i, disease in enumerate(possible_diseases):
                    category = get_category(disease, data)
                    nutrition = suggest_nutrition(category)
                    link = get_disease_link(disease)

                    # Display disease info in grid
                    with cols[i % 3]:  # Fill the columns sequentially
                        st.markdown(f"### **{disease}**")
                        st.markdown(f"**Category:** {category}")
                        st.markdown(f"**Nutrition Suggestion:** {nutrition}")
                        st.markdown(f"**[More Information]({link})**")

                st.success("These conditions can range from less complex to serious ones. It's always recommended to consult a doctor for an accurate diagnosis.")

                feedback = st.radio("Was this suggestion helpful?", ["Yes", "No"])
                if feedback == "Yes":
                    st.write("Great! We're happy to help!")
                else:
                    st.write("Thank you for your feedback, we'll improve our suggestions.")

                log_interaction(user_input, possible_diseases, feedback)
            else:
                st.error("Sorry, I couldn't detect any known symptoms. Please try again with more details.")

if __name__ == "__main__":
    main()
