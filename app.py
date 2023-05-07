import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle
from PIL import Image
import base64

# Call set_page_config() as the first Streamlit command in your script
st.set_page_config(page_title="My App", layout="centered", initial_sidebar_state="expanded")


# Define mappings
COUNTRY_MAP = {'Zambia': 0, 'Ethiopia': 1, 'Nigeria': 2, 'Tanzania': 3, "C�te d'Ivoire": 4, 'Mozambique': 5, 'Others': 6, 'Zimbabwe': 7, 'South Africa': 8, 'Rwanda': 9, 'Haiti': 10, 'Vietnam': 11, 'Uganda': 12}
FULFILL_VIA_MAP = {'From RDC': 0, 'Direct Drop': 1}
SHIPMENT_MODE_MAP = {'Truck': 0, 'Air': 1, 'Air Charter': 2, 'Ocean': 3}
SUB_CLASSIFICATION_MAP = {'Adult': 0, 'Pediatric': 1, 'HIV test': 2, 'HIV test - Ancillary': 3, 'Malaria': 4, 'ACT': 5}
BRAND_MAP = {'Generic': 0, 'Others': 1, 'Determine': 2, 'Uni-Gold': 3}
FIRST_LINE_DESIGNATION_MAP = {'Yes': 0, 'No': 1}



# Define the Streamlit app
def app():
    # Set background color and image
    st.markdown(
        f"""
        <style>
        .reportview-container {{
            background: url(r'C:/Users/Admin/Documents/Shipment_pricing_Project/templatesbackground.jpg') no-repeat center center fixed;
            -webkit-background-size: cover;
            -moz-background-size: cover;
            -o-background-size: cover;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Add CSS styling for the app title
    st.markdown(
        f"""
        <style>
        .title {{
            font-size: 48px !important;
            color: white !important;
            text-shadow: black 0px 2px 4px;
            margin-top: 30px;
            margin-bottom: 50px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Add CSS styling for the form labels and inputs
    st.markdown(
        f"""
        <style>
        label {{
            color: white !important;
            font-size: 24px !important;
            text-shadow: black 0px 2px 4px;
        }}
        input {{
            font-size: 24px !important;
            padding: 10px !important;
            border-radius: 5px !important;
        }}
        select {{
            font-size: 24px !important;
            padding: 10px !important;
            border-radius: 5px !important;
        }}
        button {{
            font-size: 24px !important;
            padding: 10px !important;
            border-radius: 5px !important;
            background-color: #44aaff !important;
            color: white !important;
            text-shadow: black 0px 2px 4px;
            margin-top: 20px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Set the app title
    st.markdown("<h1 class='title'>Shipment Price Prediction App</h1>", unsafe_allow_html=True)





# Collect user input
    pack_price = st.number_input('Pack Price')
    unit_price = st.number_input('Unit Price')
    weight_kg = st.number_input('Weight (kg)')
    line_item_quantity = st.number_input('Line Item Quantity')
    fulfill_via = st.selectbox('Fulfillment Via', list(FULFILL_VIA_MAP.keys()))
    shipment_mode = st.selectbox('Shipment Mode', list(SHIPMENT_MODE_MAP.keys()))
    country = st.selectbox('Country', list(COUNTRY_MAP.keys()))
    brand = st.selectbox('Brand', list(BRAND_MAP.keys()))
    sub_classification = st.selectbox('Sub-Classification', list(SUB_CLASSIFICATION_MAP.keys()))
    first_line_designation = st.selectbox('First Line Designation', list(FIRST_LINE_DESIGNATION_MAP.keys()))

    # Convert categorical variables to numerical format
    fulfill_via = FULFILL_VIA_MAP[fulfill_via]
    shipment_mode = SHIPMENT_MODE_MAP[shipment_mode]
    country = COUNTRY_MAP[country]
    brand = BRAND_MAP[brand]
    sub_classification = SUB_CLASSIFICATION_MAP[sub_classification]
    first_line_designation = FIRST_LINE_DESIGNATION_MAP[first_line_designation]

    if st.button('Predict Shipment Price'):
        # Preprocess the user input and make a prediction
        user_input = pd.DataFrame({'Pack_Price': [pack_price],
                                'Unit_Price': [unit_price],
                                'Weight_Kilograms_Clean': [weight_kg],
                                'Line_Item_Quantity': [line_item_quantity],
                                'Fulfill_Via': [fulfill_via],
                                'Shipment_Mode': [shipment_mode],
                                'Country': [country],
                                'Brand': [brand],
                                'Sub_Classification': [sub_classification],
                                'First_Line_Designation': [first_line_designation]})
        
        # Load preprocessor and model
        with open('C:/Users/Admin/Documents/Shipment_pricing_Project/prediction_files/preprocessed.pkl', 'rb') as f:
            preprocessor = pickle.load(f)

        model = joblib.load('C:/Users/Admin/Documents/Shipment_pricing_Project/prediction_files/model.pkl')
        input_data = preprocessor.transform(user_input) # Apply preprocessor to user input
        prediction = model.predict(input_data.reshape(1,-1))[0]

        # Display the prediction to the user
        st.write(f'The predicted shipment price is {round(prediction, 2)}')
        
if __name__ == '__main__':
    app()