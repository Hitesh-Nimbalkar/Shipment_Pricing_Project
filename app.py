import streamlit as st
import pickle
import numpy as np

# Load the pickled model
with open('C:/Users/Admin/Documents/Shipment_pricing_Project/prediction_files/model.pkl', 'rb') as f:
    model = pickle.load(f)



# Add your Streamlit app code below

# Define a function to make predictions
def predict_price(features):
    # Convert the inputs to a numpy array
    inputs = np.array(features).reshape(1, -1)
    # Make the prediction using the loaded model
    prediction = model.predict(inputs)[0]
    return prediction

# Create a Streamlit app function
def run():
    # Set page title
    st.set_page_config(page_title="Shipment Pricing Prediction")

    # Display page header
    st.title("Shipment Pricing Prediction")

    # Add input fields for features


    # Define the CSS styles for the number input buttons
    number_input_style = """
        background-color: grey;
        border: 2px solid #ccc;
        border-radius: 4px;
        color: #333;
        font-size: 16px;
        padding: 8px 12px;
        width: 100%;
    """

# Create the number input buttons and apply the CSS styles
    unit_of_measure = st.number_input('Unit of Measure (Per Pack)', value=0.0, step=1.0, format="%.2f")
    line_item_quantity = st.number_input('Line Item Quantity', value=0.0, step=1.0, format="%.2f")
    pack_price = st.number_input('Pack Price', value=0.0, step=1.0, format="%.2f")
    unit_price = st.number_input('Unit Price', value=0.0, step=1.0, format="%.2f")
    weight_kilograms = st.number_input('Weight Kilograms', value=0.0, step=1.0, format="%.2f")


    shipment_modes = ['Air', 'Road', 'Sea']
    shipment_mode = st.selectbox('Shipment Mode', shipment_modes)

    manufacturing_sites = ['Aurobindo Unit III, India', 
                'Mylan (formerly Matrix) Nashik',
                'Hetero Unit III Hyderabad IN', 
                'Cipla, Goa, India',
                'Strides, Bangalore, India.', 
                'Alere Medical Co., Ltd.',
                'Trinity Biotech, Plc', 
                'ABBVIE Ludwigshafen Germany',
                'Inverness Japan',
                'Others', 
                'ABBVIE (Abbott) Logis. UK',
                'Chembio Diagnostics Sys. Inc.', 
                'Standard Diagnostics, Korea',
                'Aurobindo Unit VII, IN', 
                'Aspen-OSD, Port Elizabeth, SA',
                'MSD, Haarlem, NL', 
                'KHB Test Kit Facility, Shanghai China',
                'Micro labs, Verna, Goa, India', 
                'Cipla, Kurkumbh, India',
                'Emcure Plot No.P-2, I.T-B.T. Park, Phase II, MIDC, Hinjwadi, Pune, India',
                'BMS Meymac, France', 
                'Ranbaxy, Paonta Shahib, India',
                'Hetero, Jadcherla, unit 5, IN', 
                'Bio-Rad Laboratories' ,
                'ABBVIE GmbH & Co.KG Wiesbaden', 
                'Cipla, Patalganga, India',
                'Pacific Biotech, Thailand', 
                'Roche Basel',
                'Gilead(Nycomed) Oranienburg DE', 
                'Mylan,  H-12 & H-13, India']

    manufacturing_site = st.selectbox('Manufacturing Site', manufacturing_sites)

    countries = ["Côte d'Ivoire",'Zambia','Others','Nigeria','Tanzania','Mozambique','Zimbabwe','South Africa','Rwanda','Haiti','Vietnam','Uganda','Congo, DRC','Ghana','Ethiopia','Kenya','South Sudan','Sudan','Guyana','Cameroon','Dominican Republic','Burundi','Namibia','Botswana']
    country = st.selectbox('Country', countries)

    dosage_forms = ["Tablet","Oral", "Capsule", "Test kit", "Injection"]
    dosage_form = st.selectbox('Dosage Form', dosage_forms)

    first_line_designations = ['Yes', 'No']
    first_line_designation = st.selectbox('First Line Designation', first_line_designations)

    fulfill_vias = ['From RDC', 'Direct Drop']
    fulfill_via = st.selectbox('Fulfill Via', fulfill_vias)

    sub_classifications = ['Adult',
               'Pediatric',
                'HIV test',
                'HIV test - Ancillary',
               'Malaria',
               'ACT']
    sub_classification = st.selectbox('Sub Classification', sub_classifications)

    brands = ['Others',
                                                                        'Generic',
                                                                        'Determine',
                                                                        'Uni-Gold',
                                                                        'Stat-Pak',
                                                                        'Aluvia',
                                                                        'Bioline',
                                                                        'Kaletra',
                                                                        'Norvir',
                                                                        'Colloidal Gold',
                                                                        'Truvada']
    brand = st.selectbox('Brand', brands)

    # Map categorical features to encoded values
    shipment_mode_encoded = shipment_modes.index(shipment_mode)
    manufacturing_site_encoded = manufacturing_sites.index(manufacturing_site)
    country_encoded = countries.index(country)
    dosage_form_encoded = dosage_forms.index(dosage_form)
    first_line_designation_encoded = first_line_designations.index(first_line_designation)
    fulfill_via_encoded = fulfill_vias.index(fulfill_via)
    sub_classification_encoded = sub_classifications.index(sub_classification)
    brand_encoded = brands.index(brand)

    


    

    # Create a button to trigger the prediction
    if st.button('Predict'):
        # Combine the input features into a list

# Combine the input features into a list
        inputs = [unit_of_measure, line_item_quantity,
                pack_price, unit_price, weight_kilograms,
                shipment_mode_encoded, manufacturing_site_encoded, country_encoded,
                dosage_form_encoded, first_line_designation_encoded, fulfill_via_encoded,
                sub_classification_encoded,brand_encoded]


        # Make a prediction using the loaded model
        prediction = predict_price(inputs)


        # Display the prediction
        st.success(f'The estimated shipment price is {prediction:.2f} USD.')

# Run the Streamlit app
if __name__ == '__main__':
    run()
