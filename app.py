import streamlit as st
import numpy as np
import pickle

# Define the mappings
fulfill_via_mapping = {'From RDC': 0, 'Direct Drop': 1}
shipment_mode_mapping = {'Truck': 0, 'Air': 1, 'Ocean': 2, 'Air Charter': 3}
product_group_mapping = {'ARV': 0, 'HRDT': 1, 'ACT': 2, 'ANTM': 3, 'MRDT': 4}
sub_classification_mapping = {'Pediatric': 0, 'Adult': 1, 'HIV test': 2, 'HIV test - Ancillary': 3, 'ACT': 4, 'Malaria': 5}

# Load the pickled model
with open('C:/Users/Admin/Documents/Shipment_pricing_Project/prediction_files/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define a function to make predictions
def predict_price(features):
    # Convert the inputs to a numpy array
    inputs = np.array(features).reshape(1, -1)
    # Make the prediction using the loaded model
    prediction = model.predict(inputs)[0]
    return prediction

# Create the Streamlit app
def app():
    st.title("Categorical and Numerical Inputs")

    # Add the dropdowns for the categorical variables
    fulfill_via = st.selectbox("Fulfill_Via", list(fulfill_via_mapping.keys()), key="fulfill_via")
    shipment_mode = st.selectbox("Shipment_Mode", list(shipment_mode_mapping.keys()), key="shipment_mode")
    product_group = st.selectbox("Product_Group", list(product_group_mapping.keys()), key="product_group")
    sub_classification = st.selectbox("Sub_Classification", list(sub_classification_mapping.keys()), key="sub_classification")

    # Add the numeric input fields
    unit_of_measure = st.number_input("Unit_of_Measure_(Per_Pack)", value=0.0, step=0.1)
    line_item_quantity = st.number_input("Line_Item_Quantity", value=0.0, step=0.1)
    pack_price = st.number_input("Pack_Price", value=0.0, step=0.1)
    unit_price = st.number_input("Unit_Price", value=0.0, step=0.1)
    weight_kg = st.number_input("Weight_Kilograms_Clean", value=0.0, step=0.1)

    # Map the selected values to their numerical representations
    fulfill_via_encoded = fulfill_via_mapping[fulfill_via]
    shipment_mode_encoded = shipment_mode_mapping[shipment_mode]
    product_group_encoded = product_group_mapping[product_group]
    sub_classification_encoded = sub_classification_mapping[sub_classification]

    # Display the encoded values and numeric inputs
    st.write("Encoded values:")
    st.write("Fulfill_Via:", fulfill_via_encoded)
    st.write("Shipment_Mode:", shipment_mode_encoded)
    st.write("Product_Group:", product_group_encoded)
    st.write("Sub_Classification:", sub_classification_encoded)
    st.write("Numeric inputs:")
    st.write("Unit_of_Measure_(Per_Pack):", unit_of_measure)
    st.write("Line_Item_Quantity:", line_item_quantity)
    st.write("Pack_Price:", pack_price)
    st.write("Unit_Price:", unit_price)
    st.write("Weight_Kilograms_Clean:", weight_kg)

    # Combine the encoded values and numeric inputs into a feature vector
    features = [fulfill_via_encoded, shipment_mode_encoded, product_group_encoded,
                sub_classification_encoded, unit_of_measure, line_item_quantity,
                pack_price, unit_price, weight_kg]

    # Add a button to make predictions using the loaded model
    if st.button("Predict"):
        # Call the prediction function
        prediction = predict_price(features)
        # Display the predicted price
        st.write("The predicted price is", prediction)


if __name__ == "__main__":
    app()
