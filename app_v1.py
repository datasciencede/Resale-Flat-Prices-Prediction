import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import joblib

# Load the decision tree model
decision_tree_model = joblib.load('/Users/mohitr/Desktop/Resale Flat Prices Prediction/decision_tree_model.pkl')

# Define the encoded values and their corresponding categories
town_names = {
    'ANG MO KIO':0,
    'BEDOK':1,
    'BISHAN':2,
    'BUKIT BATOK':3,
    'BUKIT MERAH':4,
    'BUKIT PANJANG':5,
    'BUKIT TIMAH':6,
    'CENTRAL AREA':7,
    'CHOA CHU KANG':8,
    'CLEMENTI':9,
    'GEYLANG':10,
    'HOUGANG':11,
    'JURONG EAST':12,
    'JURONG WEST':13,
    'KALLANG/WHAMPOA':14,
    'LIM CHU KANG':15,
    'MARINE PARADE':16,
    'PASIR RIS':17,
    'PUNGGOL':18,
    'QUEENSTOWN':19,
    'SEMBAWANG':20,
    'SENGKANG':21,
    'SERANGOON':22,
    'TAMPINES':23,
    'TOA PAYOH':24,
    'WOODLANDS':25,
    'YISHUN':26
}
room_types = {
    '1 ROOM': 0,
    '2 ROOM': 1,
    '3 ROOM': 2,
    '4 ROOM': 3,
    '5 ROOM': 4,
    'EXECUTIVE': 5,
    'MULTI-GENERATION': 6
}
storey_range_values = {
    '01 TO 03':0,
    '04 TO 06':1,
    '07 TO 09':2,
    '10 TO 12':3, 
    '13 TO 15':4,
    '16 TO 18':5,
    '19 TO 21':6,
    '22 TO 24':7, 
    '25 TO 27':8,
    '28 TO 30':9,
    '31 TO 33':10,
    '34 TO 36':11,
    '37 TO 39':12,
    '40 TO 42':13, 
    '43 TO 45':14,
    '46 TO 48':15,
    '49 TO 51':16
}
flat_model_types = {
    '2-ROOM': 0,
    '3GEN': 1,
    'ADJOINED FLAT': 2,
    'APARTMENT': 3,
    'DBSS': 4,
    'IMPROVED': 5,
    'IMPROVED-MAISONETTE': 6,
    'MAISONETTE': 7,
    'MODEL A': 8,
    'MODEL A-MAISONETTE': 9,
    'MODEL A2': 10,
    'MULTI GENERATION': 11,
    'NEW GENERATION': 12,
    'PREMIUM APARTMENT': 13,
    'PREMIUM APARTMENT LOFT': 14,
    'PREMIUM MAISONETTE': 15,
    'SIMPLIFIED': 16,
    'STANDARD': 17,
    'TERRACE': 18,
    'TYPE S1': 19,
    'TYPE S2': 20
}

# Define the title and header of the app
st.title(':rainbow[Flat Resale Price Prediction]')

st.subheader(':green[Introduction]')
st.markdown('''The objective of this project is to develop a machine learning model and deploy it as a user-friendly 
            web application that predicts the resale prices of flats in Singapore. 
            This predictive model will be based on historical data of resale flat transactions.''')
st.subheader(':blue[Technologies Used]')
st.markdown(''' Python, Pandas, Numpy, SKlearn, Pickling, Streamlit, Render''')

st.markdown(''':violet[Below UI will provides the user to select the values for each options and get the 
            resale price of flats in Singapore using the Predicted model.]''')

# Define user input fields

# Define user input fields for town and storey_range
town = st.selectbox('Town', options=list(town_names.keys()))
storey_range = st.selectbox('Storey Range', options=list(storey_range_values.keys()))

# Define the range for price_per_sqm
min_price_per_sqm = 160
max_price_per_sqm = 8000

# Define the options for price_per_sqm
price_options = list(range(min_price_per_sqm, max_price_per_sqm + 1, 100))  

# Define the user input field for price_per_sqm using select_slider
price_per_sqm = st.select_slider('Price per Square Meter', options=price_options)


# Define user input fields for flat type and flat model
flat_type = st.selectbox('Flat Type', options=list(room_types.keys()))
flat_model = st.selectbox('Flat Model', options=list(flat_model_types.keys()))

floor_area_sqm = st.number_input('Floor Area (sqm)', min_value=0.0, max_value=500.0, value=100.0)

# Define the range for age_of_property
min_age_of_property = 2
max_age_of_property = 58

# Define the user input field for age_of_property
age_of_property = st.number_input('Age of Property (Years)', min_value=min_age_of_property, 
                                  max_value=max_age_of_property, value=min_age_of_property)


# Define the year range for lease_commence_date
min_year = 1966
max_year = 2022

# Define the user input field for lease_commence_date
lease_commence_date = st.number_input('Lease Commencement Year', min_value=min_year, max_value=max_year, 
                                      value=min_year)

# Define the range for current_remaining_lease
min_current_remaining_lease = 40
max_current_remaining_lease = 97

# Define the user input field for current_remaining_lease
current_remaining_lease = st.number_input('Current Remaining Lease (Years)', min_value=min_current_remaining_lease, 
                                  max_value=max_current_remaining_lease, value=min_current_remaining_lease)


# Define the range for remaining_lease
min_remaining_lease = 40
max_remaining_lease = 99

# Define the user input field for remaining_lease
remaining_lease = st.number_input('Remaining Lease (Years)', min_value=min_remaining_lease, 
                                  max_value=max_remaining_lease, value=min_remaining_lease)

# Define the range for years_holding
min_years_holding = 0
max_years_holding = 60

# Define the user input field for years_holding
years_holding = st.number_input('Years Holding (Years)', min_value=min_years_holding, 
                                  max_value=max_years_holding, value=min_years_holding)


# Define a function to make predictions
def predict_price(town,flat_type,storey_range,floor_area_sqm,flat_model,
                           lease_commence_date,remaining_lease,price_per_sqm,years_holding,
                          current_remaining_lease,age_of_property):
    
    # Get the encoded values for flat type and flat model
    encoded_flat_type = room_types[flat_type]
    encoded_flat_model = flat_model_types[flat_model]
    encoded_town = town_names[town]
    encoded_storey_range = storey_range_values[storey_range]

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'town':[encoded_town],
        'flat_type': [encoded_flat_type],
        'storey_range':[encoded_storey_range],
        'floor_area_sqm': [floor_area_sqm],
        'flat_model':[encoded_flat_model],
        'lease_commence_date': [lease_commence_date], 
        'remaining_lease':[remaining_lease],
        'price_per_sqm': [price_per_sqm],     
        'years_holding':[years_holding],    
        'current_remaining_lease':[current_remaining_lease],
        'age_of_property': [age_of_property],                      
           
    })
    
    # Prediction with decision tree model
    prediction = decision_tree_model.predict(input_data)

    # Return the predicted price
    return prediction

# Get user inputs and make prediction when 'Predict' button is clicked
if st.button('Predict'):    

    prediction = predict_price(town,flat_type,storey_range,floor_area_sqm,flat_model,
                           lease_commence_date,remaining_lease,price_per_sqm,years_holding,
                          current_remaining_lease,age_of_property)
    st.write(':green[Predicted Resale Price:]', prediction)


