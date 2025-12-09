import streamlit as st
import numpy as np
import pandas as pd
import joblib
import Feature_engineering as fe

df = pd.read_csv("Cleaned_database.csv")

st.title("Australian Car Market")
st.markdown(
    """
    ---
 
    """
)
st.text('Complete this form with the specifications about the car you like to sell')

#Creating the form
Brand = st.selectbox('Brand 👨🏻‍🔧',sorted(df['Brand'].unique())) #Brand input

Model_array = np.sort(df[df['Brand'] == Brand]['Model'].unique())
Model= st.selectbox('Model 🛠️',sorted(Model_array)) #Model input

Variant_array = 'Select all variants'
Variant_array = np.append(Variant_array,np.sort(df[df['Model'] == Model]['Variant'].unique()))
Variant= st.selectbox('Variant 🔧',Variant_array) #Variant input

Gearbox = st.radio('Select Gearbox: ',['Automatic','Manual']) #Gearbox input

Year = st.number_input('Year :',2000,2025)

Kilometers = st.number_input('Kilometers :',0,350000)

clicked = st.button("Submit",type = "primary")


st.markdown(
    """
    ---
 
    """
)

#Load the model
loaded_model = joblib.load('best_gbr_model.joblib')

if clicked:
    input_vector = [[Year,Kilometers,Gearbox,Brand,Model,Variant]]
    
    input_df = pd.DataFrame(input_vector,columns=['Year','Kilometers','Gearbox','Brand','Model','Variant'])
    transformed_input_vector = fe.preprocessor_final.transform(input_df)
    y_pred_GBR_output = loaded_model.predict(transformed_input_vector)
    pred_price = y_pred_GBR_output[0]
    st.subheader(f'The predicted price of this vehicle is **${pred_price:,.0f}** .') #Showing predicted price
    