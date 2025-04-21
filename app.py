import streamlit as st
import pickle as pkl
import pandas as pd
import numpy as np

scaler = pkl.load(open("scaler.pkl", "rb"))
ohe = pkl.load(open("onehot.pkl", "rb"))
label_map = pkl.load(open("label_map.pkl", "rb"))
model = pkl.load(open("TunedXG.pkl", "rb"))

def main():
    st.title('Hotel Booking Prediction')
    
    no_of_adults = st.slider("Number of Adults", min_value=1, max_value=4, value=1)
    no_of_children = st.slider("Number of Children", min_value=0, max_value=10, value=1)
    no_of_weekend_nights = st.slider("Number of Weekend Nights", min_value=0, max_value=7, value=1)
    no_of_week_nights = st.slider("Number of Week Nights", min_value=0, max_value=17, value=1)
    type_of_meal_plan = st.selectbox("Type of Meal Plan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3"])
    required_car_parking_space = st.selectbox("Required Car Parking Space", [('Not Required', 0), ('Required', 1)])
    room_type_reserved = st.selectbox("Room Type Reserved", ["Room_Type 1", "Room_Type 2", "Room_Type 3"])
    lead_time = st.slider("Lead Time", min_value=0, max_value=443, value=1)
    market_segment_type = st.selectbox("Market Segment Type", ["Complementary", "Offline", "Online", "Aviation", 'Corporate'])
    repeated_guest = st.selectbox("Repeated Guest", [('No', 0), ('Yes', 1)])
    no_of_previous_cancellations = st.slider("Number of Previous Cancellations", min_value=0, max_value=13, value=1)
    no_of_previous_bookings_not_canceled = st.slider("Number of Previous Bookings Not Canceled", min_value=0, max_value=58, value=1)
    avg_price_per_room = st.slider("Average Price Per Room", min_value=0, max_value=540, value=1)
    no_of_special_requests = st.slider("Number of Special Requests", min_value=0, max_value=5, value=1)
    
    if st.button('Make Prediction'):
        features = [no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights, type_of_meal_plan, 
                    required_car_parking_space, room_type_reserved, lead_time, market_segment_type, repeated_guest, 
                    no_of_previous_cancellations, no_of_previous_bookings_not_canceled, avg_price_per_room, no_of_special_requests]
        result = make_prediction(features)
        inverse_encode = {0: 'Not_Canceled', 1: 'Canceled'}
        original_result = [inverse_encode[p] for p in result]
        st.success(f'The prediction is: {original_result}')

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()
