import streamlit as st
import pickle as pkl
import pandas as pd
import numpy as np

scaler = pkl.load(open("scaler.pkl", "rb"))
ohe = pkl.load(open("onehot.pkl", "rb"))
label_map = pkl.load(open("label_map.pkl", "rb"))
ord_map = pkl.load(open("ordinal_map.pkl", "rb"))
model = pkl.load(open("TunedXG.pkl", "rb"))

def main():
    st.title('Hotel Booking Prediction')
    
    no_of_adults = st.slider("Number of Adults", min_value=0, max_value=4, value=1)
    no_of_children = st.slider("Number of Children", min_value=0, max_value=10, value=1)
    no_of_weekend_nights = st.slider("Number of Weekend Nights", min_value=0, max_value=7, value=1)
    no_of_week_nights = st.slider("Number of Week Nights", min_value=0, max_value=17, value=1)
    type_of_meal_plan = st.selectbox("Type of Meal Plan", ["Not Selected","Meal Plan 1", "Meal Plan 2", "Meal Plan 3"])
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
        input_data = pd.DataFrame([{
            'no_of_adults': no_of_adults,
            'no_of_children': no_of_children,
            'no_of_weekend_nights': no_of_weekend_nights,
            'no_of_week_nights': no_of_week_nights,
            'type_of_meal_plan': type_of_meal_plan,
            'required_car_parking_space': 0 if required_car_parking_space == 'Not Required' else 1,
            'room_type_reserved': room_type_reserved,
            'lead_time': lead_time,
            'market_segment_type': market_segment_type,
            'repeated_guest': 0 if repeated_guest == 'No' else 1,
            'no_of_previous_cancellations': no_of_previous_cancellations,
            'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
            'avg_price_per_room': avg_price_per_room,
            'no_of_special_requests': no_of_special_requests
        }])

        input_data['market_segment_type'] = input_data['market_segment_type'].map(ord_map)

        ohe_cols = ['type_of_meal_plan', 'room_type_reserved']
        encode = ohe.transform(input_data[ohe_cols])
        ohe_df = pd.DataFrame(encode, columns=ohe.get_feature_names_out(ohe_cols))
        
        input_data = input_data.drop(ohe_cols, axis = 1)
        input_data = pd.concat([input_data, ohe_df], axis=1)

        scale_cols = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
              'lead_time', 'repeated_guest', 'no_of_previous_cancellations','no_of_previous_bookings_not_canceled','avg_price_per_room', 'no_of_special_requests']
        input_data[scale_cols] = scaler.transform(input_data[scale_cols])

        prediction = model.predict(input_data)[0]
        inverse_encode = {0: 'Not Canceled', 1: 'Canceled'}
        original_result = inverse_encode[prediction]
        st.success(f'The booking is predicted to be: {original_result}')
        
def show_example_table():
    data = {
        'no_of_adults': [2, 3],
        'no_of_children': [3, 2],
        'no_of_weekend_nights': [4, 2],
        'no_of_week_nights': [5, 10],
        'type_of_meal_plan': ['Meal Plan 3', 'Meal Plan 2'],
        'required_car_parking_space': ['Required 1', 'Required 0'],
        'room_type_reserved': ['Room_Type 2', 'Room_Type 3'],
        'lead_time': [87, 223],
        'market_segment_type': ['Corporate', 'Online'],
        'repeated_guest': ["No, 0", "No, 0"],
        'no_of_previous_cancellations': [4, 6],
        'no_of_previous_bookings_not_canceled': [0, 7],
        'avg_price_per_room': [325, 180],
        'no_of_special_requests': [2, 1],
        'Prediction': ['Not_Canceled', 'Canceled']
    }

    df = pd.DataFrame(data)
    st.subheader("Example Prediction Cases")
    st.dataframe(df)

if __name__ == '__main__':
    main()
    show_example_table()
