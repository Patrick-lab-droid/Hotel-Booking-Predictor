import pickle as pkl
import pandas as pd


# Load pickles
scaler = pkl.load(open("scaler.pkl", "rb"))
ohe = pkl.load(open("onehot.pkl", "rb"))
label_map = pkl.load(open("label_map.pkl", "rb"))
ord_map = pkl.load(open("ordinal_map.pkl", "rb"))
model = pkl.load(open("TunedXG.pkl", "rb"))

sample = pd.DataFrame([
    {
    'no_of_adults': 2,
    'no_of_children': 1,
    'no_of_weekend_nights': 2,
    'no_of_week_nights': 3,
    'type_of_meal_plan': 'Meal Plan 1',
    'required_car_parking_space': 1,
    'room_type_reserved': 'Room_Type 1',
    'lead_time': 134,
    'market_segment_type': 'Online',
    'repeated_guest': 0,
    'no_of_previous_cancellations': 0,
    'no_of_previous_bookings_not_canceled': 1,
    'avg_price_per_room': 220,
    'no_of_special_requests': 2
    },

    {
    'no_of_adults': 1,
    'no_of_children': 0,
    'no_of_weekend_nights': 0,
    'no_of_week_nights': 1,
    'type_of_meal_plan': 'Meal Plan 2',
    'required_car_parking_space': 0,
    'room_type_reserved': 'Room_Type 2',
    'lead_time': 7,
    'market_segment_type': 'Offline',
    'repeated_guest': 1,
    'no_of_previous_cancellations': 2,
    'no_of_previous_bookings_not_canceled': 0,
    'avg_price_per_room': 80.0,
    'no_of_special_requests': 0,
    }])

sample['market_segment_type'] = sample['market_segment_type'].map(ord_map)

ohe_cols = ['type_of_meal_plan', 'room_type_reserved']
encoded = ohe.transform(sample[ohe_cols])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols), index=sample.index)

sample = sample.drop(ohe_cols, axis=1)
sample = pd.concat([sample, encoded_df], axis=1)

scale_cols = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
              'lead_time', 'repeated_guest', 'no_of_previous_cancellations','no_of_previous_bookings_not_canceled','avg_price_per_room', 'no_of_special_requests']
sample[scale_cols] = scaler.transform(sample[scale_cols])

prediction = model.predict(sample)

inverse_encode = {0: 'Not_Canceled', 1: 'Canceled'}
original_predictions = [inverse_encode[p] for p in prediction]

print("Prediction:", original_predictions)