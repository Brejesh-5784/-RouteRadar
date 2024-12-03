import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from geopy.distance import geodesic
import folium
from streamlit_folium import folium_static

# Page configuration
st.set_page_config(page_title="Road Type Prediction", layout="wide")

# Load and prepare data
road_network = pd.read_csv('road_data.csv')
gnss_data = pd.read_csv('gnss_vehicle_data.csv')

# Train model
@st.cache_data
def train_model():
    gnss_data_clean = gnss_data.dropna()
    road_network_clean = road_network.dropna()

    def calculate_distance(lat1, lon1, lat2, lon2):
        return geodesic((lat1, lon1), (lat2, lon2)).meters

    # Calculate the minimum distance from GNSS data to the road segments
    gnss_data_clean['min_distance'] = float('inf')
    for _, row in road_network_clean.iterrows():
        gnss_data_clean[f'distance_to_segment_{row["road_segment_id"]}'] = gnss_data_clean.apply(
            lambda x: calculate_distance(x['latitude'], x['longitude'], row['start_latitude'], row['start_longitude']), axis=1
        )
    
    gnss_data_clean['min_distance'] = gnss_data_clean[[f'distance_to_segment_{i}' for i in road_network_clean['road_segment_id']]].min(axis=1)
    gnss_data_clean['road_type_pred'] = gnss_data_clean.apply(
        lambda x: road_network_clean.loc[
            road_network_clean['road_segment_id'] == road_network_clean['road_segment_id'].iloc[
                np.argmin([x[f'distance_to_segment_{i}'] for i in road_network_clean['road_segment_id']])
            ],
            'road_type'
        ].values[0], axis=1
    )
    
    gnss_data_clean['speed_diff'] = np.abs(gnss_data_clean['speed'] - gnss_data_clean['min_distance'])
    gnss_data_clean['curvature'] = gnss_data_clean['road_type_pred'].map(
        lambda x: road_network_clean.loc[road_network_clean['road_type'] == x, 'curvature'].values[0]
    )
    
    X = gnss_data_clean[['min_distance', 'speed_diff', 'curvature']]
    y = gnss_data_clean['road_type']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return clf

# Train the model
clf = train_model()

# Sidebar for user inputs
st.sidebar.header("Road Type Prediction")

# Map and input for latitude and longitude
st.sidebar.subheader("Enter Location")
latitude = st.sidebar.number_input("Latitude", min_value=-90.0, max_value=90.0, value=12.9716)
longitude = st.sidebar.number_input("Longitude", min_value=-180.0, max_value=180.0, value=77.5946)

# Visual map to show the location selected
st.subheader("Map View")
location_map = folium.Map(location=[latitude, longitude], zoom_start=14)
folium.Marker([latitude, longitude], popup="Selected Location").add_to(location_map)
folium_static(location_map)

# Prediction
st.subheader("Road Type Prediction")
if st.sidebar.button("Predict Road Type"):
    input_data = pd.DataFrame({'latitude': [latitude], 'longitude': [longitude]})
    
    # Calculate distances to road segments
    def calculate_distance(lat1, lon1, lat2, lon2):
        return geodesic((lat1, lon1), (lat2, lon2)).meters
    
    input_data['min_distance'] = float('inf')
    for _, row in road_network.iterrows():
        input_data[f'distance_to_segment_{row["road_segment_id"]}'] = input_data.apply(
            lambda x: calculate_distance(x['latitude'], x['longitude'], row['start_latitude'], row['start_longitude']), axis=1
        )
    
    input_data['min_distance'] = input_data[[f'distance_to_segment_{i}' for i in road_network['road_segment_id']]].min(axis=1)
    input_data['road_type_pred'] = input_data.apply(
        lambda x: road_network.loc[
            road_network['road_segment_id'] == road_network['road_segment_id'].iloc[
                np.argmin([x[f'distance_to_segment_{i}'] for i in road_network['road_segment_id']])
            ],
            'road_type'
        ].values[0], axis=1
    )
    
    input_data['speed_diff'] = 0  # Placeholder as speed is not provided
    input_data['curvature'] = input_data['road_type_pred'].map(
        lambda x: road_network.loc[road_network['road_type'] == x, 'curvature'].values[0]
    )
    
    X_input = input_data[['min_distance', 'speed_diff', 'curvature']]
    
    # Predict the road type
    prediction = clf.predict(X_input)
    
    st.write(f"The predicted road type at the location is: **{prediction[0]}**")
else:
    st.write("Enter the location coordinates in the sidebar and click 'Predict Road Type'.")

