# Road Type Prediction - README

## Overview
The **Road Type Prediction** application is a web-based tool built with **Streamlit** that uses **machine learning** to predict the type of road (e.g., highway, residential) based on a given GPS location (latitude and longitude). The app integrates road network data and GNSS vehicle data, employing a **Random Forest Classifier** to predict the road type at a specific location.

The system uses **geodesic distance calculations** to identify the closest road segments and incorporate relevant features (e.g., speed, curvature) for prediction. It also displays the selected location on an interactive **map** powered by **Folium**.

---

## Features
- **Location Input**: Enter the latitude and longitude of a location to predict the road type.
- **Real-Time Prediction**: The app predicts the road type using a trained **Random Forest** model based on GNSS and road network data.
- **Map Display**: View the entered location on an interactive map.
- **Machine Learning**: Predict road type by evaluating distance to road segments, speed differences, and road curvature.

---

## Technologies Used
- **Streamlit**: A Python framework for building interactive web applications.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computing.
- **scikit-learn**: Machine learning library, specifically for the **Random Forest Classifier**.
- **Geopy**: For geodesic distance calculation between two geographic points.
- **Folium**: For visualizing maps and locations interactively.
- **Streamlit-Folium**: Integration to display **Folium** maps in **Streamlit** apps.

---

## Files Overview

### `app.py`
The main Python script running the Streamlit app, which includes the following functionalities:
- **Load and Prepare Data**: Load and clean the road network and GNSS vehicle data.
- **Model Training**: Train a **Random Forest** classifier on the GNSS and road network data to predict the road type.
- **User Input**: Accept user inputs (latitude and longitude) from the sidebar.
- **Map Visualization**: Show the selected location on a **Folium** map.
- **Prediction**: Display the predicted road type based on the input data.

### `road_data.csv`
CSV file containing road network data with columns such as:
- `road_segment_id`: Unique ID for each road segment.
- `start_latitude`, `start_longitude`: Coordinates of the start of the road segment.
- `road_type`: Type of road (e.g., highway, residential, etc.).
- `curvature`: Curvature of the road, used as a feature in the prediction model.

### `gnss_vehicle_data.csv`
CSV file containing GNSS data for vehicles with columns such as:
- `latitude`, `longitude`: GPS coordinates of the vehicle.
- `speed`: Speed of the vehicle at the given coordinates.

---

## How to Run the Application

### Prerequisites
Ensure you have the following Python libraries installed:
- Streamlit
- Pandas
- NumPy
- scikit-learn
- Geopy
- Folium
- Streamlit-Folium

Install the required libraries using `pip`:
```bash
pip install streamlit pandas numpy scikit-learn geopy folium streamlit-folium
```

### Steps to Run
1. Clone the repository or download the files:
   ```bash
   git clone https://github.com/your-repo/road-type-prediction.git
   cd road-type-prediction
   ```

2. Prepare your data files `road_data.csv` and `gnss_vehicle_data.csv` (ensure they are in the same directory as the script).

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Open the app in your browser (usually accessible at `http://localhost:8501`).

---

## Usage

1. **Input Location**: In the sidebar, enter the latitude and longitude of the location you want to predict the road type for. 
   
2. **Map Display**: After entering the location, the map will display the selected point.

3. **Predict Road Type**: Click the "Predict Road Type" button to predict the road type at the entered coordinates based on the trained model. The app will display the predicted road type (e.g., highway, residential).

---

## Model Explanation

The model is based on a **Random Forest Classifier** that predicts the type of road (e.g., highway, residential) based on the following features:
- **Distance to Road Segments**: The distance between the input location and the closest road segment.
- **Speed Difference**: The difference between the vehicle's speed and the road's speed limit (if available).
- **Curvature**: The curvature of the road (a feature derived from the road type).

### Model Training
The model is trained on the **gnss_vehicle_data.csv** and **road_data.csv** datasets using the **RandomForestClassifier** from **scikit-learn**. The model is evaluated using accuracy metrics.

---

## Example Prediction
Given the location:
- Latitude: 12.9716
- Longitude: 77.5946

The model might predict:
- **Road Type**: Residential

This prediction is based on the nearest road segment and its features (e.g., curvature, distance from the point).

---

## Future Enhancements
- **Speed Data Integration**: Integrate actual vehicle speed data to improve prediction accuracy.
- **Route Prediction**: Extend the model to predict road types along a route, rather than a single point.
- **Real-Time Data**: Integrate real-time GNSS data for dynamic predictions.
- **User Authentication**: Add user login to store and track predictions.

---

## License
This project is licensed under the **MIT License**.

---
=
