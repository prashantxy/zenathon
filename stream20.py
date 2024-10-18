import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import requests

# Title of the app
st.title('Live Server Location Prediction')

# Sidebar input for user to provide IP address
ip_address = st.sidebar.text_input("Enter Server IP Address", value="192.168.0.1")

# Function to get geolocation from IP address
def get_geolocation(ip_address):
    response = requests.get(f"https://ipinfo.io/{ip_address}/geo")
    if response.status_code == 200:
        data = response.json()
        st.write("API Response:", data)  # Debug: print the entire response
        if 'loc' in data:
            lat, lon = data['loc'].split(',')
            return float(lat), float(lon)
        else:
            st.write(f"Error: 'loc' field not found in API response for IP {ip_address}")
    else:
        st.write(f"Error: Unable to fetch data for IP {ip_address}. Status code: {response.status_code}")
    return None

# Simulated or dummy data for server locations based on IP (replace with actual data)
data = {
    'IP': ['192.168.0.1', '192.168.0.2', '192.168.0.3'],
    'lat': [37.7749, 34.0522, 40.7128],
    'lon': [-122.4194, -118.2437, -74.0060]
}
df = pd.DataFrame(data)

# Convert IP address to a numerical format (simple example for IP handling)
df['IP_num'] = df['IP'].apply(lambda x: int(x.split('.')[-1]))

# Train a simple ML model for latitude and longitude prediction
X = df[['IP_num']]
y_lat = df['lat']
y_lon = df['lon']

# Split data into training and test sets
X_train, X_test, y_train_lat, y_test_lat = train_test_split(X, y_lat, test_size=0.2)
X_train, X_test, y_train_lon, y_test_lon = train_test_split(X, y_lon, test_size=0.2)

# Define the models
model_lat = RandomForestRegressor()
model_lon = RandomForestRegressor()

# Train the models
model_lat.fit(X_train, y_train_lat)
model_lon.fit(X_train, y_train_lon)

# Get the geolocation of the provided IP address (external API)
location = get_geolocation(ip_address)
if location:
    lat, lon = location
    st.write(f"Real-time Location from API: Latitude = {lat}, Longitude = {lon}")
    
    # Show the location on the map
    map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
    st.map(map_data)
else:
    # Predict the location using the trained model if the IP is in our data
    if ip_address in df['IP'].values:
        ip_num = int(ip_address.split('.')[-1])
        predicted_lat = model_lat.predict([[ip_num]])[0]
        predicted_lon = model_lon.predict([[ip_num]])[0]

        st.write(f"Predicted Location: Latitude = {predicted_lat}, Longitude = {predicted_lon}")

        # Visualize the predicted location on the map
        map_data = pd.DataFrame({'lat': [predicted_lat], 'lon': [predicted_lon]})
        st.map(map_data)
    else:
        st.write("IP address not found in database or API response.")
