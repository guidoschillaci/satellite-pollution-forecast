
import os
import requests
import datetime
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
import ee
import geemap

# Function to initialize Google Earth Engine
def initialize_earth_engine():
    ee.Initialize()

# Function to filter and select satellite images
def get_satellite_images(start_date, end_date, location, cloud_cover=20):
    collection = ee.ImageCollection('COPERNICUS/S2') \
        .filterDate(start_date, end_date) \
        .filterBounds(location) \
        .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', cloud_cover)
    return collection

# Function to download images
def download_images(image_collection, folder_path, scale=10, crs='EPSG:4326'):
    size = image_collection.size().getInfo()
    image_list = image_collection.toList(size)
    for i in range(size):
        image = ee.Image(image_list.get(i))
        timestamp = datetime.datetime.fromtimestamp(image.get('system:time_start').getInfo() / 1000).strftime('%Y-%m-%d')
        filename = f'{folder_path}/image_{timestamp}.tif'
        geemap.ee_export_image(image, filename=filename, scale=scale, crs=crs)
        print(f'Downloaded {filename}')

# Function to download pollution data
def download_pollution_data():
    base_url = "https://discomap.eea.europa.eu/map/fme/AirQualityExport.htm"
    params = {
        "CountryCode": "IT",  # Country code for Italy
        "CityName": "Florence",  # City name
        "Pollutant": "PM10,PM2.5",  # Pollutants
        "Year_from": "2020",  # Start year
        "Year_to": "2020",  # End year
        "Stationtype": "All",  # Station type
        "Output": "CSV"  # Output format
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        with open('florence_pollution_data.csv', 'wb') as file:
            file.write(response.content)
        print("Pollution data downloaded successfully.")
    else:
        print(f"Failed to download pollution data. Status code: {response.status_code}")

# Function to preprocess the pollution data
def preprocess_pollution_data():
    pollution_data = pd.read_csv('florence_pollution_data.csv')
    pollution_data['DatetimeBegin'] = pd.to_datetime(pollution_data['DatetimeBegin'])
    pollution_data = pollution_data[pollution_data['Pollutant'].isin(['PM10', 'PM2.5'])]
    pollution_data.to_csv('florence_processed_pollution_data.csv', index=False)
    print("Processed pollution data saved successfully.")
    return pollution_data

# Function to preprocess images
def preprocess_image(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize to [0, 1]
    return image

# Function to align pollution data with images
def align_data(image_paths, pollution_data):
    timestamps = [os.path.basename(path).split('_')[1].split('.')[0] for path in image_paths]
    pollution_data['date'] = pd.to_datetime(pollution_data['DatetimeBegin']).dt.date
    aligned_data = []
    for timestamp in timestamps:
        date = datetime.datetime.strptime(timestamp, '%Y-%m-%d').date()
        closest_date = pollution_data.iloc[(pollution_data['date'] - date).abs().argmin()]
        aligned_data.append(closest_date['Concentration'])
    return np.array(aligned_data)

# Main function to run the script
def main():
    # Initialize Google Earth Engine
    initialize_earth_engine()

    # Define the location and date range
    florence_coords = [11.2558, 43.7696]
    location = ee.Geometry.Point(florence_coords)
    start_date = '2020-01-01'
    end_date = '2020-12-31'

    # Create a folder to store the images
    os.makedirs('satellite_images', exist_ok=True)

    # Get and download satellite images
    satellite_images = get_satellite_images(start_date, end_date, location)
    download_images(satellite_images, 'satellite_images')

    # Download and preprocess pollution data
    download_pollution_data()
    pollution_data = preprocess_pollution_data()

    # Load and preprocess all images
    image_paths = glob('satellite_images/*.tif')
    images = np.array([preprocess_image(img_path) for img_path in image_paths])

    # Align the pollution data with the images
    pollution_levels = align_data(image_paths, pollution_data)

    # Display a few samples to verify
    for i in range(5):
        plt.imshow(images[i])
        plt.title(f'Pollution Level: {pollution_levels[i]}')
        plt.show()

if __name__ == '__main__':
    main()
