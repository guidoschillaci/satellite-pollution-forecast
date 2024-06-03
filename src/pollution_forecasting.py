# src/pollution_forecasting.py

import os
import numpy as np
import pandas as pd
import requests
import ee
import geemap
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


def initialize_earth_engine():
    # Initialize the Earth Engine module.
    ee.Initialize()


def download_pollution_data():
    # Download pollution data from the EEA API.
    url = "https://discomap.eea.europa.eu/map/fme/airqualityexport.htm"  # EEA API endpoint
    params = {
        "sql": "select * from table_name where area='Florence' and pollutant='PM10'",
        "format": "csv"
    }
    response = requests.get(url, params=params)
    with open('florence_pollution_data.csv', 'w') as file:
        file.write(response.text)


def preprocess_pollution_data():
    # Load and preprocess the pollution data.
    df = pd.read_csv('florence_pollution_data.csv', delimiter=';')
    df['DatetimeBegin'] = pd.to_datetime(df['DatetimeBegin'])
    return df


def get_satellite_images(start_date, end_date, location):
    # Get satellite images from Google Earth Engine.
    collection = ee.ImageCollection('COPERNICUS/S2') \
        .filterDate(start_date, end_date) \
        .filterBounds(location)
    return collection


def download_images(collection, output_dir):
    # Download satellite images to the specified directory.
    for i, image in enumerate(collection.getInfo()['features']):
        image_id = image['id']
        img = ee.Image(image_id)
        path = f"{output_dir}/image_{i}.png"
        geemap.ee_export_image(img, filename=path, scale=10)


def preprocess_images(image_dir):
    # Preprocess images for the model.
    images = []
    for filename in os.listdir(image_dir):
        img = cv2.imread(os.path.join(image_dir, filename))
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Resize to 128x128
            images.append(img)
    return np.array(images)


def merge_data(pollution_data, images):
    # Merge pollution data with images based on the date.
    # For simplicity, we'll assume each image corresponds to a date in the pollution data.
    pollution_data['image'] = images[:len(pollution_data)]
    return pollution_data


class PollutionDataset(Dataset):
    def __init__(self, images, targets):
        self.images = images
        self.targets = targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        target = torch.tensor(target, dtype=torch.float32)
        return image, target


class PollutionModel(nn.Module):
    def __init__(self):
        super(PollutionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader)}")


if __name__ == "__main__":
    initialize_earth_engine()
    download_pollution_data()
    data = preprocess_pollution_data()

    florence_coords = [11.2558, 43.7696]
    location = ee.Geometry.Point(florence_coords)
    start_date = '2020-01-01'
    end_date = '2020-12-31'
    os.makedirs('satellite_images', exist_ok=True)
    satellite_images = get_satellite_images(start_date, end_date, location)
    download_images(satellite_images, 'satellite_images')

    images = preprocess_images('satellite_images')
    merged_data = merge_data(data, images)

    X = np.stack(merged_data['image'].values)
    y = merged_data['PM10'].values  # Assuming 'PM10' is the target pollution variable

    dataset = PollutionDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = PollutionModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, dataloader, criterion, optimizer)

    # Save the model
    torch.save(model.state_dict(), 'pollution_model.pth')
    print("Model saved to pollution_model.pth")
