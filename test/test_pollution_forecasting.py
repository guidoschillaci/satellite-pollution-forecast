# tests/test_pollution_forecasting.py

import os
import pytest
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.pollution_forecasting import (
    initialize_earth_engine, download_pollution_data,
    preprocess_pollution_data, get_satellite_images,
    download_images, preprocess_images, merge_data,
    PollutionDataset, PollutionModel, train_model
)

def test_download_pollution_data():
    download_pollution_data()
    assert os.path.exists('florence_pollution_data.csv')

def test_preprocess_pollution_data():
    df = preprocess_pollution_data()
    assert isinstance(df, pd.DataFrame)
    assert 'DatetimeBegin' in df.columns

def test_preprocess_images():
    os.makedirs('satellite_images', exist_ok=True)
    with open('satellite_images/dummy.png', 'w') as f:
        f.write("dummy image content")
    images = preprocess_images('satellite_images')
    assert images.shape[1:] == (128, 128, 3)

def test_merge_data():
    data = preprocess_pollution_data()
    images = preprocess_images('satellite_images')
    merged_data = merge_data(data, images)
    assert 'image' in merged_data.columns

def test_dataset():
    images = np.random.rand(10, 128, 128, 3)
    targets = np.random.rand(10)
    dataset = PollutionDataset(images, targets)
    assert len(dataset) == 10
    image, target = dataset[0]
    assert image.shape == (3, 128, 128)
    assert isinstance(target, torch.Tensor)

def test_model():
    model = PollutionModel()
    images = torch.rand(2, 3, 128, 128)
    outputs = model(images)
    assert outputs.shape == (2, 1)

@pytest.mark.slow
def test_training():
    images = np.random.rand(100, 128, 128, 3)
    targets = np.random.rand(100)
    dataset = PollutionDataset(images, targets)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = PollutionModel()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_model(model, dataloader, criterion, optimizer, epochs=1)
    assert True  # If it reaches here, training ran without error

if __name__ == "__main__":
    pytest.main()
