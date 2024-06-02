# test_pollution_forecasting.py

import unittest
import os
import sys

# Add src directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pollution_forecasting as pf
import pandas as pd


class TestPollutionForecasting(unittest.TestCase):

    def setUp(self):
        # Initialize Earth Engine
        pf.initialize_earth_engine()

    def test_initialize_earth_engine(self):
        # Test Earth Engine initialization
        try:
            pf.initialize_earth_engine()
            initialized = True
        except Exception as e:
            print(e)
            initialized = False
        self.assertTrue(initialized)

    def test_download_pollution_data(self):
        # Test downloading pollution data
        pf.download_pollution_data()
        self.assertTrue(os.path.exists('florence_pollution_data.csv'))

    def test_preprocess_pollution_data(self):
        # Test preprocessing pollution data
        pf.download_pollution_data()
        pollution_data = pf.preprocess_pollution_data()
        self.assertIsInstance(pollution_data, pd.DataFrame)
        self.assertTrue('DatetimeBegin' in pollution_data.columns)

    def test_download_images(self):
        # Test downloading satellite images
        florence_coords = [11.2558, 43.7696]
        location = pf.ee.Geometry.Point(florence_coords)
        start_date = '2020-01-01'
        end_date = '2020-12-31'
        os.makedirs('satellite_images', exist_ok=True)
        satellite_images = pf.get_satellite_images(start_date, end_date, location)
        pf.download_images(satellite_images, 'satellite_images')
        image_files = os.listdir('satellite_images')
        self.assertTrue(len(image_files) > 0)


if __name__ == '__main__':
    unittest.main()
