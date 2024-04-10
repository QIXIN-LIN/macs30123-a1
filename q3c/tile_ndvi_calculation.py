import numpy as np
import rasterio
import time
import matplotlib.pyplot as plt

# Define a function for the NDVI calculation
def compute_ndvi_serial(red, nir):
    # NDVI calculation
    ndvi = (nir - red) / (nir + red)
    return ndvi

# Path to the Landsat bands
red_band_path = '/project2/macs30123/landsat8/LC08_B4.tif'
nir_band_path = '/project2/macs30123/landsat8/LC08_B5.tif'

# Read the bands once outside the loop
with rasterio.open(red_band_path) as red_band, rasterio.open(nir_band_path) as nir_band:
    red_original = red_band.read(1).astype('float64')
    nir_original = nir_band.read(1).astype('float64')

# Define the scale factors to test
scale_factors = [50, 100, 150]

# Loop over the scale factors and compute NDVI for each
for scale_factor in scale_factors:
    # Tile arrays to simulate additional data
    red_tiled = np.tile(red_original, scale_factor)
    nir_tiled = np.tile(nir_original, scale_factor)

    # Time the NDVI computation for the non-GPU code
    start_time = time.time()
    ndvi = compute_ndvi_serial(red_tiled, nir_tiled)
    end_time = time.time()

    # Calculate and print the execution time
    execution_time = end_time - start_time
    print(f"NDVI computation without GPU took {execution_time} seconds for a scale factor of {scale_factor}.")
