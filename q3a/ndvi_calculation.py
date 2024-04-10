# Import required libraries
import rasterio
import numpy as np
import time

# Start timing
start_time = time.time()

# Import bands as separate images; in /project2/macs30123 on Midway2
band4 = rasterio.open('/project2/macs30123/landsat8/LC08_B4.tif') #red
band5 = rasterio.open('/project2/macs30123/landsat8/LC08_B5.tif') #nir

# Convert nir and red objects to float64 arrays
red = band4.read(1).astype('float64')
nir = band5.read(1).astype('float64')

# NDVI calculation
ndvi = (nir - red) / (nir + red)

# End timing
end_time = time.time()

# Calculate and print the execution time
execution_time = end_time - start_time
print(f"NDVI computation without GPU took {execution_time} seconds")

# Optionally, save the NDVI image using matplotlib
import matplotlib.pyplot as plt
plt.imsave('ndvi_image_cpu.png', ndvi)