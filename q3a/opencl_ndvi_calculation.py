import numpy as np
import pyopencl as cl
import pyopencl.array
import rasterio
import time
import matplotlib.pyplot as plt

# Start timing
start_time = time.time()

# Setup OpenCL context and command queue
platform = cl.get_platforms()[0]  # Select the first platform
device = platform.get_devices()[0]  # Select the first device on the platform
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Import bands as separate images
band4 = rasterio.open('/project2/macs30123/landsat8/LC08_B4.tif')  # red
band5 = rasterio.open('/project2/macs30123/landsat8/LC08_B5.tif')  # nir

# Convert nir and red objects to float64 arrays
red = band4.read(1).astype('float64')
nir = band5.read(1).astype('float64')

# Transfer arrays to device memory
red_dev = cl.array.to_device(queue, red)
nir_dev = cl.array.to_device(queue, nir)

# NDVI calculation using ElementwiseKernel
ndvi_kernel = cl.elementwise.ElementwiseKernel(
    context,
    "double *nir, double *red, double *ndvi",
    "ndvi[i] = (nir[i] - red[i]) / (nir[i] + red[i])",
    "ndvi_kernel"
)

# Allocate memory for the result
ndvi_dev = cl.array.empty_like(red_dev)

# Execute the kernel
ndvi_kernel(nir_dev, red_dev, ndvi_dev)

# Copy result from device to host
ndvi = ndvi_dev.get()

# End timing
end_time = time.time()

# Calculate and print the execution time
execution_time = end_time - start_time
print(f"NDVI computation with GPU took {execution_time} seconds")

# Save the NDVI image using matplotlib
plt.imsave('ndvi_image_gpu.png', ndvi)
