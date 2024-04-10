import numpy as np
import pyopencl as cl
import pyopencl.array
import rasterio
import time
import matplotlib.pyplot as plt

# Setup OpenCL context and command queue
platform = cl.get_platforms()[0]  # Select the first platform
device = platform.get_devices()[0]  # Select the first device on the platform
context = cl.Context([device])  # Create a context for the selected device
queue = cl.CommandQueue(context)  # Create a command queue for the selected device

# Import bands as separate images
band4 = rasterio.open('/project2/macs30123/landsat8/LC08_B4.tif')  # red
band5 = rasterio.open('/project2/macs30123/landsat8/LC08_B5.tif')  # nir

# Function to calculate NDVI in parallel using OpenCL
def calculate_ndvi_parallel(red, nir, queue, context):
    # Transfer arrays to device memory
    red_dev = cl.array.to_device(queue, red)
    nir_dev = cl.array.to_device(queue, nir)
    # Allocate memory for the result on the device
    ndvi_dev = cl.array.empty_like(red_dev)
    # NDVI calculation using ElementwiseKernel
    ndvi_kernel = cl.elementwise.ElementwiseKernel(
        context,
        "double *nir, double *red, double *ndvi",
        "ndvi[i] = (nir[i] - red[i]) / (nir[i] + red[i])",
        "ndvi_kernel"
    )
    ndvi_kernel(nir_dev, red_dev, ndvi_dev)
    # Copy result from device to host
    ndvi = ndvi_dev.get()
    return ndvi

# Function to calculate NDVI serially
def calculate_ndvi_serial(red, nir):
    ndvi = (nir - red) / (nir + red)
    return ndvi

def validate_ndvi(ndvi_parallel, ndvi_serial, tolerance=1e-6):
    """
    Validate that the parallel and serial NDVI calculations produce the same result within a specified tolerance.
    """
    diff = np.abs(ndvi_parallel - ndvi_serial)
    max_diff = np.max(diff)
    print(f"Maximum absolute difference between parallel and serial NDVI: {max_diff}")
    return max_diff <= tolerance

# Data sizes to simulate
data_sizes = [50, 100, 150]

for size in data_sizes:
    print(f"Processing equivalent of {size} Landsat scenes")
    
    # Convert nir and red objects to float64 arrays
    red_original = band4.read(1).astype('float64')
    nir_original = band5.read(1).astype('float64')
    
    # Tile arrays to simulate additional data
    red = np.tile(red_original, size)
    nir = np.tile(nir_original, size)
    
    # Time serial calculation
    start_time = time.time()
    ndvi_serial = calculate_ndvi_serial(red, nir)
    serial_time = time.time() - start_time
    print(f"Serial NDVI computation took {serial_time} seconds")
    
    # Time parallel calculation
    start_time = time.time()
    ndvi_parallel = calculate_ndvi_parallel(red, nir, queue, context)
    parallel_time = time.time() - start_time
    print(f"Parallel NDVI computation with GPU took {parallel_time} seconds")

    # Call the validation function after both NDVI calculations have been performed
    if validate_ndvi(ndvi_parallel, ndvi_serial):
        print("Parallel and serial NDVI calculations are equivalent within the specified tolerance.")
    else:
        print("Parallel and serial NDVI calculations differ more than the specified tolerance.")