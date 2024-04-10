import numpy as np
import rasterio
import pyopencl as cl
import pyopencl.array
import time
import matplotlib.pyplot as plt

def compute_ndvi(red, nir):
    # Convert input data to float32 arrays if not already
    red = np.asarray(red, dtype=np.float32)
    nir = np.asarray(nir, dtype=np.float32)
    
    # Set up PyOpenCL context and queue
    context = cl.create_some_context()
    queue = cl.CommandQueue(context)

    # Transfer arrays to the GPU
    mf = cl.mem_flags
    red_cl = cl.array.to_device(queue, red)
    nir_cl = cl.array.to_device(queue, nir)

    # Prepare output array
    ndvi_cl = cl.array.empty_like(red_cl)

    # NDVI computation kernel
    kernel = """
    __kernel void compute_ndvi(__global const float *red, __global const float *nir, __global float *ndvi) {
        int i = get_global_id(0);
        float red_val = red[i];
        float nir_val = nir[i];
        float ndvi_val = (nir_val + red_val) == 0 ? 0 : (nir_val - red_val) / (nir_val + red_val);
        ndvi[i] = ndvi_val;
    }
    """

    # Build and execute the kernel
    prg = cl.Program(context, kernel).build()
    prg.compute_ndvi(queue, red.shape, None, red_cl.data, nir_cl.data, ndvi_cl.data)
    queue.finish()

    # Read back the result
    ndvi = ndvi_cl.get()
    return ndvi

# Path to the Landsat bands
red_band_path = '/project2/macs30123/landsat8/LC08_B4.tif'
nir_band_path = '/project2/macs30123/landsat8/LC08_B5.tif'

# Read the bands once outside the loop
with rasterio.open(red_band_path) as red_band, rasterio.open(nir_band_path) as nir_band:
    red_original = red_band.read(1).astype('float32')
    nir_original = nir_band.read(1).astype('float32')

# Define the scale factors to test
scale_factors = [50, 100, 150]

# Loop over the scale factors and compute NDVI for each
for scale_factor in scale_factors:
    # Tile arrays to simulate additional data
    red_tiled = np.tile(red_original, scale_factor)
    nir_tiled = np.tile(nir_original, scale_factor)

    # Time the NDVI computation with GPU
    start_time = time.time()
    ndvi = compute_ndvi(red_tiled, nir_tiled)
    end_time = time.time()

    # Calculate and print the execution time
    execution_time = end_time - start_time
    print(f"NDVI computation with GPU took {execution_time} seconds for a scale factor of {scale_factor}.")
