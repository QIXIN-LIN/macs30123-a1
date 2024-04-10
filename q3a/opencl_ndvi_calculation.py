import numpy as np
import rasterio
import pyopencl as cl
import pyopencl.array
import time

def compute_ndvi(red_band_path, nir_band_path):
    # Read the bands using rasterio
    with rasterio.open(red_band_path) as red_band, rasterio.open(nir_band_path) as nir_band:
        red = red_band.read(1).astype('float32')
        nir = nir_band.read(1).astype('float32')

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
        float ndvi_val = (nir_val - red_val) / (nir_val + red_val);
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

# Time the NDVI computation
start_time = time.time()
ndvi = compute_ndvi(red_band_path, nir_band_path)
end_time = time.time()

print(f"NDVI computation with GPU took {end_time - start_time} seconds.")

# Optionally, save the NDVI image using matplotlib
import matplotlib.pyplot as plt
plt.imsave('ndvi_image_gpu.png', ndvi)