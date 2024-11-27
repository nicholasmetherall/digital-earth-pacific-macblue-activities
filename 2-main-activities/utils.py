from dask.distributed import Client as DaskClient
from odc.stac import load, configure_s3_access  # Correct source for `load`

def load_data(items, bbox):
    """
    Load data into a dataset with specified measurements and configurations.

    Parameters:
    - items: List of STAC items to load.
    - bbox: Bounding box for the region of interest.

    Returns:
    - data: The loaded dataset.
    """
    data = load(
        items,
        measurements=[
            "red", "green", "blue", "nir08", "swir16", "scl", 
            "coastal", "nir09", "cloud", "rededge1", 
            "rededge3", "rededge2", "nir"
        ],
        bbox=bbox,
        chunks={"x": 2048, "y": 2048},
        groupby="solar_day",
    )
    return data

import xarray as xr

def mask_and_scale(data):
    """
    Applies a Sentinel-2 cloud mask and scales the data values.

    Parameters:
    - data (xarray.DataArray): The input data array containing Sentinel-2 data with a 'scl' band (scene classification layer).

    Returns:
    - xarray.DataArray: The masked and scaled data.
    """
    # Mask out clouds and scale values
    # Sentinel-2 cloud mask flags (1: defective, 3: shadow, 9: high confidence cloud, 10: thin cirrus)
    mask_flags = [1, 3, 9, 10]

    # Apply the cloud mask (invert the mask to keep non-cloud values)
    cloud_mask = ~data.scl.isin(mask_flags)
    masked = data.where(cloud_mask)

    # Apply scaling and clip values from 0 to 1
    scaled = (masked.where(masked != 0) * 0.0001).clip(0, 1)
    
    return scaled
