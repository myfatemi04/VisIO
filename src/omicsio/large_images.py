from typing import Tuple
import PIL.Image
from .datasets import SpotLocations

PIL.Image.MAX_IMAGE_PIXELS = 11140240175

def load_image(path: str):
    if path.endswith('.svs'):
        import tifffile

        image = PIL.Image.fromarray(tifffile.imread(path)[:, :, :3])
    else:
        image = PIL.Image.open(path)

    return image

def crop_large_image_for_slide(image: PIL.Image.Image, spot_locations: SpotLocations, padding: int = 1024) -> Tuple[PIL.Image.Image, SpotLocations]:
    # Crop the image to the area of the tissue
    xmin, xmax, ymin, ymax = spot_locations.image_x.min(), spot_locations.image_x.max(), spot_locations.image_y.min(), spot_locations.image_y.max()
    xmin = max(int(xmin) - padding, 0)
    xmax = min(int(xmax) + padding, image.width)
    ymin = max(int(ymin) - padding, 0)
    ymax = min(int(ymax) + padding, image.height)

    width = xmax - xmin
    height = ymax - ymin

    print('Cropping image to', xmin, xmax, ymin, ymax, 'and adding', padding, 'pixels of padding')
    print('New image size will be', width, 'x', height)
    
    result_image = image.crop((xmin, ymin, xmax, ymax))

    # Adjust so (xmin, ymin) is at (0, 0)
    cropped_image_x = spot_locations.image_x - xmin
    cropped_image_y = spot_locations.image_y - ymin

    result_spot_locations = SpotLocations(
        image_x=cropped_image_x,
        image_y=cropped_image_y,
        row=spot_locations.row,
        col=spot_locations.col,
        dia=spot_locations.dia,
    )

    return result_image, result_spot_locations

def downsample_image(image: PIL.Image.Image, spot_locations: SpotLocations, downsample: int) -> Tuple[PIL.Image.Image, SpotLocations]:
    import numpy as np
    
    result_image = PIL.Image.fromarray(np.array(image[::downsample, ::downsample, :]))
    
    downsampled_image_x = spot_locations.image_x // downsample
    downsampled_image_y = spot_locations.image_y // downsample

    result_spot_locations = SpotLocations(
        image_x=downsampled_image_x,
        image_y=downsampled_image_y,
        row=spot_locations.row,
        col=spot_locations.col,
        dia=spot_locations.dia,
    )
    
    return result_image, result_spot_locations
