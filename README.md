# OmicsIO - Spatial Omics Data Processing

This module provides classes and functions to handle Visium spatial transcriptomics data. The main classes are `SpotLocations`, `Slide`, `PatchDataset`. The module also includes utility functions to load raw data from a folder and preprocess it.

If you have any questions, feel free to reach out to me on the EDIT Slack!

## Quickstart

Installation:
```
pip install git+https://github.com/myfatemi04/omicsio/
```

Example code to load all expression data, associated molecule names, and locations on the image into a single `Slide` object:
```python3
from omicsio.datasets import load_slide_from_folder

# This might take a while
# Returns an omicsio.datasets.Slide object
slide = load_slide_from_folder(<path to whole slide image corresponding to this slide>, "/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Interns_2023/data/metals_data/colon/SP19_31508_A3/ST_data")

# Because it takes a while to run the above code, you can save it with this, and then load the Slide object with Pickle later
slide.save("saved_slide.pkl")
```

The slide object contains attributes `spot_counts`, `spot_locations`, and `genes`.
For example, `slide.spot_counts[0]` contains the gene signatures at spot `0`
And you can find that spot `0` has location (`spot_locations.image_x[0]`, `spot_locations.image_y[0]`) in the image, and that the expression for a given gene (let’s say `"EPCAM"` for example) would be at `slide.spot_counts[0, slide.genes.index("EPCAM")]`.

## API Description

### SpotLocations

The `SpotLocations` class is a data container that holds information about the location of spots in an image, including their x and y pixel locations, row and column in the hexagonal array, and spot diameters in pixels. It provides methods for scaling and selecting subsets of spot locations.

#### Attributes

- `image_x`: x pixel locations of capture spots (torch.Tensor)
- `image_y`: y pixel locations of capture spots (torch.Tensor)
- `row`: row of capture spots in hexagonal array (torch.Tensor)
- `col`: column of capture spots in hexagonal array (torch.Tensor)
- `dia`: spot diameters in pixels (torch.Tensor, Optional)

#### Methods

- `__mul__(self, scale_factor: float)`: Scale spot locations, image_x, image_y, and dia by a factor.
- `__div__(self, dividend: float)`: Divide spot locations, image_x, image_y,and dia by a factor.
- `select_subset(self, mask)`: Select a subset of spot locations based on a boolean mask.
- `__len__(self)`: Get the number of spots in the `SpotLocations` instance.

### Slide

The `Slide` class represents a single slide from Visium spatial transcriptomics data. It stores the image path, spot locations, spot counts, and gene information. The class provides various methods for manipulating and rendering the slide.

#### Attributes

- `image_path`: Path to the slide image file (str)
- `spot_locations`: SpotLocations instance containing spot location information
- `spot_counts`: Spot count values for each gene (torch.Tensor)
- `genes`: List of gene names (list)

#### Methods

- `get_quadrant_masks(self)`: Get boolean masks representing the four quadrants of the slide.
- `create_quadrants(self)`: Split the slide into four quadrants and their complements.
- `image_region(self, x, y, w, h)`: Load a region of the image with given coordinates and dimensions.
- `log1p(self)`: Apply log1p transformation to the spot counts.
- `binary(self)`: Binarize the spot counts based on their median values.
- `select_genes(self, genes, suppress_errors=False)`: Select a subset of genes from the slide.
- `render(self, downsample, spot_counts, spot_size, cmap=matplotlib.colormaps['inferno'])`: Render the slide image with spots colored based on their counts.

### PatchDataset

The `PatchDataset` class is a PyTorch Dataset that provides patches from a slide, along with their spot counts.

#### Attributes

- `slide`: Slide instance (Slide)
- `patch_size`: Size of the patches to be extracted (int)
- `patch_transform`: Transformation applied to the extracted patches (callable, Optional)
- `device`: Device to which the data will be sent (str)
- `magnify`: Magnification factor for the patches (int)

#### Methods

- `__getitem__(self, index: int)`: Get a patch and its spot count for a given index.
- `__len__(self)`: Get the number of patches in the dataset.

### Utility Functions

- `load_spot_locations_json(src: str)`: Load spot locations from a JSON file.
- `load_spot_locations_csv(src: str)`: Load spot locations from a CSV file.
- `load_compressed_tsv(path)`: Load data from a compressed TSV file.
- `load_counts(matrix_path)`: Load spot counts from a matrix file.
- `load_slide_from_folder(main_folder: str, image_path: str, spot_image_scaling: float = 1.0)`: Load a slide from a folder containing Visium data.
- `load_single_cell_from_folder(main_folder: str)`: Loads single-cell data. This is stored as a `Slide`, without the corresponding `spot_locations` or `image` attributes.
