# RSICD Dataset Example

## Overview

This directory contains a **sample subset** of the **RSICD (Remote Sensing Image Captioning Dataset)**, which is a large-scale dataset of remote sensing images paired with natural language descriptions. This example dataset is perfect for understanding the structure and format of the complete dataset before downloading the full version.

## About RSICD

The Remote Sensing Image Captioning Dataset (RSICD) is a comprehensive collection of aerial and satellite imagery with detailed textual captions. It's designed for training and evaluating models that can understand and describe remote sensing images, combining computer vision with natural language processing.

### Key Characteristics:
- **Image Type**: Aerial/Remote sensing photographs
- **Content**: Various geographic features and man-made structures
- **Captions**: Detailed English descriptions of what's visible in each image
- **Use Case**: Perfect for CNN+LSTM and image captioning models

## Dataset Structure

```
dataset_example/
├── images/              # Contains all remote sensing images
│   └── *.jpg           # Image files organized by scene type
├── captions.json       # JSON file mapping images to their captions
└── readme.md           # This file
```

## Contents of This Example

### Image Categories

This sample includes remote sensing images from **10 different scene types**:

1. **Pond** - Water bodies with surrounding vegetation and structures
2. **Railway Station** - Transit infrastructure with buildings
3. **Resort** - Beachfront and recreational facilities
4. **River** - Waterways with agricultural and natural areas
5. **Sparse Residential** - Low-density housing areas
6. **Square** - Urban plazas and public spaces
7. **Stadium** - Sports facilities and athletic complexes
8. **Storage Tanks** - Industrial infrastructure
9. **Viaduct** - Bridge and elevated infrastructure
10. **Storehouse** - Warehouse and storage facilities

### Sample Data

- **Total Images**: 40 images in this example
- **Captions Format**: JSON key-value pairs (filename → description)
- **Example**:
  ```json
  {
    "pond_229.jpg": "Many buildings and green trees surround a nearly rectangular pond.",
    "railway_station_19.jpg": "There is a railway station at a T-shaped junction, with white awnings over the tracks."
  }
  ```

## Downloading the Full Dataset

To access the **complete RSICD dataset** with thousands of images and captions:

📥 **Download from Kaggle**: [RSICD Image Caption Dataset](https://www.kaggle.com/datasets/thedevastator/rsicd-image-caption-dataset)

The full dataset contains:
- Significantly more images (several thousand)
- More diverse scene types
- Better representation of different geographic features
- Sufficient data for training robust CNN+LSTM models

## Usage

This example dataset is ideal for:
- Understanding the data format and structure
- Quick prototyping and testing of captioning models
- Demonstration and educational purposes
- Debugging CNN+LSTM pipeline before full dataset training

### Quick Start

1. **Load the captions**:
   ```python
   import json
   with open('captions.json', 'r') as f:
       captions = json.load(f)
   ```

2. **Access images and their descriptions**:
   ```python
   for image_file, caption in captions.items():
       print(f"{image_file}: {caption}")
   ```

3. **Prepare for model training**:
   - Use images from the `images/` directory
   - Pair with corresponding captions from `captions.json`
   - Normalize and preprocess as needed for your model

## Next Steps

1. **Explore this example** to understand the data format
2. **Download the full dataset** from Kaggle
3. **Train your CNN+LSTM model** on the complete dataset
4. **Fine-tune** on specific scene types if needed

## Citation

If you use this dataset in your research, please cite the original RSICD dataset:

```
Remote Sensing Image Captioning Dataset (RSICD)
Available at: https://www.kaggle.com/datasets/thedevastator/rsicd-image-caption-dataset
```

---

**Note**: This is a sample/example version of RSICD. For production models and comprehensive research, use the full dataset available on Kaggle.
