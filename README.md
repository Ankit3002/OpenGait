# GaitBERT: Gait Recognition by Predicting Masked Frame

## Overview

GaitBERT is a deep learning model designed for gait recognition by predicting masked frames within video sequences. This approach utilizes a U-Net architecture combined with frame embedding, positional encoding, and a combination of regression and triplet losses to accurately reconstruct and identify gait patterns.

<div align="center"><img src="./assets/nm.gif" width = "100" height = "100" alt="nm" /><img src="./assets/bg.gif" width = "100" height = "100" alt="bg" /><img src="./assets/cl.gif" width = "100" height = "100" alt="cl" /></div>

------------------------------------------

## Architecture

The model leverages a U-Net architecture for frame reconstruction and identification. The U-Net consists of an encoder and decoder with skip connections to preserve spatial information across layers.

![image](https://github.com/arunpatwa/coding/assets/91215615/8aec4e28-8f88-4c5d-90c4-eb9ca8425f0d)


### Model Pipeline

1. **Input Frames**:
   - The input consists of T frames of a video sequence of an ID (a 3D volume or sequence).
   - Some frames within the sequence have masked portions in random locations.

2. **Frame Embedding and Positional Encoding**:
   - Each frame undergoes embedding with positional encoding to retain temporal context.

3. **U-Net Encoder**:
   - The encoder processes the input frames through multiple convolutional layers with ReLU activation and max pooling, progressively reducing the spatial dimensions.

4. **Video Embedding**:
   - The compressed representation is further processed to generate a video embedding.

5. **U-Net Decoder**:
   - The decoder reconstructs the frames from the compressed representation using transposed convolutions and concatenation with skip connections from the encoder layers.

6. **Loss Functions**:
   - **Regression Loss**: Applied to the reconstructed frames to minimize the difference between the predicted frames and the ground truth.
   - **Triplet + Cross Entropy Loss**: Ensures that the embeddings are well-clustered for frames of the same identity and well-separated for different identities.

## Usage

### Prerequisites

- Python 3.8 or higher
- PyTorch
- NumPy
- OpenCV

### Installation

1. Clone the repository:
   ```bash
   pip install gdown
   # Downloading the pkl file of the dataset over here...
   gdown --id 1j4phfJPn3gj6QhgrFy6FZIEvny-8htzB
   unzip /kaggle/working/CASIA-B-pkl
   git clone https://github.com/Ankit3002/OpenGait.git
   mv OpenGait/* ./
   rm -rf OpenGait

   # download the model checkpoint (weights) of the encoder from here...
   gdown --id 1IurwhECWDjkcuD1XwJTB2DBIsYeavZtR

   # download the whole model weights over here...
   gdown --id 10L-MZcV4cR-cxrNGgSyBM9R-r1N-jTnu


Take a look at https://drive.google.com/file/d/1tYRMVAeI2BP9T2rsenuYIKK4VucWRb23/view?usp=sharing for downloading the python file which contains all the code ...
   


Training
Prepare your dataset with video frames and corresponding masks. Organize the dataset in the following structure:

```bash

  dataset/
  ├── train/
  │   ├── ID1/
  │   │   ├── frame0.png
  │   │   ├── frame1.png
  │   │   └── ...
  │   └── ID2/
  │       ├── frame0.png
  │       ├── frame1.png
  │       └── ...
  └── test/
      ├── ID1/
      │   ├── frame0.png
      │   ├── frame1.png
      │   └── ...
      └── ID2/
          ├── frame0.png
          ├── frame1.png
          └── ...
```

