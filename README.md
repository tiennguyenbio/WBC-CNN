# White Blood Cell Image Classification with Convolutional Neural Networks

## Table of Contents

## Overview

In medical diagnostics, traditional methods like Giemsa staining are time-consuming and involve hazardous chemicals. This project proposes a deep learning approach using Convolutional Neural Networks (CNNs) to classify unstained WBC images, enabling safer and more efficient identification of WBCs.

The model achieves an accuracy of 92% on a validation dataset. 

## Prerequisites

Torch
Pytorch

## Usage

1. **Demo**
You can try out the model online via this [Demo Link](https://tiennguyenbio-wbc-cnn.hf.space/).
For example, see this [Demo Session](https://tiennguyenbio-wbc-cnn.hf.space/?__theme=system&deep_link=jCDZYGuxwIQ) to explore how it works.

To test the model with sample images:

1. Use the images in the [Demo folder](https://github.com/tiennguyenbio/WBC-CNN/tree/main/Demo).
2. Each image’s label corresponds to its **subfolder name**.

3.  **Dataset:**  The Double-labeled RaabinWBC  dataset is available on [Raabin Data](https://dl.raabindata.com/WBC/Cropped_double_labeled/) and the Peripheral Blood Cell (PBC) dataset is available on [PBC](https://data.mendeley.com/datasets/snkd93bnjr/1). *Note: Due to size limitations, this repository includes only a small subset of the dataset for demonstration purposes. To run the full training, you will need to download the full dataset.*

4.  **Notebook** The notebook also includes code for training the model from scratch, but this is not required to run the demo.

5.  **Trained Model** Load the wbc_model_gray.pth file to the notebook and use image files from Demo folder to Test the model.

## License

[MIT License](LICENSE)
