# White Blood Cell Image Classification with Convolutional Neural Networks

## Table of Contents

## Overview

In medical diagnostics, traditional methods like Giemsa staining are time-consuming and involve hazardous chemicals. This project proposes a deep learning approach using Convolutional Neural Networks (CNNs) to classify unstained WBC images, enabling safer and more efficient identification of WBCs.

The model achieves an accuracy of 92% on a validation dataset. 

## Prerequisites

1. Python 3.8
2. Pytorch 2.2.2

## Usage

1.  **Dataset:**  The Double-labeled RaabinWBC  dataset is available on [Raabin Data](https://dl.raabindata.com/WBC/Cropped_double_labeled/) and the Peripheral Blood Cell (PBC) dataset is available on [PBC](https://data.mendeley.com/datasets/snkd93bnjr/1). *Note: Due to size limitations, this repository includes only a small subset of the dataset for demonstration purposes. To run the full training, you will need to download the full dataset.*

2.  **Notebook** The notebook also includes code for training the model from scratch, but this is not required to run the demo.

3.  **Trained Model** Download the wbc_model_gray.pth file

## License

[MIT License](LICENSE)
