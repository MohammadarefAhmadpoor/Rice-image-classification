# Rice Image Classification using Deep Learning

This repository contains a Tensorflow implementation of a Convolutional Neural Network (CNN) model for classifying different varieties of rice grains based on their images. The model is trained on the Rice Image Dataset, which consists of images of five different rice varieties: Arborio, Basmati, Ipsala, Jasmine, and Karacadag.

## Dataset

The Rice Image Dataset used in this project is available on Kaggle. It can be downloaded from the following link:

https://www.kaggle.com/datasets/dollarzerobidsl/rice-image-dataset

The dataset contains a total of 5,947 images of rice grains, categorized into five different classes. The images are stored in separate folders for each class.
[![](./src/Rice1.png)](#)
[![](./src/Rice2.png)](#)

## Requirements

To run this project, you need to have the following dependencies installed:

- Tensorflow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- splitfolders


## Usage

1. Clone this repository to your local machine.
2. Download the Rice Image Dataset from the provided Kaggle link and extract it to a suitable location.
3. Update the `df_path` variable in the code to point to the location of the extracted dataset.
4. Run the script, and it will preprocess the data, split it into training, validation, and test sets, and train the CNN model.
5. The trained model will be saved as `CNN_Model.h5` in the current working directory.

## Code Overview
 
 Steps:

1. **Data Loading and Exploration**: The images and their corresponding labels are loaded from the dataset directory, and some exploratory visualizations are generated to understand the data distribution.
2. **Data Preprocessing**: The dataset is split into training, validation, and test sets using the `splitfolders` library.
3. **Data Preparation**: The images are loaded and preprocessed using Keras' `image_dataset_from_directory` function, which handles resizing, batching, and one-hot encoding of labels.
4. **Model Definition**: A sequential CNN model is defined using Keras' functional API. The model consists of convolutional layers, max-pooling layers, flattening, and dense layers with ReLU and softmax activations.
5. **Model Training**: The model is trained on the training set, with the validation set used for monitoring the model's performance during training.
6. **Visualization of Training Progress**: The training and validation loss and accuracy are plotted to monitor the model's convergence.
7. **Model Saving and Loading**: The trained model is saved as an HDF5 file (`CNN_Model.h5`) and then loaded back for evaluation or inference.

## Results

The trained CNN model achieves an accuracy of around 95% on the validation set after 15 epochs of training, but it isn't be. The exact performance may vary depending on the specific hardware and randomization used during training.

## Contributing

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
