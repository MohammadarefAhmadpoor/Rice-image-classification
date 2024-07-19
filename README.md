# Rice Image Classification using Deep Learning

This repository contains Tensorflow and Pytorch implementation of a Convolutional Neural Network (CNN) model for classifying different varieties of rice grains based on their images. The model is trained on the [Rice Image Dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset), which consists of images of five different rice varieties: Arborio, Basmati, Ipsala, Jasmine, and Karacadag.

## Abstract

Rice, which is among the most widely produced grain products worldwide, has many genetic varieties. These varieties are separated from each other due to some of their features. These are usually features such as texture, shape, and color. With these features that distinguish rice varieties, it is possible to classify and evaluate the quality of seeds. This repository used Arborio, Basmati, Ipsala, Jasmine and Karacadag, which are five different varieties of rice often grown in Turkey. A total of 75,000 grain images, 15,000 from each of these varieties, are included in the dataset. A second dataset with 106 features including 12 morphological, 4 shape and 90 color features obtained from these images was used. Models were created using the Convolutional Neural Network (CNN) using both Pytorch and Tesnorflow libraries for building different CNN architectures. Classification successes from the models were achieved as 99.01% with Tensorflow and 99.95% for DNN and 100% for CNN. With the results, it is seen that the models used in the study in the classification of rice varieties can be applied successfully in this field


The dataset contains a total of 5,947 images of rice grains, categorized into five different classes. The images are stored in separate folders for each class.



## Requirements



## Usage

1. Clone this repository to your local machine.
2. Download the Rice Image Dataset from the provided Kaggle link and extract it to a suitable location.
3. Update the `df_path` variable in the code to point to the location of the extracted dataset.
4. Run the script, and it will preprocess the data, split it into training, validation, and test sets, and train the CNN model.
5. The trained model will be saved as `CNN_Model.h5` in the current working directory.


