# Applied-ML-Group23
This is the repository for the model deployment assignment for the applied machine learning course.

# Dataset
Chest X-Ray Images (Pneumonia):
- Link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- Version: Version 2, Updated 2018/01/06

# Contributors
Assignment group 23
- Kevin Kuipers (s5051150)
- Federico Berdugo Morales (s5363268)
- Sían Bos García (s5962277)
- Mahmoud Saad (S6175767)

# Install dependencies and launch API
text

## Windows
```console
text
```
## Model description
Our project uses chest X-ray images to detect pneumonia in patiens through the use of a machine learning pipeline consisting of PCA which is mainly used for dimensionality reduction and a random forest classifier. The dataset is a standard chest X-ray pneumonia dataset which can be accessed above. Our final depluyed model is a random forest which. The CNN branch is maily experimental and not used in deployment. 

## What we have implemented
- We have loaded Chest X-Ray dataset into a train, validation and test sets
- Converted images to grayscale and resized them to 256 x 256
- Flattened images into vectors
- Appliued incremental PCA with 648 componets
- Splitting the PCA-compressed data into train/test
- Achieved an accuracy of 8

#

##
## Linux and macOS
```console
text
```
