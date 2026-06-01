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
## Project Description
Our project aims to detect the pnuemonia in patients using X-ray imagse. By implementing a machine-learning pipeline which consists of PCA for dimensionality reduction and a Random Forest classifier. The dataset we are using is a standard Chest X-ray Pneumonia data set (link is above). The final deployed model is a random forest whilst the CNN branch is purely experimental and not used in deployment.
This project detects pneumonia from chest X‑ray images using a machine‑learning pipeline consisting of PCA for dimensionality reduction and a Random Forest classifier. The dataset is the standard Chest X‑Ray Pneumonia dataset (Kermany et al.). The final deployed model is the Random Forest; the CNN branch is experimental and not used in deployment.

## What we have implemented
- We have loaded the X-ray dataset (train/val/test)
- Converted the images to grayscale (256 x 256)
- Flattened the images into vector
- Applied incremental PCA with 648 components
- Created CSV datasets for train, validation and test which are image_data.csv, dev_data.csv and test_data.csv.
- Trained a RandomForestClassifier with class_weight="balanced"
- Split the PCA-compressed data into train/test
- We now have proper dataset splitting
- Branch structure which clearly divides the work (main, Baseline_model_final, cnn_plus_baseline, notebook-fixes)
- Created a experimental CNN branch.

## What we achieved
Our random forest baseline achieved around 84% validation score, which after calculating is better than random guessing strategies like uniform guessing which has 50%, highest frequency guessing 72% and stratisfied guessing is 61%. 
<img width="388" height="21" alt="image" src="https://github.com/user-attachments/assets/216046e3-8bb4-49d2-9dcf-3a181a5f4fdd" />


## API documentation
Showing API documentation
<img width="1600" height="658" alt="image" src="https://github.com/user-attachments/assets/1654d2c7-b180-4567-9b7e-675c1731de25" />
Example request and response
<img width="1600" height="766" alt="image" src="https://github.com/user-attachments/assets/4c1e053a-f3fe-4a0e-b1dd-7e85e399a459" />

## How the endpoint works
An image is uploaded
1. The api reads the file and uses graycale conversion
2. Image is resized to 256 x 256
3. Image is flattened into a 1-D vector
4. Pca transformer reduces it to 648 components
5. Random Forest model predicts the class
6. Then finally API returns the prediction JSON 
## 
## Linux and macOS
```console
text
```
