# Malignant Melanoma Detection System

## Overview:
This MATLAB script implements a comprehensive pipeline for analyzing lesion images in medical imaging. The pipeline encompasses various stages including data importing, image pre-processing, feature extraction, model training, and performance evaluation. It aims to assist medical professionals in diagnosing and characterizing skin lesions by automating the analysis process and providing quantitative insights into lesion properties.

## Usage:
1. Run the script `main.m` in MATLAB.
2. The script automates the following steps:
   - **Data Importing**: Load lesion images and corresponding masks.
   - **Image Pre-processing**: Enhance image quality, remove noise, and prepare images for feature extraction.
   - **Feature Extraction**: Extract diverse features such as symmetry, border irregularity, color histograms, texture features, GLCM features, HOG features, compactness, radial variance, and statistical features.
   - **Model Training and Evaluation**: Train SVM classifiers using extracted features and evaluate performance using cross-validation. Display performance metrics like accuracy, sensitivity, precision, and specificity. Visualize the confusion matrix and correctly/incorrectly classified images.

## File Structure:
- `main.m`: Main MATLAB script containing the entire pipeline.
- `preprocessing.m`: MATLAB class for image pre-processing operations.
- `featureExtraction.m`: MATLAB class for feature extraction operations.
- `groundtruth.mat`: Ground truth labels for the lesion images.
- `lesionimages/`: Directory containing lesion images.
- `masks/`: Directory containing corresponding masks for lesion images.

System will take time to preprocess the images beacause of the hair removal algorithm.
A display message will appear in the command window displaying current progress of image by a number.

