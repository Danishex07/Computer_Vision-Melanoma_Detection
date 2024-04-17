%% Step - 1
% ================== Importing the data =====================

% Create image datastore for lesion images
lesionDatastore = imageDatastore('lesionimages');

% Read all lesion images from the datastore
lesionImages = readall(lesionDatastore);

% Count the number of lesion images in the datastore
numLesionImages = length(lesionImages);
disp(['Number of lesion images: ' num2str(numLesionImages)]);


% Create image datastore for masks
maskDatastore = imageDatastore('masks');

% Read all mask images from the datastore
maskImages = readall(maskDatastore);

% Count the number of mask images in the datastore
numMaskImages = length(maskImages);
disp(['Number of mask images: ' num2str(numMaskImages)]);

disp('Loading');

addpath('.')
% addpath('preprocessing');
% addpath('featureExtraction');

% Creating an instance of the preprocessing class
imageProcessor = preprocessing;

% Creating an instance of the featureExtraction class
featuresObj = featureExtraction;


%% Step - 2

% ================== Image Pre Preprocessing =====================


numImages = length(lesionImages);

% figure;
% subplot(1, 5, 1), imshow(imgs{1}), title('Original Image');

% Iterate over each image for preprocessing
for imageIndex = 1:numImages

    disp(['Image-' num2str(imageIndex)]);
    
    % Histogram equalization for contrast enhancement
    equalizedHist = imageProcessor.histogramEqualization(lesionImages{imageIndex});
    
    % Gamma for further contrast adjustment
    gammaAdjusted = imageProcessor.changeGamma(equalizedHist, 0.9);
    
    % Convert to grayscale
    grayImage = rgb2gray(gammaAdjusted);
    
    % Laplacian sharpening
    sharpenedImage = imageProcessor.laplacianSharpen(grayImage);
    
    % Median filter for denoising
    denoisedImage = imageProcessor.medianFilter(sharpenedImage, 3);
    
    % Remove hair 
    hairRemovedImage = imageProcessor.removeHair(denoisedImage);
    
    % Apply binary mask
    segmentedImage = imageProcessor.applyBinaryMask(hairRemovedImage, maskImages{imageIndex});
    
    % Sharpen the segmented image
    finalImage = imsharpen(segmentedImage);
    
    % Update the processed images array
    processedImages{imageIndex} = finalImage;
end

% subplot(1, 5, 2), imshow((imgs_hist{1})), title('Histogram Image');
% subplot(1, 5, 3), imshow(imgs_grey{1}), title('Gray Image');
% subplot(1, 5, 4), imshow(hair_removed{1}), title('Hairless Image');
% subplot(1, 5, 5), imshow(segmented_imgs{1}), title('Segmented Image');

disp('Preprocessing Done');

%% Step - 3


% ==================== Symmetry Calculation ==============================

% Loop through the mask images and calculate symmetry values
numImages = length(maskImages);
symmetryMatrix = zeros(numImages, 6);

for imageIndex = 1:numImages
    symmetryMatrix(imageIndex, :) = featuresObj.calculateSingleSymmetry(maskImages{imageIndex});
end

disp('Symmetry Calculation Completed');


% ================== Border Irregularity ==========================

% Calculate border irregularity for each mask
for imageIndex = 1:numImages
    borderFeatures = featuresObj.borderIrregularityEdgeDetection(maskImages{imageIndex});
    borderMatrix(imageIndex) = borderFeatures(end);
end

borderMatrix = reshape(borderMatrix, numImages, 1);

disp('Border');


% ================ Colour Histogram and PCA ===============================

% Extracted lesion images for colour processing
colourImages = lesionImages;

% Initialize matrix to store colour histograms for each image
colourHistogramsMatrix = zeros(numImages, 512);

for imageIdx = 1:numImages
    % Create colour histogram for the current image
    currentColourHist = featuresObj.colourHistogram(colourImages{imageIdx});
    
    % Flatten and store the colour histogram in the matrix
    colourHistogramsMatrix(imageIdx, :) = currentColourHist(:);
end

% Perform Principal Component Analysis (PCA) on colour data
[principalComponents, eigenvalues, projectedData] = featuresObj.performPCA(colourHistogramsMatrix);

% Retain the first 5 principal components for further analysis
projectedImages_five_axis = projectedData(:, 1:5);

% Retain the first 200 principal components for further analysis
projectedImages_twoHundred_axis = projectedData(:, 1:200);

% Display progress message
disp('Colour Histogram and PCA Processing Completed.');



% =============== Diameter/Circularity Calculation ==========================

% Apply Gaussian filter on the mask
for imageIndex = 1:numImages
    gaussianImage{imageIndex} = imgaussfilt(maskImages{imageIndex});
end

gaussianImage = reshape(gaussianImage, numImages, 1);

% Calculate circularity for each mask
circularityMatrix = zeros(numImages, 1);

for imageIndex = 1:numImages
    circularityMatrix(imageIndex) = featuresObj.circularity(gaussianImage{imageIndex});
end

disp('Diameter/Circularity Calculation Completed');


% =============== LBP Feature Extraction ======================

graySegmentedImages = processedImages;

% Extract LBP texture features for each preprocessed image
for imageIndex = 1:numImages
    textureFeatures = extractLBPFeatures(graySegmentedImages{imageIndex});
    textureMatrix(imageIndex, :) = textureFeatures(:);
end

disp('LBP Processing Completed');


% ============= Gray-Level Co-occurrence Matrix (GLCM)  =================

for imageIndex = 1:numImages
    % Calculate the Gray-Level Co-occurrence Matrix (GLCM) for the current image
    glcm = graycomatrix(processedImages{imageIndex});
    
    % Extract texture features using graycoprops for the GLCM
    contrastStats = graycoprops(glcm, {'contrast'});
    homogeneityStats = graycoprops(glcm, {'homogeneity'});
    correlationStats = graycoprops(glcm, {'correlation'});
    energyStats = graycoprops(glcm, {'energy'});

    % Extract specific texture features from the statistics
    contrastFeature = contrastStats.Contrast;
    homogeneityFeature = homogeneityStats.Homogeneity;
    correlationFeature = correlationStats.Correlation;
    energyFeature = energyStats.Energy;

    % Concatenate the extracted features into a single row vector
    glcmTotal = horzcat(contrastFeature, homogeneityFeature, correlationFeature, energyFeature);
    
    % Store the feature vector in the glcmMatrix
    glcmMatrix(imageIndex, :) = glcmTotal(:);
end


disp('GLCM Processing Completed');


% ==================== HOG Feature Extraction ==============================


% Extract HOG features for each preprocessed image
for imageIndex = 1:numImages
    grayImage = processedImages{imageIndex};

    hogFeatures = extractHOGFeatures(grayImage);

    hogMeans{imageIndex} = mean(hogFeatures);

    hogFeatureMatrix(imageIndex) = hogMeans{imageIndex}(end);
end

hogFeatureMatrix = reshape(hogFeatureMatrix, numImages, 1);

disp('HOG Feature');


% =============== Compactness Feature Extraction ======================

compactnessMatrix = zeros(numImages, 1);

for imageIndex = 1:numImages
    segmented_image = processedImages{imageIndex};

    % Calculate compactness for each mask
    compactnessValue = featuresObj.calculateCompactness(segmented_image);
    compactnessMatrix(imageIndex) = compactnessValue;
end

% Reshape matrix if needed
compactnessMatrix = reshape(compactnessMatrix, numImages, 1);

% Display a message
disp('Compactness calculations completed.');


% =============== Radial Variance Feature Extraction ======================

radialVarianceMatrix = zeros(numImages, 1);

% Outer loop for each mask
for imageIndex = 1:numImages

    segmented_image = processedImages{imageIndex}; 

    % Calculate radial variance for each mask
    radialVarianceValue = featuresObj.calculateRadialVariance(segmented_image);
    radialVarianceMatrix(imageIndex) = radialVarianceValue;
end

radialVarianceMatrix = reshape(radialVarianceMatrix, numImages, 1);

% Display a message
disp('Radial Variance calculations completed.');


% =============== Statistical Feature Extraction ======================

statisticalFeaturesMatrix = zeros(numImages, 4); % Getting 4 statistical features

for imageIndex = 1:numImages
   
    segmented_image = processedImages{imageIndex}; 
    [meanValue, stdDevValue, skewnessValue, kurtosisValue] = featuresObj.calculateIntensityStatistics(segmented_image);

    statisticalFeaturesMatrix(imageIndex, :) = [meanValue, stdDevValue, skewnessValue, kurtosisValue];
end

statisticalFeaturesMatrix = reshape(statisticalFeaturesMatrix, numImages, 4);

% Display a message
disp('Statistical features calculations completed.');


%% Feature combination and Model Testing


% Combining Asymetry, Border, Colour, Diameter
featuresMat = cat(2, symmetryMatrix, borderMatrix, circularityMatrix);
featuresMatrix = cat(2, featuresMat, colourHistogramsMatrix);

                        
% Combining Research Papers Finding Features
paperFeaturesMatrix = cat(2,glcmMatrix, hogFeatureMatrix);


% Combining Texture, Shape Statistical features
shapeFeatures = cat(2, textureMatrix,compactnessMatrix);
shapeFeatures_Stats = cat(2, shapeFeatures, statisticalFeaturesMatrix);
statsFeatures_RV = cat(2, shapeFeatures,radialVarianceMatrix);

% PCA with 5 axis -> projectedImages_five_axis
% PCA with 200 axis -> projectedImages_twoHundred_axis
imfeatures_1 = cat(2, featuresMatrix,paperFeaturesMatrix, shapeFeatures, projectedImages_five_axis);
imfeatures_2 = cat(2, featuresMatrix,paperFeaturesMatrix, statsFeatures_RV, projectedImages_twoHundred_axis);
imfeatures_3 = cat(2, featuresMatrix,paperFeaturesMatrix, shapeFeatures_Stats, projectedImages_twoHundred_axis);
imfeatures_4 = cat(2, featuresMatrix,paperFeaturesMatrix, shapeFeatures, projectedImages_twoHundred_axis);
imfeatures_withoutHist = cat(2, featuresMatrix,featuresMat, shapeFeatures, projectedImages_twoHundred_axis);

featureSets = {imfeatures_1,imfeatures_2,imfeatures_3,imfeatures_4,imfeatures_withoutHist};
featureNames = {'Feature Set 1', 'Feature Set 2', 'Feature Set 3', 'Feature Set 4', 'Feature Set W/o histogram'};

disp('Combining Features');

disp('------------------------------')


% ---------------------------- Model ------------------------------- %

% Getting labels for the images
groundtruth = getLabels("groundtruth.mat", lesionImages);

for i = 1:numel(featureSets)
    imfeatures = featureSets{i};
  
    [cm, order, pred] = performSVMClassificationWithCV(imfeatures,groundtruth);
   
    % Call the function to calculate performance metrics
    [accuracy, sensitivity, precision, specificity] = calculatePerformanceMetrics(cm);

    % Display the current accuracy
    
    disp('Performance Metrics:');
    disp(['Current Feature Set: ' featureNames{i}]);
    disp(['Accuracy: ' num2str(accuracy)]);
    disp(['Sensitivity: ' num2str(sensitivity)]);
    disp(['Specificity: ' num2str(specificity)]);

    disp('------------------------------')
end
%% 

imfeatures = imfeatures_4;

[cm, order, pred] = performSVMClassificationWithCV(imfeatures,groundtruth);

% Call the function to calculate performance metrics
[accuracy, sensitivity, precision, specificity] = calculatePerformanceMetrics(cm);

    disp('Final Performance Metrics:');
    disp(['Current Feature Set-4 ']);
    disp(['Accuracy: ' num2str(accuracy)]);
    disp(['Sensitivity: ' num2str(sensitivity)]);
    disp(['Specificity: ' num2str(specificity)]);
    disp(['Precision: ' num2str(precision)]);

    disp('------------------------------')

% Display a confusion chart for feature-4 visualizing classification performance.
confusionchart (cm, order);

% Display Correctly Classified vs Incorrectly Classified Image
displayClassifiedImages(lesionImages, pred, groundtruth);

% Call the function
visualizeMetricsPercentage(accuracy, sensitivity, precision, specificity);




% ------------------------ Functions ------------------------------


% Function to get labels from a text file
function labels = getLabels(textFile, images)
    % Load the content of the groundtruth.mat file into a structure
    data = load(textFile);
    
    % Access the cell array named 'groundtruth'
    labels = data.groundtruth;
    numImages = length(images);
    labels = labels(1:numImages);    
end


function [cm, order, pred] = performSVMClassificationWithCV(imfeatures, groundtruth)
    % Set the seed for the random number generator
    rng(1);
    % Set up the SVM model
    svm = fitcsvm(imfeatures, groundtruth);
    % Perform 10-fold cross-validation
    cvsvm = crossval(svm);
    % Obtain predictions from the dataset
    pred = kfoldPredict(cvsvm);
    % Getting the confusion matrix
    [cm, order] = confusionmat(groundtruth, pred);
end



% Function to calculate performance metrics (accuracy, sensitivity, precision, and specificity)
function [accuracy, sensitivity, precision, specificity] = calculatePerformanceMetrics(confusionMatrix)
    % True Positive, False Negative, True Negative, False Positive
    TP = confusionMatrix(2, 2);
    FN = confusionMatrix(2, 1);

    TN = confusionMatrix(1, 1);
    FP = confusionMatrix(1, 2);

    % Calculate accuracy
    accuracy = (TP + TN) / sum(confusionMatrix(:));

    % Calculate sensitivity (true positive rate or recall)
    sensitivity = TP / (TP + FN);

    % Calculate precision (positive predictive value)
    precision = TP / (TP + FP);

    % Calculate specificity (true negative rate)
    specificity = TN / (TN + FP);
end



function visualizeMetricsPercentage(accuracy, sensitivity, precision, specificity)
    % Convert metrics to percentages
    accuracyPercent = accuracy * 100;
    sensitivityPercent = sensitivity * 100;
    precisionPercent = precision * 100;
    specificityPercent = specificity * 100;

    % Create a horizontal bar chart
    figure;
    metrics = [accuracyPercent, sensitivityPercent, precisionPercent, specificityPercent];
    bar(metrics, 'FaceColor', [0.5 0.7 0.9]);

    % Add labels and title
    xlabel('Metrics');
    ylabel('Values (%)');
    title('Performance Metrics Bar Chart');

    % Customize axis ticks and labels
    xticks(1:4);
    xticklabels({'Accuracy', 'Sensitivity', 'Precision', 'Specificity'});

    % Set custom y-axis limits to leave space from the top
    ylim([0, 100]);  %  The upper limit of chart

    % Display the values on top of the bars
    text(1:length(metrics), metrics, num2str(metrics', '%0.2f%%'), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');

    % Add a title to the current figure
    title('Performance Metrics Graph');

    grid on;
    box on;
end


function displayClassifiedImages(lesionImages, pred, groundtruth)
    correct_indices = find(strcmp(pred, groundtruth));
    incorrect_indices = find(~strcmp(pred, groundtruth));

    % Display correctly classified images
    figure;
    for i = 1:min(3, numel(correct_indices)) % Display up to 3 correct images
        subplot(2, 3, i);
        imshow(lesionImages{correct_indices(i)});
        title(['Predicted: ', pred{correct_indices(i)}, ', Ground Truth: ', groundtruth{correct_indices(i)}]);
    end

    % Display misclassified images
    for i = 1:min(3, numel(incorrect_indices)) % Display up to 3 incorrect images
        subplot(2, 3, i+3);
        imshow(lesionImages{incorrect_indices(i)});
        title(['Predicted: ', pred{incorrect_indices(i)}, ', Ground Truth: ', groundtruth{incorrect_indices(i)}]);
    end
    sgtitle('Correct and Misclassified Images');
end
