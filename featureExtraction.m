classdef featureExtraction
    methods(Static)

        % Function to calculate symmetry values for a given mask image
        % at six different rotations (60 degrees apart).
        function symmetryValues = calculateSingleSymmetry(inputMask)
            % Initialize an array to store symmetry values for each rotation
            symmetryValues = zeros(1, 6);
            
            % Copy the input mask image to a variable 'rotatedMask'
            rotatedMask = inputMask;
            
            % Iterate six times over a rotation of 60 degrees each (by degrees).
            for rotationIndex = 1:6
                % Rotate the image by 60 degrees
                rotatedMask = imrotate(rotatedMask, 60);
                
                % Calculate symmetry value for the current rotation
                % Symmetry is evaluated as the ratio of overlapping pixels to total pixels
                symmetryValues(rotationIndex) = sum((rotatedMask & fliplr(rotatedMask))) / sum((rotatedMask | fliplr(rotatedMask)));
            end
        end
        
        
        % Function for border irregularity edge detection
        function irregularityValue = borderIrregularityEdgeDetection(binaryMask)
            % Convert binary mask to logical
            binaryMask = logical(binaryMask);
            
            % Perform Canny edge detection on the binary mask
            edges = edge(binaryMask, 'Canny');
            
            % Calculate perimeter and area from the binary mask
            perimeter = sum(edges(:));
            area = sum(binaryMask(:));
            
            % Calculate irregularity using perimeter and square root of area
            irregularityValue = perimeter / sqrt(area);
        end
        
        
        
        % Function for creating a colour histogram
        function histogram = colourHistogram(image)
            % Define the number of bins for each colour channel
            numBins = 8;
            binWidth = 256 / numBins;
            
            % Initialize the histogram matrix
            histogram = zeros(numBins, numBins, numBins);
            
            % Reshape the image data for easier indexing
            [rows, cols, channels] = size(image);
            data = reshape(image, rows * cols, channels);
            
            % Calculate the indices for each pixel in the histogram
            indices = floor(double(data) / binWidth) + 1;
            
            % Update the histogram counts based on pixel indices
            for i = 1:length(indices)
                histogram(indices(i, 1), indices(i, 2), indices(i, 3)) = ...
                    histogram(indices(i, 1), indices(i, 2), indices(i, 3)) + 1;
            end
            
            % Normalize the histogram
            histogram = histogram / sum(sum(sum(histogram)));
        end
        
        
        % Function for Principal Component Analysis (PCA)
        function [principalComponents, sortedEigenvalues, projectedData] = performPCA(inputData)
            % Calculate the covariance matrix of the input data
            covarianceMatrix = cov(inputData);
            
            % Compute the eigenvectors and eigenvalues from the covariance matrix
            [eigenvectors, eigenvalueMatrix] = eig(covarianceMatrix);
            eigenvalues = diag(eigenvalueMatrix);
            
            % Sort the eigenvectors and eigenvalues in descending order
            [~, sortedEigenvalueIndices] = sort(eigenvalues, 'descend');
            principalComponents = eigenvectors(:, sortedEigenvalueIndices);
            sortedEigenvalues = eigenvalues(sortedEigenvalueIndices);
            
            % Project the input data onto the principal components
            projectedData = inputData * principalComponents;
        end
        
        
        
        % Function to calculate texture features using Local Binary Patterns (LBP)
       function textureMatrix = calculateTexture(grayImages)
            
            textureMatrix = zeros(length(grayImages), numel(extractLBPFeatures(grayImages{1})));
            
            for imageIndex = 1:length(grayImages)
                % Extract LBP features for the current grayscale image
                textureFeatures = extractLBPFeatures(grayImages{imageIndex});
                
                textureMatrix(imageIndex, :) = textureFeatures(:);
            end
        end
        
        
        % Function to calculate circularity of a binary mask
        function circularityValue = circularity(binaryMask)
            % Ensure binary mask is logical
            binaryMask = logical(binaryMask);
            
            % Compute region properties (area and perimeter) of the binary mask
            stats = regionprops(binaryMask, 'Area', 'Perimeter');
            
            % Extract area and perimeter from region properties
            area = stats.Area;
            perimeter = stats.Perimeter;
            
            % Calculate circularity using area and perimeter
            circularityValue = (4 * pi * area) / (perimeter^2);
        end
        
        
        function radialVarianceValue = calculateRadialVariance(image)
            % Employing binary mask logical
            binaryMask = logical(image);
            
            % Find region properties
            stats = regionprops(binaryMask, 'PixelIdxList', 'Centroid');
            
            % Extract pixel indices and centroid from region properties
            pixelIdxList = stats.PixelIdxList;
            centroid = stats.Centroid;
            
            % Calculate radial distances
            [rows, cols] = ind2sub(size(binaryMask), pixelIdxList);
            radialDistances = sqrt((rows - centroid(2)).^2 + (cols - centroid(1)).^2);
            
            % Compute radial variance
            radialVarianceValue = var(radialDistances);
        end
        
        
        function [meanValue, stdDevValue, skewnessValue, kurtosisValue] = calculateIntensityStatistics(image)
        
            % Convert the image to a column vector for statistical calculations
            pixelValues = double(image(:));
        
            % Calculate statistical features
            meanValue = mean(pixelValues);
            stdDevValue = std(pixelValues);
            skewnessValue = skewness(pixelValues);
            kurtosisValue = kurtosis(pixelValues);
        end
        
        
        function compactnessValue = calculateCompactness(binaryMask)
            % Employing binary mask logical
            binaryMask = logical(binaryMask);
            
            % Compute region properties
            stats = regionprops(binaryMask, 'Area', 'Perimeter');
            
            % Extract area and perimeter from region properties
            area = stats.Area;
            perimeter = stats.Perimeter;
            
            % Calculate compactness
            compactnessValue = (perimeter^2) / area;
        end
    end
end
