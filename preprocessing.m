classdef preprocessing
    methods(Static)
        % Function for Histogram Equalization
        function histogramEq = histogramEqualization(img)
            % Compute the histogram of the input image
            hImg = imhist(img);
            
            % Normalize the histogram to obtain probability distribution
            hImgPx = hImg / numel(img);
            
            % Compute the cumulative sum of the normalized histogram
            cumSum = cumsum(hImgPx);
            
            % Perform histogram equalization using cumulative distribution
            histogramEq = im2uint8(cumSum(img + 1));
        end
        
        
        % Function to adjust gamma of an image
        function gammaAdjusted = changeGamma(img, val)
            % Adjust gamma of the input image
            gammaAdjusted = imadjust(img, [], [], val);
        end
        
        
        % Function for Laplacian Sharpening
        function sharpenedImage = laplacianSharpen(grayImg)
            % Apply Matlab Laplacian filter to the grayscale image
            laplacianImg = imfilter(grayImg, fspecial('laplacian'));
            
            % Add the Laplacian image to the original image for sharpening
            sharpenedImage = grayImg + laplacianImg;
            
            % Clip the resulting image values to the valid range [0, 255]
            sharpenedImage = uint8(max(0, min(255, sharpenedImage)));
        end
        
        
        % Function for Median Filtering
        function denoisedImage = medianFilter(grayImg, windowSize)
            % Apply median filtering to the grayscale image
            denoisedImage = medfilt2(grayImg, [windowSize, windowSize]);
        end
        
        
        
        % Function to apply binary mask on an image
        function segmentedImg = applyBinaryMask(grayImg, binaryMask)
            % Convert binary mask to logical
            binaryMask = imbinarize(binaryMask);
            
            % Zero out pixels in the grayscale image where the binary mask is false
            grayImg(~binaryMask) = 0;
            
            % Output the segmented image
            segmentedImg = grayImg;
        end

        % Function to remove hair from an image
        function resultImage = removeHair(image)
            % Convert image to double precision
            image = im2double(image);
            
            % Define a structuring element for morphological operations
            structuringElement = strel('disk', 20);
            
            % Perform bottom-hat operation to enhance hair 
            hairs = imbothat(image, structuringElement);
            
            % Use region filling to remove hair 
            resultImage = regionfill(image, hairs);
        end
    end
end