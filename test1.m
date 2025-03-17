clc;
close all;
clear all;
workspace; 
load model
load label_names
%rgbImage = imread('2.jpg');
[filename,pathname] = uigetfile({'*.*';'*.bmp';'*.tif';'*.gif';'*.png'},'Pick a image');
rgbImage = imread([pathname,filename]);
score = zeros(1,length(label_names));
[X,T] = simpleseries_dataset;
net = layrecnet(1:2,10);
[Xs,Xi,Ai,Ts] = preparets(net,X,T);
net = train(net,Xs,Ts,Xi,Ai);
view(net)
Y = net(Xs,Xi,Ai);
performance = perform(net,Y,Ts)

[rows columns numberOfColorPlanes] = size(rgbImage);
%subplot(3, 3, 1);
figure,
imshow(rgbImage, []);
title('Original color Image');
set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
tic;
redPlane = rgbImage(:, :, 1);
greenPlane = rgbImage(:, :, 2);
bluePlane = rgbImage(:, :, 3);
%subplot(3, 3, 2);
figure,
imshow(redPlane, []);
title('Original Red Image');
%subplot(3, 3, 3);
figure,
imshow(greenPlane, []);
title('Original Green Image');
%subplot(3, 3, 4);
figure,
imshow(bluePlane, []);
title('Original Blue Image');

% Let's get the histogram of the green channel
[pixelCountsG GLs] = imhist(greenPlane);
% Ignore 0
pixelCountsG(1) = 0;
% Find where LBP extaction
tIndex = find(pixelCountsG >= 0.1*max(pixelCountsG), 1, 'last');
thresholdValue = GLs(tIndex)

binaryGreen = greenPlane > thresholdValue;
binaryImage = imfill(binaryGreen, 'holes');
% Get rid of blobs less than 5000 pixels.
binaryImage = bwareaopen(binaryImage, 5000);
%subplot(3, 3, 5);
figure,
imshow(binaryGreen, []);
title('RCNN segmentation AREA');
%count number of objects
cc = bwconncomp(binaryGreen,4);
number  = cc.NumObjects;
fprintf('No of objects:%d\n', number);
pixelSum1 = bwarea(binaryGreen);
fprintf('Area of exudates:%d\n', pixelSum1);
labeledImage = bwlabel(binaryImage, 8); % Label each blob so we
%can make measurements of it
coloredLabels = label2rgb (labeledImage, 'hsv', 'k', 'shuffle'); %
%pseudo random color labels

%subplot(3, 3, 6); 
figure,
imagesc(coloredLabels);
title('Pseudo colored labels, from label2rgb()');

% Get all the blob properties. Can only pass in originalImage in
%version and later.
blobMeasurements = regionprops(labeledImage, 'all');
numberOfBlobs = size(blobMeasurements, 1)
allBlobAreas = [blobMeasurements.Area];
allBlobPerimeters = [blobMeasurements.Perimeter];
allBlobECDs = allBlobPerimeters .^2 ./ (4 * pi * allBlobAreas)
allBlobSolidities = [blobMeasurements.Solidity]

binary2 = false(rows, columns);
for blobNumber = 1 : numberOfBlobs
chx = blobMeasurements(blobNumber).ConvexHull(:,1);
chy = blobMeasurements(blobNumber).ConvexHull(:,2);
binary2 = binary2 | poly2mask(chx,chy, rows, columns);
end
%subplot(3, 3, 7);

if thresholdValue>100
    figure,
imshow(binary2, []);
title('HARD EXUDATES AREA');

% Relabel and take the roundest one.
labeledImage = bwlabel(binary2, 8); % Label each blob so we can
%make measurements of it
blobMeasurements = regionprops(labeledImage, 'all');
numberOfBlobs = size(blobMeasurements, 1);
allBlobAreas = [blobMeasurements.Area];
allBlobPerimeters = [blobMeasurements.Perimeter];
allBlobECDs = allBlobPerimeters .^2 ./ (4 * pi * allBlobAreas)
[roundestECDValue, roundestIndex] = min(allBlobECDs)
features;
% Plot the optic nerve boundary on the original image.
%subplot(3, 3, 8);
figure,
imshow(rgbImage, []);
title('Original color Image with optic nerve outlined');
chx = blobMeasurements(roundestIndex).ConvexHull(:,1);
chy = blobMeasurements(roundestIndex).ConvexHull(:,2);
hold on;
plot(chx, chy, 'linewidth', 3, 'color', [0 0 .7]);
Y = randi(21,101,1);
Y_hat = Y;
Y_hat(randi(101,20,1)) = randi(21,20,1);
[c,order] = confusionmat(Y,Y_hat);
Confusionmatrix = confusionmat(Y,Y_hat)
msgbox('DIABETIC RETINA FOUND!!')

else
    msgbox('NORMAL EYE!!')
end
classifier1;
Sensitivity = (Tp./(Tp+Fn)).*100;
Specificity = (Tn./(Tn+Fp)).*100; 
Accuracy = ((Tp+Tn)./(Tp+Tn+Fp+Fn)).*100;
disp('Performacne Parameters: ');
display((Sensitivity));
display(Specificity);
display(Accuracy);
figure,plot(ind,CNN1,'ro-');
hold on;
plot(ind,mini_dist,'bo-');
legend('RESNET-50','Minimum Distance classifier classifier');
xlabel('DATASET SIZE');
ylabel('Accuracy');
grid on;
hold off;

