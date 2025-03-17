function feat=GLCM(I)
% Otsu Binarization for segmentation
level = graythresh(I);
%gray = gray>80;
img = im2bw(I,.6);
img = bwareaopen(img,80); 
img2 = im2bw(I);

signal1 = img2(:,:);

[cA1,cH1,cV1,cD1] = dwt2(signal1,'db4');
[cA2,cH2,cV2,cD2] = dwt2(cA1,'db4');
[cA3,cH3,cV3,cD3] = dwt2(cA2,'db4');

DWT_feat = [cA3,cH3,cV3,cD3];
G = pca(DWT_feat);
% whos DWT_feat
% whos G

g = graycomatrix(G);
stats = graycoprops(g,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast
Correlation = stats.Correlation
Energy = stats.Energy
Homogeneity = stats.Homogeneity
Mean = mean2(G)
Standard_Deviation = std2(G)
Entropy = entropy(G)
RMS = mean2(rms(G))
%Skewness = skewness(img)
Variance = mean2(var(double(G)))
a = sum(double(G(:)));
Smoothness = 1-(1/(1+a))
Kurtosis = kurtosis(double(G(:)))
Skewness = skewness(double(G(:)))
% Inverse Difference Movement
m = size(G,1);
n = size(G,2);
in_diff = 0;
for i = 1:m
    for j = 1:n
        temp = G(i,j)./(1+(i-j).^2);
        in_diff = in_diff+temp;
    end
end
in_diff=in_diff
[x1 sc] = princomp(I,'Economy');
x1=x1(:,1)';
%Features
feat = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness,in_diff];
