clc;
clear;
close all;

%----------------------read image---------------------------%
image_name = '1.jpg';
img = imread(char(strcat('./bw_images/', image_name)));
if(size(img, 3)==3)
    img = rgb2gray(img);
end

%----------------convert to B/W image-----------------------%
img = 255 - img;
for i = 1:size(img, 1)
    for j = 1:size(img, 2)
        if (img(i,j)>=127)
            img(i,j)=1;
        elseif (img(i,j)<127)
            img(i,j)=0;
        end
    end
end

%------------------------find connected components-------------------%
conn = 8;
CC = bwconncomp(img,conn);
L = labelmatrix(CC);
L = L+1;

%---------filler takes the summation of all intensities in a label---------%  

% channel - R
counterr = zeros(max(L(:))+1, 1);
fillerr = zeros(max(L(:))+1, 1);
fillerr = double(fillerr);

% channel - G
counterg = zeros(max(L(:))+1, 1);
fillerg = zeros(max(L(:))+1, 1);
fillerg = double(fillerg);

% channel - B
counterb =zeros(max(L(:))+1, 1);
fillerb = zeros(max(L(:))+1, 1);
fillerb = double(fillerb);
filler = zeros(max(L(:))+1, 3);

%--------------------read output image from cGAN--------------------------% 
image_name = '1.jpg';
img2 = imread(char(strcat('./gan_images/', image_name)));

%---------------reshape cGAN's output to the input image size-------------%
[a,b] = size(L);
img2 = imresize(img2, [a,b], 'bicubic');

%--------------find mean color intensity in each color channel------------%
for i=1:size(L,1)
    for j=1:size(L,2)   
        counterr(L(i,j))=counterr(L(i,j))+1;
        fillerr(L(i,j))=fillerr(L(i,j))+double(img2(i,j,1));

        counterg(L(i,j))=counterg(L(i,j))+1;
        fillerg(L(i,j))=fillerg(L(i,j))+double(img2(i,j,2));

        counterb(L(i,j))=counterb(L(i,j))+1;
        fillerb(L(i,j))=fillerb(L(i,j))+double(img2(i,j,3));
    end
end

for i=1:size(fillerr,1)
    if(counterr(i)~=0)
        fillerr(i)=fillerr(i)/counterr(i);
    end

    if(counterg(i)~=0)
        fillerg(i)=fillerg(i)/counterg(i);
    end

    if(counterb(i)~=0)
        fillerb(i)=fillerb(i)/counterb(i);
    end
end

%% 
%-----------------replacing mean values in the final image----------------%
final = cat(3, img, img, img);
for i = 1:size(L, 1)
    for j = 1:size(L, 2) 
        final(i,j,1) = fillerr(L(i,j));
        final(i,j,2) = fillerg(L(i,j));
        final(i,j,3) = fillerb(L(i,j));    
    end
end
imwrite(final,char(strcat('./results/', image_name)));

 
%% 
%----------------------------applying k-means-----------------------------%
ab = final;
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,3);
ab = im2double(ab);
nColors = 50;
[cluster_idx, cluster_center] = kmeans(ab, nColors, 'distance', 'sqEuclidean', 'Replicates', 3);
for i = 1:size(ab,1)
    ab(i,1) = cluster_center(cluster_idx(i), 1);
    ab(i,2) = cluster_center(cluster_idx(i), 2);
    ab(i,3) = cluster_center(cluster_idx(i), 3);
end
ab = reshape(ab, nrows, ncols, 3);

%------------------------------increase saturation------------------------%
x = ab;
HSV = rgb2hsv(x);
HSV(:, :, 2) = HSV(:, :, 2) * 1.5;
HSV(HSV > 1) = 1; 
x = im2uint8(hsv2rgb(HSV));
imwrite(x, char(strcat('./results/', image_name)));

%% 
%----------------smoothing image and applying unsharp masking-------------%
inp = imread(char(strcat('./input_images/', image_name)));  
inp = rgb2gray(inp);

%-------------------------get edges by sobel------------------------------% 
h = fspecial('sobel');
BW1 = imfilter(inp, h, 'replicate');
BW2 = imfilter(inp, h', 'replicate');
BW = BW1+BW2;
BW = 255-BW;
BW = im2double(BW);

%-----------------gaussian filt on cGAN's prediction----------------------%
test = imgaussfilt(x, 1.5, 'FilterSize', 15);
BW3 = imbinarize(BW,0.6);
BW3 = 1-BW3;
BW3 = uint8(BW3);

%-----------------------apply unsharp masking-----------------------------%
alpha = 10;
test = test - alpha*BW3;

%-----------------------apply shading effect------------------------------%
for i = 1:size(test, 3)
 test(:,:,i) = test(:,:,i)-uint8((double(255-inp))/2.5);
end

result_dir = './results/';
imwrite(test, char(strcat(result_dir, image_name)));
figure, imshow(test);