ab=imread('./gan_images/res1.jpg');
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,3);
ab=im2double(ab);
nColors=10;
[cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean','Replicates',3);
%pixel_labels = reshape(cluster_idx,nrows,ncols);

for i=1:size(ab,1)
ab(i,1)=cluster_center(cluster_idx(i),1);
ab(i,2)=cluster_center(cluster_idx(i),2);
ab(i,3)=cluster_center(cluster_idx(i)   ,3);
end
ab=reshape(ab,nrows,ncols,3);
%  ab(:,:,1)=ab(:,:,1).*255/max(ab())
%% 

x=ab;
% x=(im2uint8(ab));
HSV = rgb2hsv(x);
HSV(:, :, 2) = HSV(:, :, 2) * 1.2;
HSV(HSV > 1) = 1; 
x = im2uint8(hsv2rgb(HSV));
% imshow(x);

inp=imread('./input_images/1.jpeg');
inp=rgb2gray(inp);
inp=imgaussfilt(inp,'sigma'=0.4,'FilterSize'=5);

 for i=1:size(x,3)
 x(:,:,i)=x-uint8((double(255-inp(i)))/3);
 end