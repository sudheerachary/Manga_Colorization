clear;
close all;
clc;
%8 images in testing 
for o=1:8
if(o==3)
continue;
end
%-------------------------Reading image---------------------------%
imagename=strcat(string(o),'.jpg');                     
img=imread(char(strcat('./bw_images/',imagename)));

if(size(img,3)==3)
img=rgb2gray(img);
end

%----------------Converting to binary image-----------------------%
img=255-img;
for i=1:size(img,1)
for j=1:size(img,2)
if(img(i,j)>=127)
img(i,j)=1;
elseif(img(i,j)<127)
img(i,j)=0;
end
end
end

%------------------------finding connected components-------------------%
conn=8;
CC = bwconncomp(img,conn);
L = labelmatrix(CC);
imwrite(label2rgb(L),char(strcat('./results/label_',imagename)));

L=L+1;              %----------------- L starts from 0--------------------%
imshow(label2rgb(L));

% counterr=zeros(max(L(:))+1,1);
% counterg=zeros(max(L(:))+1,1);
% fillerg=zeros(max(L(:))+1,1);
% counterb=zeros(max(L(:))+1,1);
% fillerb=zeros(max(L(:))+1,1);
% filler=zeros(max(L(:))+1,3);
% mode_rgb=zeros(max(L(:)),256,256,256);
%modeg=zeros(max(L(:)),256);
%modeb=zeros(max(L(:)),256);

%--------------------counter counts no of pixels in a label-------------%
%-----filler takes the summation of all intensities in a label-----------%  

counterr=zeros(max(L(:))+1,1);
fillerr=zeros(max(L(:))+1,1);
fillerr=double(fillerr);

counterg=zeros(max(L(:))+1,1);
fillerg=zeros(max(L(:))+1,1);
fillerg=double(fillerg);

counterb=zeros(max(L(:))+1,1);
fillerb=zeros(max(L(:))+1,1);
fillerb=double(fillerb);
filler=zeros(max(L(:))+1,3);


s1='./gan_images/';
s2=string(o);
s3='.jpg';

%----------------------read output of image from cGan--------------------% 
img2=imread(char(strcat(s1,s2,s3)));

%-----------------------reshape gan o/p to the input image size----------%
[a,b]=size(L);
img2 = imresize(img2, [a,b], 'bicubic');

%------------------find mean color intensity in each color channel-------%

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

% 
% 
% 
% for i=1:size(L,1)
% for j=1:size(L,2)   
%  
%     mode_rgb(L(i,j),img2(i,j,1)+1,img2(i,j,2)+1,img2(i,j,3)+1)=mode_rgb(L(i,j),img2(i,j,1)+1,img2(i,j,2)+1,img2(i,j,3)+1) +1;
% %     modeg(L(i,j),img2(i,j,2)+1)= modeg(L(i,j),img2(i,j,2)+1)+1;
% %     modeb(L(i,j),img2(i,j,3)+1)= modeb(L(i,j),img2(i,j,3)+1)+1;
% 
% end
% end


% for i=1:max(L(:))
% temp_count=0;
% temp_r=0;temp_g=0;temp_b=0;
%     for j=1:256
%     for k=1:256
%     for l=1:256
%     if(mode_rgb(i,j,k,l)>temp_count)
%     temp_count=mode_rgb(i,j,k,l);
%     temp_r=j-1;temp_g=k-1;temp_b=l-1;
%     end
%     end
%     end
%     end
% filler(i,1)=temp_r;
% filler(i,2)=temp_g;
% filler(i,3)=temp_b;
% 
% end
%  
% % 
% %  fillerr=(fillerr.*255)/max(fillerr);
% %  fillerg=(fillerg.*255)/max(fillerg);
% %  fillerb=(fillerb.*255)/max(fillerb);

%% 
%----------------------replacing mean values in the final image-----------%
final=cat(3,img,img,img);
for i=1:size(L,1)
    for j=1:size(L,2)
        
        final(i,j,1)=fillerr(L(i,j));
        final(i,j,2)=fillerg(L(i,j));
        final(i,j,3)=fillerb(L(i,j));    
    end
end

% imshow((final));

s1='./results/';
s2=string(o);
s3='_seg';
s4='.jpg';
imwrite(final,char(strcat(s1,s2,s3,s4)));


%% 
% 
%  filler=uint8(filler);
% %  fillerg=uint8(fillerg);
% %  fillerb=uint8(fillerb);
%  
% final=cat(3,img,img,img);
% for i=1:size(L,1)
%     for j=1:size(L,2)
%         
%         final(i,j,1)=filler(L(i,j),1);
%         final(i,j,2)=filler(L(i,j),2);
%         final(i,j,3)=filler(L(i,j),3);
% 
% %                 final(i,j,1)=img2(i,j,1);
% %                 final(i,j,2)=img2(i,j,2);
% %                 final(i,j,3)=img2(i,j,3);
%          
%     end
% end
% 
% imshow(final);
% 
% 
%% 

%----------------------applying k-means---------------------------------%
ab=final;
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,3);
ab=im2double(ab);
nColors=50;
[cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean','Replicates',3);


for i=1:size(ab,1)
ab(i,1)=cluster_center(cluster_idx(i),1);
ab(i,2)=cluster_center(cluster_idx(i),2);
ab(i,3)=cluster_center(cluster_idx(i)   ,3);
end
ab=reshape(ab,nrows,ncols,3);

%------------------------------increase saturation------------------------%
x=ab;
HSV = rgb2hsv(x);
HSV(:, :, 2) = HSV(:, :, 2) * 1.5;
HSV(HSV > 1) = 1; 
x = im2uint8(hsv2rgb(HSV));

s1='./results/';
s2=string(o);
s3='_kmeans';
s4='.jpg';

imwrite(x,char(strcat(s1,s2,s3,s4)));
% imshow(x);

%% 

%---------------------smoothing image and applying unsharp masking---------%
 
s1='./input_images/';  
 s2=string(o);
 s3='.jpeg';
 inp=imread(char(strcat(s1,s2,s3)));  %-------read input image----------%
 inp=rgb2gray(inp);
 
%-------------------sobel filter on input image for edges----------------% 
h = fspecial('sobel');
BW1=imfilter(inp,h,'replicate');
BW2=imfilter(inp,h','replicate');
BW=BW1+BW2;
BW=255-BW;
BW=im2double(BW);

test=imgaussfilt(x,1.5,'FilterSize',15); %-----gaussian filt on gan o/p--%
BW3=im2bw(BW,0.6);
BW3=1-BW3;
BW3=uint8(BW3);

%------------------increasing edge strength value--------------------%
test=(test-BW3*10);
% figure,imshow(test); 

%----------------------apply shading effect------------------------------%
for i=1:size(test,3)
 test(:,:,i)=test(:,:,i)-uint8((double(255-inp))/2.5);
end
% imshow(test);

s1='./results/';
s2=string(o);
s3='_final_result';
s4='.jpg';
imwrite(test,char(strcat(s1,s2,s3,s4)));
figure,imshow(test);
 
   end
