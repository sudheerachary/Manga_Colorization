
 imagename=strcat(string(5),'.jpg');
img=imread(char(strcat('./bw_images/',imagename)));
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
conn=8;
CC = bwconncomp(img,conn);
L = labelmatrix(CC);
L=L+1;
%  imshow(label2rgb(L));
% counterr=zeros(max(L(:))+1,1);
 fillerr=zeros(max(L(:))+1,3);
% counterg=zeros(max(L(:))+1,1);
 fillerg=zeros(max(L(:))+1,1);
% counterb=zeros(max(L(:))+1,1);
 fillerb=zeros(max(L(:))+1,1);
% filler=zeros(max(L(:))+1,3);
moder=zeros(max(L(:)),256);
modeg=zeros(max(L(:)),256);
modeb=zeros(max(L(:)),256);

 %modeg=zeros(max(L(:)),256);
%modeb=zeros(max(L(:)),256);

s1='./gan_images/';
s2=string(5);
s3='.jpg';

img2=imread(char(strcat(s1,s2,s3)));
% img2=imread('./gan_images/1.jpeg');

[a,b]=size(L);
img2 = imresize(img2, [a,b], 'bicubic');
% 
% HSV = rgb2hsv(img2);
% % "20% more" saturation:
% HSV(:, :, 2) = HSV(:, :, 2) * 1.2;
% % or add:
% % HSV(:, :, 2) = HSV(:, :, 2) + 0.2;
% HSV(HSV > 1) = 1;  % Limit values
% img2 = im2uint8(hsv2rgb(HSV));



for i=1:size(L,1)
for j=1:size(L,2)   
 
    moder(L(i,j),img2(i,j,1)+1)=moder(L(i,j),img2(i,j,1)+1) +1;
    modeg(L(i,j),img2(i,j,2)+1)= modeg(L(i,j),img2(i,j,2)+1)+1;
    modeb(L(i,j),img2(i,j,3)+1)= modeb(L(i,j),img2(i,j,3)+1)+1;

end
end

% 
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
for i=1:max(L(:))

    [val,col]=max(moder(i,:));
    fillerr(i)=col;
    
    [val,col]=max(modeg(i,:));
    fillerg(i)=col;
    
    [val,col]=max(modeb(i,:));
    fillerb(i)=col;

end

 fillerr=(fillerr.*255)/max(fillerr);
 fillerg=(fillerg.*255)/max(fillerg);
 fillerb=(fillerb.*255)/max(fillerb);
%% 

  fillerr=uint8(filler);
  fillerg=uint8(fillerg);
  fillerb=uint8(fillerb);
 
final=cat(3,img,img,img);
for i=1:size(L,1)
    for j=1:size(L,2)
        
        final(i,j,1)=filler(L(i,j));
        final(i,j,2)=fillerg(L(i,j));
        final(i,j,3)=fillerb(L(i,j));

%                 final(i,j,1)=img2(i,j,1);
%                 final(i,j,2)=img2(i,j,2);
%                 final(i,j,3)=img2(i,j,3);
         
    end
end

imshow(final);
% 
% s1='./results/';
% s2=string(o);
% s3='_seg';
% s4='.jpg';
% imwrite(final,char(strcat(s1,s2,s3,s4)));
% 
%% 
% 
% ab=final;
% nrows = size(ab,1);
% ncols = size(ab,2);
% ab = reshape(ab,nrows*ncols,3);
% ab=im2double(ab);
% nColors=10;
% [cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean','Replicates',3);
% %pixel_labels = reshape(cluster_idx,nrows,ncols);
% 
% for i=1:size(ab,1)
% ab(i,1)=cluster_center(cluster_idx(i),1);
% ab(i,2)=cluster_center(cluster_idx(i),2);
% ab(i,3)=cluster_center(cluster_idx(i)   ,3);
% end
% ab=reshape(ab,nrows,ncols,3);
% %  ab(:,:,1)=ab(:,:,1).*255/max(ab())
% x=ab;
% HSV = rgb2hsv(x);
% HSV(:, :, 2) = HSV(:, :, 2) * 1.2;
% HSV(HSV > 1) = 1; 
% x = im2uint8(hsv2rgb(HSV));
% 
% s1='./results/';
% s2=string(o);
% s3='_kmeans';
% s4='.jpg';
% 
% imwrite(x,char(strcat(s1,s2,s3,s4)));
% imshow(x);
% s1='./input_images/';   % directory for input images
% s2=string(o);
% s3='.jpeg';
% 
% %% 
% 
% inp=imread(char(strcat(s1,s2,s3)));
% inp=rgb2gray(inp);
% 
% 
% for i=1:size(x,3)
%  x(:,:,i)=x-uint8((double(255-inp(i)))/3);
%  end
% 
% s1='./results/';
% s2=string(o);
% s3='_final_result';
% s4='.jpg';
% imwrite(x,char(strcat(s1,s2,s3,s4)));
% imshow(x);
% 
%  end
