clear;
clc;
close all;

imagename=strcat(string(1),'.jpg');
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
counterr=zeros(max(L(:))+1,1);
fillerr=zeros(max(L(:))+1,1);
fillerr=double(fillerr);
counterg=zeros(max(L(:))+1,1);
fillerg=zeros(max(L(:))+1,1);
fillerg=double(fillerg);
counterb=zeros(max(L(:))+1,1);
fillerb=zeros(max(L(:))+1,1);
fillerb=double(fillerb);

% filler=zeros(max(L(:))+1,3);

modes=zeros(max(L(:))*255,1);
img2=imread('./gan_images/1.jpg');
[a,b]=size(L);
img2 = imresize(img2, [a,b], 'bicubic');


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
counterr
fillerr

% % filler=uint8(filler);
% fillerr=double(fillerr);
% fillerg=double(fillerg);
% fillerb=double(fillerb);

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

% fillerr=im2double()

% 
%  fillerr=(fillerr.*255)/max(fillerr);
%  fillerg=(fillerg.*255)/max(fillerg);
%  fillerb=(fillerb.*255)/max(fillerb);
% 
% fillerr=uint8(fillerr);
%  fillerg=uint8(fillerg);
%  fillerb=uint8(fillerb);
%  %disp(filler);
z=zeros(a,b);
final=cat(3,img,img,img);
for i=1:size(L,1)
    for j=1:size(L,2)
%         
        final(i,j,1)=fillerr(L(i,j));
        final(i,j,2)=fillerg(L(i,j));
        final(i,j,3)=fillerb(L(i,j));

%                 final(i,j,1)=img2(i,j,1);
%                 final(i,j,2)=img2(i,j,2);
%                 final(i,j,3)=img2(i,j,3);
         
    end
end

imshow((final));
