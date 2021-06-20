% rectangle('position',[1,1,5,5],'curvature',[1,1],'edgecolor','r','facecolor','g');
% 'position',[1,1,5,5]表示从（1,1）点开始高为5，宽为5；
% 'curvature',[1,1]表示x,y方向上的曲率都为1，即是圆弧；
% 'edgecolor','r'表示边框颜色是红色；
% 'facecolor','g'表示面内填充颜色为绿色。

clear all
clc
close all

rectangle('position',[-1,-1,2,2],'edgecolor','k', 'facecolor','y');

A = cell(1,20);
A(1) = {[0.1, 0.1]};
A(2) = {[0.3, 0.5]};
A(3) = {[0.6, 0.2]};
A(4) = {[0.825, 0.5]};
A(5) = {[0.1, 0.75]};
A(6) = {[-0.1, 0.8]};
A(7) = {[-0.4, 0.5]};
A(8) = {[-0.6, 0.2]};
A(9) = {[-0.8, 0.7]};
A(10) = {[-0.9, 0.1]};

A(11) = {[-0.1, -0.75]};
A(12) = {[-0.4, -0.8]};
A(13) = {[-0.3, -0.5]};
A(14) = {[-0.6, -0.2]};
A(15) = {[-0.825, -0.5]};
A(16) = {[0.1, -0.5]};
A(17) = {[0.3, -0.2]};
A(18) = {[0.5, -0.75]};
A(19) = {[0.725, -0.3]};
A(20) = {[0.9, -0.9]};

radius=zeros(1,20);
radius(1)=0.225;
radius(2)=0.125;
radius(3)=0.3;
radius(4)=0.075;
radius(5)=0.1;
radius(6)=0.075;
radius(7)=0.3;
radius(8)=0.1;
radius(9)=0.2;
radius(10)=0.1;
radius(11)=0.1;
radius(12)=0.075;
radius(13)=0.075;
radius(14)=0.125;
radius(15)=0.3;
radius(16)=0.075;
radius(17)=0.175;
radius(18)=0.125;
radius(19)=0.1;
radius(20)=0.1;


for i = 1 : 20
    centroid = cell2mat(A(i));
    rectangle('Position',[centroid(1),centroid(2),radius(i),radius(i)],'Curvature',[1,1], 'facecolor','w');
    hold on;
end