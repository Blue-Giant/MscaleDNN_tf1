clear all
close all
clc
data2mesh = load('meshXY6.mat');
meshxy = data2mesh.meshXY;
meshx = meshxy(1,1:end);
meshy = meshxy(2,1:end);

utrue=(sin(pi*meshx).*sin(pi*meshy)+0.05*sin(20*pi*meshx).*sin(20*pi*meshy));
figure('name', 'ueps')
plot3(meshx,meshy,utrue,'b.')
hold on