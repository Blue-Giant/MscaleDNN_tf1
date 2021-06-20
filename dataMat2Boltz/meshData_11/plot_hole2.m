function plot_hole2(num2holes)
    A = cell(1,20);
    A(1) = {[0.1, 0.1]};
    A(2) = {[0.3, 0.5]};
    A(3) = {[0.6, 0.2]};
    A(4) = {[0.825, 0.5]};
    A(5) = {[0.1, 0.75]};
    A(6) = {[0.8, 0.75]};
    A(7) = {[0.4, 0.75]};

    radius=zeros(1,20);
    radius(1)=0.225;
    radius(2)=0.16;
    radius(3)=0.2;
    radius(4)=0.125;
    radius(5)=0.1;
    radius(6)=0.16;
    radius(7)=0.09;



    for i = 1 : num2holes
        centroid = cell2mat(A(i));
        rectangle('Position',[centroid(1),centroid(2),radius(i),radius(i)],'Curvature',[1,1], 'facecolor','w');
        hold on;
    end
end