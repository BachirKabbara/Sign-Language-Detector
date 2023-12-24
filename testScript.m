clc;
close all;
clear all;
warning off;

% Initialize a webcam object
c=webcam;
% Load a pre-trained neural network for classification
load myNet;
% Set the x-coordinate of the top-left corner of the bounding box to 0
x=500;
% Set the y-coordinate of the top-left corner of the bounding box to 0
y=0;
% Set the height of the bounding box to 300 pixels
height=300;
% Set the width of the bounding box to 300 pixels
width=300;
% Create a bounding box with the above dimensions
bboxes=[x y height width];

% Start a continuous loop that runs until the script is manually stopped
while true

    % Capture a snapshot from the webcam
    e=c.snapshot;
    % Flip the image horizontally
    e = fliplr(c.snapshot);
    % Insert a rectangle around the region of interest in the captured frame
    IFaces = insertObjectAnnotation(e,'rectangle',bboxes,'Processing Area');
    % Crop the region of interest from the captured frame
    es=imcrop(e,bboxes);
    % Resize the region of interest to a fixed size of 227x227 pixels
    es=imresize(es,[227 227]);
    % Classify the object in the region of interest using the pre-trained neural network
    label=classify(myNet1,es);
    % Display the captured frame with the rectangle around the region of interest
    imshow(IFaces);
    % Set the title of the figure to the predicted label of the object
    title(char(label));
    % Update the figure with the latest snapshot and label
    drawnow;

end

