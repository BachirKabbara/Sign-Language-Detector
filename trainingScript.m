clc
clear all
close all
warning off
            
% Define the input size for the AlexNet architecture
inputSize = [227 227 3];

% Load the AlexNet architecture and replace the last layers
g = alexnet;
layers = g.Layers;
layers(end-2) = fullyConnectedLayer(29);
layers(end) = classificationLayer;

% Define the imageDatastore with preprocessing to resize images
allImages = imageDatastore('English Sign Language Alphabet\asl_alphabet_train\asl_alphabet_train','IncludeSubfolders',true, 'LabelSource','foldernames');
allImages = augmentedImageDatastore(inputSize, allImages);

% Define the training options
opts = trainingOptions('sgdm', 'InitialLearnRate', 0.001, 'MaxEpochs', 15, 'MiniBatchSize', 64, 'ExecutionEnvironment', 'auto');

% Train the network
myNet1 = trainNetwork(allImages, layers, opts);

% Save the trained network
save myNet;