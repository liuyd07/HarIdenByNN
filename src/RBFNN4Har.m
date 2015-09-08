function [ampNet, phaseNet] = RBFNN4Har(RBFNNPara)
% the RNFNN algorithm is from the paper by _Radial-Basis-Function-Based
% Neural Network for Harmonic Detection_ by Gary Chang in 2010. The result
% is represented as relative error.

trainData = RBFNNPara.trainData;
Amp = RBFNNPara.trainAmp;
Phase = RBFNNPara.trainPhase;

% test calculate the harmonic estimation by newrb
trainDataNum = int32(length(trainData)/RBFNNPara.inputNum);
trainData = reshape(trainData,[RBFNNPara.inputNum, trainDataNum]);
trainAmp = repmat(Amp, [1 (trainDataNum/RBFNNPara.trainSampleNum)]);
trainAmp = reshape(trainAmp', [RBFNNPara.outputNum trainDataNum]);
trainPhase = repmat(Phase, [1 (trainDataNum/RBFNNPara.trainSampleNum)]);
trainPhase = reshape(trainPhase', [RBFNNPara.outputNum trainDataNum]);
save;

ampNet = newrb(trainData, trainAmp, 1e-4, 50, trainDataNum, 10);
save;

phaseNet = newrb(trainData, trainPhase, 1e-3, 50, trainDataNum, 10);
save;

