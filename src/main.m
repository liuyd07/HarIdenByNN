%% test the various neural network behavior on harmonic sources identification
% DL algorithm is employed to improve the harmonic sources identification
% performance. RBFNN RNN BPN are realized as comparison.
% the evaluation procedure is:
% 1. input the harmonic wavefrom detected from the NI device
% 2. set parameters for each algorithm
% 3. output the relative error.
%% input the harmonic waveform
% simulate data
sampleFreq = 3840;
t_step = 1/sampleFreq;
t_lim = 1;
t = t_step:t_step:t_lim;
% harmonic parameter
harOrder = 4;
Amp = unidrnd(100,10,4);
Phase = unidrnd(360,10,4);
trainData = [];

for idx = 1:10
    signal = generateSimVoltage(t,harOrder,Amp(idx,:),Phase(idx,:));
    trainData = [trainData;signal];
end

%% DL algorithm
% NNPara = ;
%% RBFNN
% set the initial options of the RBFNN. The recommended values are from the
% paper by _Radial-Basis-Function-Based Neural Network for Harmonic
% Detection_ by Gary Chang in 2010.
RBFNNPara.inputNum = 32;
RBFNNPara.outputNum = 4;
RBFNNPara.maxHiddenNum = 20;
RBFNNPara.hiddenNum = 8;
RBFNNPara.weightLearningRate = 0.3;
RBFNNPara.centerLearningRate = 0.01;
RBFNNPara.stdErrLearningRate = 0.6;
RBFNNPara.maxIterationNum = 500;
RBFNNPara.trainData = trainData;

% calculate the harmonic identification result
% err = RBFNN4Har(trainData,RBFNNPara);

% test calculate the harmonic estimation by newrb
trainDataNum = floor(length(trainData)/RBFNNPara.inputNum);
trainData = reshape(trainData,[RBFNNPara.inputNum, trainDataNum]);
trainAmp = repmat(Amp, [1 (trainDataNum/10)]);
trainAmp = reshape(trainAmp', [harOrder trainDataNum]);
trainPhase = repmat(Phase, [1 (trainDataNum/10)]);
trainPhase = reshape(trainPhase', [harOrder trainDataNum]);

ampNet = newrb(trainData, trainAmp, 1e-3, 50, 1200, 10);
phaseNet = newrb(trainData, trainPhase, 1e-3, 50, 1200, 10);

%
t_lim = 10;
t = t_step:t_step:t_lim;
harOrder = 4;
testAmp = [100,30,20,15];
testPhase = [152 35 0 0];

testData = generateSimVoltage(t,harOrder,testAmp,testPhase);
testData = reshape(testData,[RBFNNPara.inputNum, trainDataNum]);

testAmpVec = repmat(testAmp', [1,trainDataNum]);
AmpTest = sim(ampNet, testData);
ampErr = max(abs(AmpTest - testAmpVec),[],2);
disp(ampErr);

testPhaseVec = repmat(testPhase',[1,trainDataNum]);
PhaseTest = sim(phaseNet, testData);
phaseErr = max(abs(PhaseTest - testPhaseVec),[],2);
disp(phaseErr);

%% RNN

%% BPN

%% error comparison

%% display result