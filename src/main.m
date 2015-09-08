%% test the various neural network behavior on harmonic sources identification
% DL algorithm is employed to improve the harmonic sources identification
% performance. RBFNN RNN BPN are realized as comparison.
% the evaluation procedure is:
% 1. input the harmonic wavefrom detected from the NI device
% 2. set parameters for each algorithm
% 3. output the relative error.
%% input the harmonic waveform
% simulate data
f0 = 50; % base frequency is 60Hz
samplesInPeriod = 64;
sampleFreq = f0*samplesInPeriod;
t_step = 1/sampleFreq;
t_lim = 0.5;
t = t_step:t_step:t_lim;
% harmonic parameter
harOrder = 4;
trainSampleNum = 10;
Amp = unidrnd(100,trainSampleNum,harOrder);
Phase = unidrnd(360,trainSampleNum,harOrder);
trainData = [];

for idx = 1:trainSampleNum
    signal = generateSimVoltage(t,harOrder,Amp(idx,:),Phase(idx,:));
    trainData = [trainData;signal];
end

%% DL algorithm
% NNPara = ;
%% RBFNN
% set the initial options of the RBFNN. The recommended values are from the
% paper by _Radial-Basis-Function-Based Neural Network for Harmonic
% Detection_ by Gary Chang in 2010.
RBFNNPara.inputNum = samplesInPeriod/2;
RBFNNPara.outputNum = harOrder;
RBFNNPara.maxHiddenNum = 20;
RBFNNPara.trainSampleNum = trainSampleNum;
RBFNNPara.hiddenNum = 8;
RBFNNPara.weightLearningRate = 0.3;
RBFNNPara.centerLearningRate = 0.01;
RBFNNPara.stdErrLearningRate = 0.6;
RBFNNPara.maxIterationNum = 500;
RBFNNPara.trainData = trainData;
RBFNNPara.trainAmp = Amp;
RBFNNPara.trainPhase = Phase;
% calculate the harmonic identification result
[ampNet, phaseNet] = RBFNN4Har(RBFNNPara);

% test calculate the harmonic estimation by newrb

% RBFNN validation with test data
t_lim = 10;
t = t_step:t_step:t_lim;
harOrder = 4;
%testAmp = [100,30,20,15];
testAmp = Amp(2,:);
testPhase = [152 35 0 0];

testData = generateSimVoltage(t,harOrder,testAmp,testPhase);
testDataNum = int32(length(testData)/RBFNNPara.inputNum);

testData = reshape(testData,[RBFNNPara.inputNum, testDataNum]);

testAmpVec = repmat(testAmp', [1,testDataNum]);
AmpTest = sim(ampNet, testData);
ampErr = max(abs(AmpTest - testAmpVec),[],2);
disp(ampErr);

testPhaseVec = repmat(testPhase',[1,testDataNum]);
PhaseTest = sim(phaseNet, testData);
phaseErr = max(abs(PhaseTest - testPhaseVec),[],2);
disp(phaseErr);

%% RNN
RNNPara.inputNum = samplesInPeriod/2;
RNNPara.outputNum = harOrder;
RNNPara.maxHiddenNum = 20;
RNNPara.trainSampleNum = trainSampleNum;
RNNPara.hiddenNum = 8;
RNNPara.weightLearningRate = 0.3;
RNNPara.centerLearningRate = 0.01;
RNNPara.stdErrLearningRate = 0.6;
RNNPara.maxIterationNum = 500;
RNNPara.trainData = trainData;
RNNPara.trainAmp = Amp;
RNNPara.trainPhase = Phase;

%
% test calculate the harmonic estimation by layer recurrent network
trainDataNum = int32(length(trainData)/RNNPara.inputNum);
trainData = reshape(trainData,[RNNPara.inputNum, trainDataNum]);
trainData = mat2cell(trainData,[RNNPara.inputNum],ones(1,trainDataNum));

trainAmp = repmat(Amp, [1 (trainDataNum/RNNPara.trainSampleNum)]);
trainAmp = reshape(trainAmp', [RNNPara.outputNum trainDataNum]);
trainAmp = mat2cell(trainAmp, [RNNPara.outputNum], ones(1,trainDataNum));

trainPhase = repmat(Phase, [1 (trainDataNum/RNNPara.trainSampleNum)]);
trainPhase = reshape(trainPhase', [RNNPara.outputNum trainDataNum]);
trainPhase = mat2cell(trainPhase, [RNNPara.outputNum], ones(1,trainDataNum));

net = layrecnet(1:2,15);
net.trainParam.goal = 1e-4;
[Xs,Xi,Ai,Ts] = preparets(net,trainData,trainAmp);
net = train(net,Xs,Ts,Xi,Ai);
view(net)
Y = net(Xs,Xi,Ai);
perf = perform(net,Y,Ts);



%
%res = RNN4Har(RNNParas);
%% BPN

%% error comparison

%% display result