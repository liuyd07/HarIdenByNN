function err = RBFNN4Har(trainData,options)
% the RNFNN algorithm is from the paper by _Radial-Basis-Function-Based
% Neural Network for Harmonic Detection_ by Gary Chang in 2010. The result
% is represented as relative error.


% test calculate the harmonic estimation by newrb
trainDataNum = floor(length(trainData)/RBFNNPara.inputNum);
trainData = reshape(trainData,[RBFNNPara.inputNum, trainDataNum]);
trainAmp = repmat(Amp', [1,trainDataNum]);
trainPhase = repmat(Phase',[1,trainDataNum]);

net = newrb(trainData, trainAmp, 0, 1,20,1);
testData = generateSimVoltage(t,harOrder,Amp,Phase);
testData = reshape(testData,[RBFNNPara.inputNum, trainDataNum]);

AmpTest = sim(net, testData);
err = max(abs(AmpTest - trainAmp),[],2);

net = newrb(trainData, trainPhase, 0, 1,20,1);
testData = generateSimVoltage(t,harOrder,Amp,Phase);
testData = reshape(testData,[RBFNNPara.inputNum, trainDataNum]);

PhaseTest = sim(net, testData);
err = max(abs(PhaseTest - trainPhase),[],2);

