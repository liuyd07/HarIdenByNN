function res = RNN4Har(trainData, RNNParas)
% The Elman Recurrent Neural Network algorithm is based on the paper A 
% comparison of MLP, RNN and ESN in determining harmonic contributions from 
% nonlinear loads in 2008. Training method is Back-propagation through time,(BPTT)

[X,T] = simpleseries_dataset;
net = layrecnet(1:2,10);
[Xs,Xi,Ai,Ts] = preparets(net,X,T);
net = train(net,Xs,Ts,Xi,Ai);
view(net)
Y = net(Xs,Xi,Ai);
perf = perform(net,Y,Ts)

res = Y;

trainData = RBFNNPara.trainData;
Amp = RBFNNPara.trainAmp;
Phase = RBFNNPara.trainPhase;

% test calculate the harmonic estimation by layer recurrent network
trainDataNum = int32(length(trainData)/RBFNNPara.inputNum);
trainData = reshape(trainData,[RBFNNPara.inputNum, trainDataNum]);
trainAmp = repmat(Amp, [1 (trainDataNum/RBFNNPara.trainSampleNum)]);
trainAmp = reshape(trainAmp', [RBFNNPara.outputNum trainDataNum]);
trainPhase = repmat(Phase, [1 (trainDataNum/RBFNNPara.trainSampleNum)]);
trainPhase = reshape(trainPhase', [RBFNNPara.outputNum trainDataNum]);


net = layrecnet(1:2,10);
[Xs,Xi,Ai,Ts] = preparets(net,X,T);
net = train(net,Xs,Ts,Xi,Ai);
view(net)
Y = net(Xs,Xi,Ai);
perf = perform(net,Y,Ts)