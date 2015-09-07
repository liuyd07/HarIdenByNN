function simVoltageSignal = generateSimVoltage(t,harOrder,Amp,Phase)
%
f0 = 50; % electric frequency is 50Hz

tLength = length(t);
A = @(tl) repmat(Amp,[tl,1]) + 0.01*rand(tl,harOrder);
f = @(tl) repmat([1 3 5 7]*f0,[tl,1]) + 0.01*rand(tl,harOrder);
phi = @(tl) repmat(Phase/180*pi,[tl,1]);

simVoltageSignal = sum(A(tLength).*...
    sin(2*pi*f(tLength).*repmat(t',[1,harOrder]) + phi(tLength)),2);
simVoltageSignal = simVoltageSignal + db2mag(-26)*rand(size(simVoltageSignal));
