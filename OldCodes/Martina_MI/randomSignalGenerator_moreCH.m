clear; clc; close all;

% This script generates a random signal consisting of a sequence of zeroes 
% and ones of length N min, such that after binning with certain dt the
% entropy of the signal equals 1.

dt=1600;

Length=60; % min
Fs=16000;
N=Fs*60*Length/dt;
disp(['Simulated data duration: ' num2str(Length) ' min.'])
sig=round(rand(N,1));
Ts=find(sig);
Ts=Ts*dt;
sig=zeros(N*dt,1);
sig(Ts)=1;
edges=dt:dt:N*dt;
count=histc(Ts,edges);
Cs=zeros(size(Ts)); % signal stored in Channel 0.
StimTimes=Ts;

%% This section adds the same signal to be recorded in part at Channel 1 and in part at channel 2.
Ts=[Ts;Ts];
secondchannel=ones(size(Cs)); secondchannel(7:7:end)=2;
Cs=[Cs; secondchannel ];
[Ts, index]=sort(Ts);
Cs=Cs(index);

%% Add channel 4 with random timing
timestamps=unique(sort(round(Length*60*Fs*rand(size(StimTimes)))));
timestamps=timestamps(3:3:end);
Cs1=ones(size(timestamps))*4;
Ts=[Ts;timestamps];
Cs=[Cs; Cs1];
[Ts, index]=sort(Ts);
Cs=Cs(index);


%%

data.Ts=Ts(1:1:end); data.Cs=Cs(1:1:end);
%data.Cs(5:5:end)=1;
% T=round(rand(N,1));
% TT=find(T);
% TTT=TT*dt;
data.StimTimes=((data.Ts(1:4:end))/Fs); % These should be in sec  should be eq to stim times in the first definition

disp(' ');
disp(['Signal contains ' num2str(length(sig)) ' data points']);
disp(['Signal contains ' num2str(length(data.Ts)) ' ones']);
disp(' ');
disp(['Entropy signal = ' num2str(entropy(sig))])
disp(['Entropy binned signal = ' num2str(entropy(count))])
disp(['Mean signal = ' num2str(mean(sig))])
disp(['SD signal = ' num2str(std(sig))])


eval(['save simulation2_quarterTsEqstim_random_4ch_' num2str(Length) '_' num2str(dt) ' data'])
Cs=data.Cs; Ts=data.Ts; StimTimes=data.Ts;