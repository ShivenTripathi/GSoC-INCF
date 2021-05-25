clear; close all; clc;
%load MEA-37578-electricalStim-StimTimes-uint
load MEA-37070-electricalStim-StimTimes-uint % no activity after hour 17
%load MEA-37068-stim % no activity after hour 16 and possibly less
%somewhere halfway
% load MEA-37063-stim
% Expt 7 had very low activity.
%load MEA-37579-stim
%load MEA-37079-stim
Fs=data.sampleRate;
MaxSetSize=5;
index=find(data.Cs<=20);data.Ts=data.Ts(index); data.Cs=data.Cs(index);
for hour=2:4; %:20
    disp('===============================================');
    disp('_______________________________________________');
    disp(['Analyzing hour ' num2str(hour) '...']);
    disp('_______________________________________________');
    disp('===============================================');
    tic
    index=find(data.Ts>(hour-1)*3600*Fs & data.Ts<=hour*3600*Fs);
    Ts=data.Ts(index); Cs=data.Cs(index); StimTimes=data.StimTimes(data.StimTimes>(hour-1)*3600 & data.StimTimes<=hour*3600)*Fs;
    for stim_shift=-1:1:5
        disp('===============================================');
        disp(['stim_shift = ' num2str(stim_shift/10) ' sec'])
        [Hx, MI_max, Set MIincrease(stim_shift+2,:)] = predInfo_best_newEntropy(Ts,Cs,StimTimes,round(Fs/10),stim_shift,MaxSetSize);
        while length(Set)<MaxSetSize
            Set=[Set Set(1)]; % add repeated channel if less than in other cases (just to obtain vectors of equal length...)
        end;
        MutInf(hour,stim_shift+2)=MI_max; 
        Neurons(hour,stim_shift+2,:)=Set;
    end;
    HX(hour,1)=Hx;
    toc
end
[Maxhour, MaxShift , ~]=size(Neurons)

save MutInf.txt MutInf -ascii
save EntropyStim.txt HX -ascii
save MI_increase.txt MIincrease -ascii
for hour=1:Maxhour
    A=double(squeeze(Neurons(hour,:,:))')
    eval(['save NeuronSet_h' num2str(hour) '.txt A -ascii']);
end;
    
