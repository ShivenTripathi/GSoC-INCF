function [HX, MI_max, best_neuron MIincrease] = predInfo_best_newEntropy(Ts,Cs,StimTimes,dt,stim_shift,neur_Max)
% neur_Max should be less than 21 ???
% what is stim_shift? latency for prediction; unit is bins


if max(StimTimes)<max(Ts)/1000
    disp('StimTimes is probably not expressed in samplenumber (but is seconds?)');
    disp('Make sure that the unit of StimTimes is samplenumbers');
    return;
end;
Tmax = max(Ts(length(Ts)),StimTimes(length(StimTimes))); % total time in sample numbers
nmax = double(round(Tmax/dt)); % number of bins

%% Entropy of Stimtimes
disp('Calculating entropy(StimTimes)...');
xs = []; 
for i=stim_shift+1:nmax   % Create binned signal
    mask_stim = (StimTimes>=i*dt).*(StimTimes<(i+1)*dt);
    foo_x = min(sum(mask_stim),1);  % 0 or 1, depending on whether there was a stimulus in that time bin
    xs = [xs foo_x]; %  binned signal
end
Xs=[length(find(~xs)) length(find(xs))];
HX = Entropy_millermaddow(Xs,2);
disp(['Entropy(StimTimes)=' num2str(HX)]); % value not to be trusted
%% choose the best neuron first
disp('Calculatimg mutual information between individual electrodes and StimTimes...');

best_neuron = [];
MI_max = 0;

for j=0:max(Cs)    % Cs might be 0...
    ys = [];
    xys = [];
    for i=1:nmax-stim_shift
        neur=Cs( (Ts>=i*dt) & (Ts<(i+1)*dt) ); % List of neurons that fired per time bin
        foo_y = ismember(j,neur); % did the specified neuron fire in the specified time bin?
        ys = [ys foo_y]; % one ys for each Cs, BINNED SIGNAL. 1 if that neuron fired in a time bin, 0 otherwise
        xys = [xys [xs(i); foo_y]];
    end
    channel=j;
    Ys=[length(find(~ys)) length(find(ys))];
    [HY] = Entropy_millermaddow(Ys,2);
    XYs=[length(find(~xs&~ys)) length(find(xs&~ys));length(find(~xs&ys)) length(find(xs&ys))];
    [HXY] = Entropy_millermaddow(XYs,4);

    if j==0; disp('Elct entropy    jointEnt     MI'); end;
    mi(j+1) = HX + HY - HXY;
    disp([num2str(j) ' ' num2str(HY) ' ' num2str(HXY) ' ' num2str(mi(j+1))]);
    if mi(j+1)>MI_max
        best_neuron = j;
        MI_max = mi(j+1);
    end
end
MIincrease(1)=MI_max; % vector to keep track of added value of additional neurons


%% then choose subsequent neurons
varMI_max = 0;
for j=2:neur_Max
    disp('Try addition of other neurons...')
    % figure out which neuron to add
    % best_new_neuron = 0;
    disp('_________________________________________________');
    disp(['Current pool of best neurons: ' num2str(best_neuron,0)]);
    mappingIndex=(1:neur_Max);
    MI_increased=false; % boolean to make iteration stop if MI did not increase anymore before max set size was reached
    for k=0:max(Cs)
        if ~ismember(k,best_neuron)
            disp(['Evaluating channel ' num2str(k) '...']);
            ys = [];
            xys = [];
            for i=1:nmax-stim_shift
                neur=Cs( (Ts>=i*dt) & (Ts<(i+1)*dt) );
                % pull out the part of neur that is best_neuron + k
                allowed_neurons = [best_neuron k];
                neur = (intersect(allowed_neurons,neur)); % Neurons under consideration that fired in the specified bin (still with channel number)
                neur_indexed=ismember(allowed_neurons,neur);
                neur_indexed=mappingIndex(neur_indexed);
                foo_y = sum(2.^(neur_indexed-1)); % number, encodes combination of neurons under consideration that fired in specified bin
                ys = [ys foo_y]; % binned signal for all neurons under consideration
            end
            for n=0:2^length(allowed_neurons)-1
                Ys(n+1)=length(find(ys==n));
                XYs(1,n+1)=length(find(~xs & ys==n));
                XYs(2,n+1)=length(find( xs & ys==n));
            end;
            HY = Entropy_millermaddow(Ys,2^j);
            HXY = Entropy_millermaddow(XYs,2^(j+1));
            mi = HX + HY - HXY;
            disp(['MI = ' num2str(mi)]);
            if mi>MI_max
                best_new_neuron = k;
                MIincrease(j)=mi-MI_max; % vector to keep track of added value of additional neurons
                MI_max = mi;
                MI_increased=true;
                
            end
        end
    end
    best_neuron = unique([best_neuron best_new_neuron]);
    if ~MI_increased
        return
    end
end