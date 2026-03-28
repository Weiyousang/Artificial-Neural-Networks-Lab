clc; clear; close all;

%% ====== Settings ======
digit_labels = {'1','3','5','7','9'};
noise_levels   = 0:0.1:0.9;
missing_levels = 0:0.1:0.9;
max_cycles = 20;

%% ====== DIGITS DEFINITIONS ======
digits = {
    [ -1 -1 -1  1  1  1 -1 -1 -1 -1;
      -1 -1  1  1  1  1  1 -1 -1 -1;
      -1  1  1  1  1  1  1 -1 -1 -1;
      -1 -1 -1  1  1  1  1 -1 -1 -1;
      -1 -1 -1  1  1  1  1 -1 -1 -1;
      -1 -1 -1  1  1  1  1 -1 -1 -1;
      -1 -1 -1  1  1  1  1 -1 -1 -1;
      -1 -1 -1  1  1  1  1 -1 -1 -1;
      -1  1  1  1  1  1  1  1  1 -1;
      -1  1  1  1  1  1  1  1  1 -1; ];

    [ -1 -1 -1  1  1  1  1  1  1 -1;
      -1 -1  1  1  1  1  1  1  1  1;
      -1 -1  1  1 -1 -1 -1 -1  1  1;
      -1 -1 -1 -1 -1 -1 -1 -1  1  1;
      -1 -1 -1 -1 -1  1  1  1  1 -1;
      -1 -1 -1 -1 -1  1  1  1  1 -1;
      -1 -1 -1 -1 -1 -1 -1 -1  1  1;
      -1 -1  1  1 -1 -1 -1 -1  1  1;
      -1 -1  1  1  1  1  1  1  1  1;
      -1 -1 -1  1  1  1  1  1  1 -1; ];

    [  1  1  1  1  1  1  1  1 -1 -1;
       1  1  1  1  1  1  1  1 -1 -1;
       1  1 -1 -1 -1 -1 -1 -1 -1 -1;
       1  1 -1 -1 -1 -1 -1 -1 -1 -1;
       1  1  1  1  1  1  1 -1 -1 -1;
       1  1  1  1  1  1  1  1 -1 -1;
      -1 -1 -1 -1 -1 -1  1  1 -1 -1;
       1  1 -1 -1 -1 -1  1  1 -1 -1;
       1  1  1  1  1  1  1  1 -1 -1;
      -1  1  1  1  1  1  1 -1 -1 -1; ];

    [ -1  1  1  1  1  1  1  1  1 -1;
      -1  1  1  1  1  1  1  1  1 -1;
      -1  1  1 -1 -1 -1  1  1 -1 -1;
      -1 -1 -1 -1 -1  1  1 -1 -1 -1;
      -1 -1 -1 -1 -1  1  1 -1 -1 -1;
      -1 -1 -1 -1  1  1  1 -1 -1 -1;
      -1 -1 -1 -1  1  1 -1 -1 -1 -1;
      -1 -1 -1 -1  1  1 -1 -1 -1 -1;
      -1 -1 -1 -1  1  1 -1 -1 -1 -1;
      -1 -1 -1 -1  1  1 -1 -1 -1 -1; ];

    [ -1 -1 -1 -1  1  1  1  1  1 -1;
      -1 -1 -1  1  1  1  1  1  1  1;
      -1 -1 -1  1  1 -1 -1 -1  1  1;
      -1 -1 -1  1  1 -1 -1 -1  1  1;
      -1 -1 -1  1  1 -1 -1 -1  1  1;
      -1 -1 -1  1  1  1  1  1  1  1;
      -1 -1 -1 -1  1  1  1  1  1  1;
      -1 -1 -1 -1 -1 -1 -1 -1  1  1;
      -1 -1 -1  1  1  1  1  1  1  1;
      -1 -1 -1 -1  1  1  1  1  1 -1; ];
};

%% Flatten digits
nPatterns = numel(digits);
[nRows, nCols] = size(digits{1});
nPixels = nRows*nCols;

patterns = zeros(nPixels,nPatterns);
for i = 1:nPatterns
    patterns(:,i) = -digits{i}(:);
end

%% Hopfield weight matrix
W = patterns * patterns.'; 
W = W - diag(diag(W));

%% Table storage
eTable = table( ...
    'Size',[0 7], ...
    'VariableTypes',{'string','string','double','double','double','double','double'}, ...
    'VariableNames',{'Digit','Mode','Level','Accuracy','Success','FinalEnergy','Cycles'});

rowCount = 0;

%% ====== Noise Sweep ======

trialCount = 20;   % 每種 level 做 20 次擾動實驗

for p = 1:nPatterns
    for nl = noise_levels

        xo = patterns(:,p);  % 原圖向量

        successCounter = 0;      % 成功次數（完全100%）
        accList = zeros(trialCount,1);   % 每次的 accuracy
        energyList = zeros(trialCount,1); % 每次的 final energy

        for k = 1:trialCount

            % 執行一次擾動 + recall
            [xd, xr, ev] = add_noise_and_recall_EXT(xo, W, nl, max_cycles);

            % 計算 bit 還原率
            acc = 100 * sum(xr == xo) / nPixels;
            accList(k) = acc;

            % 成功 = bit 全部還原
            if acc == 100
                successCounter = successCounter + 1;
            end

            % 記錄最終能量
            energyList(k) = ev(end);

        end

        % 建立 eTable 紀錄
        rowCount = rowCount + 1;
        eTable(rowCount,:) = {digit_labels{p}, "noise", nl, ...
                              mean(accList), ...        % 平均 accuracy
                              successCounter, ...        % 成功次數
                              mean(energyList), ...      % 平均最終能量
                              max_cycles};               % 固定 cycles
    end
end



%% ====== Missing Sweep ======

for p = 1:nPatterns
    for ml = missing_levels

        xo = patterns(:,p);

        successCounter = 0;
        accList = zeros(trialCount,1);
        energyList = zeros(trialCount,1);

        for k = 1:trialCount

            [xd, xr, ev] = add_missing_and_recall_EXT(xo, W, ml, max_cycles);

            acc = 100 * sum(xr == xo) / nPixels;
            accList(k) = acc;

            if acc == 100
                successCounter = successCounter + 1;
            end

            energyList(k) = ev(end);

        end

        rowCount = rowCount + 1;
        eTable(rowCount,:) = {digit_labels{p}, "missing", ml, ...
                              mean(accList), ...
                              successCounter, ...
                              mean(energyList), ...
                              max_cycles};
    end
end

writetable(eTable,'result.csv');
disp(eTable);

%% Make folder
if ~exist('RepresentativeCases','dir')
    mkdir RepresentativeCases;
end

%% ===== (A) Individual digits plots (unchanged) =====

for p = 1:nPatterns
    digit = digit_labels{p};
    rowsN = eTable(eTable.Digit==digit & eTable.Mode=="noise",:);
    rowsM = eTable(eTable.Digit==digit & eTable.Mode=="missing",:);

    % max success noise
    if max(rowsN.Success)>0
        lv = rowsN.Level(rowsN.Success==max(rowsN.Success));
        lv = lv(end);
        plot_case(digit,"noise",lv,W,patterns,max_cycles,nRows,nCols);
    end

    % max success missing
    if max(rowsM.Success)>0
        lv = rowsM.Level(rowsM.Success==max(rowsM.Success));
        lv = lv(end);
        plot_case(digit,"missing",lv,W,patterns,max_cycles,nRows,nCols);
    end
end

%% ===== (B) OUTPUT 20 AllDigits images  =====

% === noise 10 levels ===
for lv = noise_levels
    plot_all_digits_SINGLE_LEVEL("noise", lv, W, patterns, max_cycles, nRows, nCols);
end

% === missing 10 levels ===
for lv = missing_levels
    plot_all_digits_SINGLE_LEVEL("missing", lv, W, patterns, max_cycles, nRows, nCols);
end



%% ---------- FUNCTIONS ----------

function [xd,xr,ev,cy] = add_noise_and_recall_EXT(x,W,lv,max_cycles)
    n=numel(x);
    k=round(lv*n);
    xd=x;
    xd(randperm(n,k)) = -xd(randperm(n,k));
    [xr,ev,cy] = hopfield_recall_EXT(W,xd,max_cycles);
end

function [xd,xr,ev,cy] = add_missing_and_recall_EXT(x,W,lv,max_cycles)
    n=numel(x);
    k=round(lv*n);
    xd=x;
    xd(randperm(n,k))=0;
    [xr,ev,cy] = hopfield_recall_EXT(W,xd,max_cycles);
end

function [x,ev,cy] = hopfield_recall_EXT(W,x0,max_cycles)
    x=x0(:);
    ev=zeros(max_cycles+1,1);
    ev(1) = -0.5*x'*W*x;
    for t=1:max_cycles
        v=W*x;
        x(v>0)=1;
        x(v<0)=-1;
        ev(t+1) = -0.5*x'*W*x;
    end
    cy=max_cycles;
end

function plot_case(digit,mode,lv,W,patterns,max_cycles,nRows,nCols)
    idxDigit = strcmp(digit,{'1','3','5','7','9'});
    xo = patterns(:,idxDigit);

    if mode=="noise"
        [xd,xr,ev] = add_noise_and_recall_EXT(xo,W,lv,max_cycles);
    else
        [xd,xr,ev] = add_missing_and_recall_EXT(xo,W,lv,max_cycles);
    end

    f=figure('Visible','off');
    subplot(2,2,1); show_pattern(xo,nRows,nCols); title(['Orig ',digit]);
    subplot(2,2,2); show_pattern(xd,nRows,nCols); title([char(mode),' ',num2str(lv)]);
    subplot(2,2,3); show_pattern(xr,nRows,nCols); title(['Recall ',digit]);
    subplot(2,2,4); plot(ev,'-o'); grid on; title('Energy');

    saveas(f, fullfile('RepresentativeCases', ...
        sprintf('%s_%s_%d.png', digit, mode, round(lv*100))));
    close(f);
end

%% === One AllDigits per level ===
function plot_all_digits_SINGLE_LEVEL(mode,lv,W,patterns,max_cycles,nRows,nCols)

    digit_labels={'1','3','5','7','9'};

    f=figure('Visible','off','Position',[50,50,1600,900]);

    for p=1:5
        xo = patterns(:,p);

        if mode=="noise"
            [xd,xr,ev] = add_noise_and_recall_EXT(xo,W,lv,max_cycles);
        else
            [xd,xr,ev] = add_missing_and_recall_EXT(xo,W,lv,max_cycles);
        end

        subplot(4,5,p)
        show_pattern(xo,nRows,nCols)
        title(['Orig ',digit_labels{p}])

        subplot(4,5,5+p)
        show_pattern(xd,nRows,nCols)
        title([mode,' ',num2str(lv)])

        subplot(4,5,10+p)
        show_pattern(xr,nRows,nCols)
        title(['Recall ',digit_labels{p}])

        subplot(4,5,15+p)
        plot(ev,'-o'); grid on
        title('Energy')
        xlabel('Cycle'); ylabel('Energy');
    end

    saveas(f, fullfile('RepresentativeCases', ...
        sprintf('%s_allDigits_%d.png', mode, round(lv*100))));
    close(f);
end

function show_pattern(x,nRows,nCols)
    imshow(reshape(x,nRows,nCols),[]);
end
