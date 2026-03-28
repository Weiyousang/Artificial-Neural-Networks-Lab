close; clc; clear;

%% ================== 資料產生 ==================
% 400 筆資料：前 300 train，後 100 test
x1 = -5 + 10 * rand(400, 1);
x2 = -5 + 10 * rand(400, 1);
xin = [x1, x2];
dn  = x1.^2 + x2.^2;

% 期望輸出正規化到 [0.2, 0.8]
d_min = min(dn);
d_max = max(dn);
d = (dn - d_min) * 0.6 / (d_max - d_min) + 0.2;

% 網格（畫 surface / gaussian 用）
x = linspace(-5, 5, 101);
y = linspace(-5, 5, 101);
[X, Y] = meshgrid(x, y);

% 期望 surface（沒正規化，只畫一次用）
[x1_grid, x2_grid] = meshgrid(linspace(-5, 5, 100), linspace(-5, 5, 100));
y_desired = x1_grid.^2 + x2_grid.^2;

%% ================== 固定迭代，掃 J 和 η ==================
iterations = 8000;             % 固定迭代次數 n
W0_range  = 0.3;               % W(0) ~ U[-0.3,0.3]

J_list   = [3 5 10 15 20 25];          % 想試的 hidden neurons 數 J
eta_list = [0.001 0.01 0.1 1];   % 想試的 learning rate

run_id  = 0;
results = [];                  % [J, eta, Etrain_final, Etest]

for J = J_list
    for eta = eta_list

        fprintf('\n===== Run with J=%d, eta=%.4f, iter=%d =====\n', J, eta, iterations);

        %% ---- 權重 / 中心 / 標準差初始化 ----
        w_bias   = -W0_range + 2*W0_range * rand;
        w_output = -W0_range + 2*W0_range * rand(J, 1);

        % center 沿對角線均勻分布在 [-5,5]
        mj1 = linspace(-5, 5, J + 2);
        mj1 = mj1(2:end-1);        % 去掉兩端
        mj  = [mj1; mj1];          % 2 x J

        sigmaJ = ones(1, J) * sqrt(2);

        % 初始 Gaussian basis（之後和訓練後比較）
        Gauss_init = calGauss(X, Y, mj, sigmaJ);

        %% ================== 訓練 ==================
        Etrain_av = zeros(iterations, 1);
        for n = 1:iterations
            Etrain = zeros(300, 1);
            for t = 1:300                 % 前 300 筆當訓練資料
                [yj, Y_out] = forwardPass(xin(t, :), mj, sigmaJ, w_output, w_bias);
                e = d(t) - Y_out;
                Etrain(t) = 0.5 * e^2;

                [w_output, w_bias, mj, sigmaJ] = ...
                    backwardPass(xin(t, :), yj, e, ...
                                 w_output, w_bias, mj, sigmaJ, eta);
            end
            Etrain_av(n) = mean(Etrain);

            if mod(n, 1000) == 0
                fprintf('  iter %5d / %5d, Train MSE = %.10f\n', n, iterations, Etrain_av(n));
            end
        end
        Etrain_final = Etrain_av(end);

        %% ================== 測試 ==================
        Etest = zeros(100, 1);
        for k = 1:100
            t = 300 + k;     % 301~400 當 test
            [~, Y_out] = forwardPass(xin(t, :), mj, sigmaJ, w_output, w_bias);
            e = d(t) - Y_out;
            Etest(k) = 0.5 * e^2;
        end
        Etest_av = mean(Etest);

        fprintf('===> J=%2d, eta=%.4f | Final Train MSE = %.10f, Test MSE = %.10f\n', ...
                J, eta, Etrain_final, Etest_av);

        run_id = run_id + 1;
        results(run_id, :) = [J, eta, Etrain_final, Etest_av];

        %% ================== 產生 surface 與 Gaussian（用訓練後的參數） ==================
        % RBF 在網格上的輸出（反正規化回原始 y）
        testing_output  = zeros(size(X));
        training_output = zeros(size(X));   

        for i = 1:size(X, 1)
            for j = 1:size(X, 2)
                [~, normalized_output] = forwardPass([X(i, j), Y(i, j)], mj, sigmaJ, w_output, w_bias);
                val = (normalized_output - 0.2) * (d_max - d_min) / 0.6 + d_min;
                testing_output(i, j)  = val;
                training_output(i, j) = val;   
            end
        end
        Gauss_final = calGauss(X, Y, mj, sigmaJ);

        % 檔名
        eta_str  = strrep(sprintf('%.4f', eta), '.', 'p');
        baseName = sprintf('J%d_eta%s_iter%d', J, eta_str, iterations);

        %% ================== 圖 1：「3-in-1」Loss + Test + Train ==================
        fig1 = figure('Visible','off');
        set(fig1, 'Position', [100 100 1600 500]);  

        % 1. Training loss
        subplot(1,3,1);
        plot(1:iterations, Etrain_av, 'LineWidth',1.5);
        grid on;
        xlabel('Iteration');
        ylabel('MSE');
        title('Training loss');

        % 2. Testing output surface
        subplot(1,3,2);
        mesh(X, Y, testing_output);
        xlabel('x_1'); ylabel('x_2'); zlabel('y');
        title('Testing output surface');

        % 3. Training output surface
        subplot(1,3,3);
        mesh(X, Y, training_output);
        xlabel('x_1'); ylabel('x_2'); zlabel('y');
        title('Training output surface');

        sgtitle(sprintf('J=%d, \\eta=%.4f, iter=%d', J, eta, iterations));

        print(fig1, [baseName '_3in1.png'], '-dpng', '-r300');
        close(fig1);

        %% ================== 圖 2：Gaussian basis 初始 vs 訓練後 ==================
        fig2 = figure('Visible','off');
        set(fig2, 'Position', [200 200 1200 500]);

        subplot(1,2,1);
        mesh(X, Y, Gauss_init);
        xlabel('x_1'); ylabel('x_2');
        title('Initial Gaussian basis');

        subplot(1,2,2);
        mesh(X, Y, Gauss_final);
        xlabel('x_1'); ylabel('x_2');
        title('Final Gaussian basis');

        sgtitle(sprintf('Gaussian basis, J=%d, \\eta=%.4f, iter=%d', J, eta, iterations));

        print(fig2, [baseName '_gaussian.png'], '-dpng', '-r300');
        close(fig2);

    end
end

%% ================== 結果表格 ==================
ResultsTable = array2table(results, ...
    'VariableNames', {'J','eta','Etrain_final','Etest'});
disp('============ Summary over (J, eta) ============');
disp(ResultsTable);

%% ================== 只畫 1 次 Desired surface==================
figure;
mesh(x1_grid, x2_grid, y_desired);
xlabel('x_1'); ylabel('x_2'); zlabel('y');
title('Desired surface: y = x_1^2 + x_2^2');
grid on;

%% ================== local functions ==================
function Gauss = calGauss(X, Y, mj, sigmaJ)     
    [rows, cols] = size(X);
    hiddensz = size(mj, 2);    
    Gauss = zeros(rows, cols);
    for i = 1:rows
        for j = 1:cols
            XY = [X(i, j); Y(i, j)];
            for k = 1:hiddensz
                D = XY - mj(:, k);
                g = exp(-(D(1)^2 + D(2)^2) / (2 * sigmaJ(k)^2));
                Gauss(i, j) = max(Gauss(i, j), g);
            end
        end
    end
end

function [yj, Y] = forwardPass(x, mj, sigmaJ, w_output, w_bias)
    hiddensz = size(mj, 2);
    yj = zeros(hiddensz, 1);
    for j = 1:hiddensz
        D = x' - mj(:, j);
        yj(j) = exp(-sum(D.^2) / (2 * sigmaJ(j)^2));
    end
    Y = w_bias + sum(w_output .* yj);
end

function [w_output, w_bias, mj, sigmaJ] = backwardPass(x, yj, e, ...
    w_output, w_bias, mj, sigmaJ, learningrate)

    hiddensz = size(mj, 2);
    delta_w     = learningrate * e * yj;
    delta_bias  = learningrate * e;
    delta_mj    = zeros(size(mj));
    delta_sig_j = zeros(size(sigmaJ));

    for j = 1:hiddensz
        D = x' - mj(:, j);
        delta_mj(:, j) = learningrate * e * w_output(j) * yj(j) .* D / sigmaJ(j)^2;
        delta_sig_j(j) = learningrate * e * w_output(j) * yj(j) * sum(D.^2) / sigmaJ(j)^3;
    end

    w_output = w_output + delta_w;
    w_bias   = w_bias   + delta_bias;
    mj       = mj       + delta_mj;
    sigmaJ   = sigmaJ   + delta_sig_j;
end
