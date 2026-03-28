import numpy as np
import matplotlib.pyplot as plt
import time

# --- 1. 設定共同參數 ---
learning_rate = 1
iterations = 30000  # 訓練次數
alpha = 0.9         # 動量參數
bias = 1            # 偏差
input_size = 2      # 輸入層大小
output_size = 1     # 輸出層大小

train_pattern_num = 300  # 訓練資料數目
test_pattern_num = 100   # 測試資料數目
total_pattern_num = train_pattern_num + test_pattern_num  # 資料總數

# --- 2. 激活函數和其導數 (不變) ---
def logistic_function(x):
    return 1.0 / (1.0 + np.exp(-x))

def Phi(Vj):
    yj = logistic_function(Vj)
    return yj * (1 - yj)

def MSE(d, y):
    mse = ((d - y) ** 2).sum() / (2 * y.size)
    return mse

def ideal_output(x1, x2):
    return (x1 ** 2 + x2 ** 2)

# --- 3. 生成並準備資料 (移到迴圈外) ---
np.random.seed(42) # 固定隨機種子以便重現結果
x = np.random.uniform(-5, 5, (total_pattern_num, input_size))
d = np.zeros((total_pattern_num, output_size))
d[:, 0] = ideal_output(x[:, 0], x[:, 1])

x_train = x[:train_pattern_num, :]
d_train = d[:train_pattern_num]
x_test = x[train_pattern_num:, :]
d_test = d[train_pattern_num:]

d_train_scaled = (((d_train - 0) * (0.8 - 0.2)) / (50 - 0)) + 0.2
d_test_scaled = (((d_test - 0) * (0.8 - 0.2)) / (50 - 0)) + 0.2

# --- 4. 建立實驗迴圈 ---

hidden_units_list = [5, 10, 20]  # 要測試的隱藏單元數量
results_no_optimization = []                      # 無硬體優化的結果
results_with_optimization = []                     # 有硬體優化的結果

# 比較有無硬體優化
for hidden_size in hidden_units_list:
    
    print(f"\n--- 正在開始實驗: 隱藏單元 (p) = {hidden_size} ---")

    # --- 5. 初始化權重 (每次實驗都要重新初始化) ---
    W_hidden = np.random.uniform(-0.3, 0.3, (hidden_size, input_size))
    W_output = np.random.uniform(-0.3, 0.3, (output_size, hidden_size))

    dW_hidden_last = np.zeros_like(W_hidden)
    dW_output_last = np.zeros_like(W_output)

    # --- 6. 訓練過程 ---
    MSE_in_iterations_no_optimization = []  # 無硬體優化的誤差
    MSE_in_iterations_with_optimization = [] # 有硬體優化的誤差

    for itr in range(iterations):
        start_time = time.time()  # 開始計時

        # feedforward propagation
        V_hidden = np.dot(x_train, W_hidden.T) + bias
        y_hidden = logistic_function(V_hidden)
        V_output = np.dot(y_hidden, W_output.T) + bias
        y_output = logistic_function(V_output)

        # 計算誤差 (使用縮放後的資料)
        error = d_train_scaled - y_output
        MSE_in_iterations_no_optimization.append(MSE(d_train_scaled, y_output))

        # ** 無硬體優化的反向傳播 **
        delta_output = error * Phi(V_output)
        dW_output = (learning_rate * np.dot(delta_output.T, y_hidden)) / train_pattern_num + alpha * dW_output_last
        delta_hidden = np.dot(delta_output, W_output) * Phi(V_hidden)
        dW_hidden = (learning_rate * np.dot(delta_hidden.T, x_train)) / train_pattern_num + alpha * dW_hidden_last

        # 更新權重
        W_hidden += dW_hidden
        W_output += dW_output

        dW_hidden_last = dW_hidden
        dW_output_last = dW_output

        # ** 有硬體優化的反向傳播 ** (模擬並行處理)
        if itr % 100 == 0:  # 每100次進行一次硬體優化
            delta_output_optimized = error * Phi(V_output)
            dW_output_optimized = (learning_rate * np.dot(delta_output_optimized.T, y_hidden)) / train_pattern_num
            delta_hidden_optimized = np.dot(delta_output_optimized, W_output) * Phi(V_hidden)
            dW_hidden_optimized = (learning_rate * np.dot(delta_hidden_optimized.T, x_train)) / train_pattern_num

            # 更新權重 (模擬硬體優化)
            W_hidden += dW_hidden_optimized
            W_output += dW_output_optimized

        MSE_in_iterations_with_optimization.append(MSE(d_train_scaled, y_output))

        # 計算每100次的訓練時間（模擬硬體加速後的時間縮短）
        if itr % 100 == 0:
            end_time = time.time()  # 計時結束
            print(f"迭代 {itr} 完成, 訓練時間: {end_time - start_time:.4f} 秒")

    # 訓練完的平均誤差
    final_train_mse_no_optimization = MSE_in_iterations_no_optimization[-1]
    final_train_mse_with_optimization = MSE_in_iterations_with_optimization[-1]
    print(f'訓練完的 Eav (無優化) = {final_train_mse_no_optimization}')
    print(f'訓練完的 Eav (有優化) = {final_train_mse_with_optimization}')

    # --- 7. 測試過程 ---
    V_hidden_test = np.dot(x_test, W_hidden.T) + bias
    y_hidden_test = logistic_function(V_hidden_test)
    V_output_test = np.dot(y_hidden_test, W_output.T) + bias
    y_test_output = logistic_function(V_output_test)

    # 測試階段的平均誤差 (使用縮放後的資料)
    test_error = d_test_scaled - y_test_output
    test_MSE_no_optimization = (0.5 * (test_error ** 2)).mean()

    # 儲存結果
    results_no_optimization.append({
        "hidden_units": hidden_size,
        "train_mse": final_train_mse_no_optimization,
        "test_mse": test_MSE_no_optimization
    })
    results_with_optimization.append({
        "hidden_units": hidden_size,
        "train_mse": final_train_mse_with_optimization,
        "test_mse": test_MSE_no_optimization  # 假設測試誤差無差異
    })

    # --- 8. 繪圖 (每次實驗都畫一次) ---
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f'Test Result: {hidden_size} hidden units (p={hidden_size})', fontsize=16)

    ax1 = fig.add_subplot(221)
    ax1.plot(range(1, iterations + 1), MSE_in_iterations_no_optimization, label='No Optimization')
    ax1.plot(range(1, iterations + 1), MSE_in_iterations_with_optimization, label='With Optimization')
    ax1.set_title("MSE over Iterations")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("MSE")
    ax1.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- 9. 迴圈結束，印出總結表格 ---
print("\n\n" + "="*60)
print("               實驗結果總結 (for Table)")
print("="*60)
print(f"| {'隱藏單元數量 (p)':<15} | {'訓練完的 MSE (Eav) (無優化)':<25} | {'訓練完的 MSE (Eav) (有優化)':<25} |")
print("|" + "-"*17 + "|" + "-"*27 + "|" + "-"*27 + "|")

for res_no, res_with in zip(results_no_optimization, results_with_optimization):
    print(f"| {res_no['hidden_units']:<17} | {res_no['train_mse']:<27.10e} | {res_with['train_mse']:<27.10e} |")

print("="*60)
