import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.linalg import inv

def find_half_band_track_new_find(mu1, x_em, flag_trace, arg_thred2, x_em_v, max_angle):
    """等效于find_half_band_track_new_find.m"""
    mu = np.zeros_like(x_em)
    sigma = 2 * np.ones_like(x_em)
    flag = np.zeros_like(flag_trace)
    
    # 标记flag_trace中值为2的位置
    flag[flag_trace == 2] = 2
    
    # 筛选有效数据
    valid_mask = (flag_trace != 2)
    x_em_act = x_em[valid_mask]
    x_em_v_act = x_em_v[valid_mask]
    
    # 排序处理
    sort_idx = np.argsort(x_em_act)
    x_em_act = x_em_act[sort_idx]
    
    # 计算边界
    x_em_bound = np.concatenate([[1], x_em_act, [max_angle]])
    bounds = 0.5*(x_em_bound[:-1] + x_em_bound[1:])
    
    # 主处理循环
    mu_ob = np.zeros_like(x_em_act)
    for k_em in range(len(x_em_act)):
        left = max(bounds[k_em], x_em_act[k_em] - arg_thred2)
        right = min(bounds[k_em+1], x_em_act[k_em] + arg_thred2)
        
        # 寻找范围内的mu
        valid_mu = mu1[(mu1 > left) & (mu1 < right)]
        if valid_mu.size == 0:
            mu_ob[k_em] = x_em_act[k_em]
            flag[np.where(x_em == x_em_act[k_em])] = 1
        else:
            min_idx = np.argmin(np.abs(valid_mu - x_em_act[k_em]))
            mu_ob[k_em] = valid_mu[min_idx]
    
    # 映射回原始结构
    for k_em in range(len(x_em_act)):
        mu[x_em == x_em_act[k_em]] = mu_ob[k_em]
    
    return mu, sigma, flag

def find_peak_sigma(signals, y):
    """等效于find_peak_sigma.m"""
    index = np.where(signals == 1)[0]
    peaks, widths = [], []
    current_segment = []
    
    # 分割连续峰区间
    for i in index:
        if not current_segment:
            current_segment.append(i)
        else:
            if i == current_segment[-1] + 1:
                current_segment.append(i)
            else:
                peaks.append(current_segment)
                current_segment = [i]
    if current_segment:
        peaks.append(current_segment)
    
    # 计算初始峰位置和宽度
    peak_list, width_list = [], []
    for seg in peaks:
        width = len(seg)
        peak = seg[0] + width//2
        peak_list.append(peak)
        width_list.append(width)
    
    # 精确峰定位
    refined_peaks = []
    for peak, width in zip(peak_list, width_list):
        if width % 2 == 0:
            start = int(peak - width/2)
            end = int(peak + width/2)
        else:
            start = int(peak - (width+1)//2)
            end = int(peak + (width+1)//2)
        
        start = max(0, start)
        end = min(len(y)-1, end)
        window = y[start:end+1]
        refined_peaks.append(start + np.argmax(window))
    
    # 去重处理
    unique_peaks = np.unique(refined_peaks)
    final_widths = [width_list[refined_peaks.index(p)] for p in unique_peaks]
    
    return unique_peaks, np.array(final_widths)

def peak_detection_zscore(y, lag, threshold, influence, fil_len, thred_std):
    """等效于peak_detection_zscore.m"""
    signals = np.zeros(len(y) + lag + 1)
    filteredY = np.zeros_like(signals)
    avgFilter = np.zeros_like(signals)
    stdFilter = np.zeros_like(signals)
    
    # 初始化
    init_segment = y[-lag:]
    avgFilter[lag] = np.mean(init_segment)
    stdFilter[lag] = np.std(init_segment)
    filteredY[:lag] = init_segment
    
    # 主处理循环
    for i in range(lag+1, len(signals)):
        idx = i - lag - 1  # 对应原始数据索引
        
        # 检测峰值
        if (y[idx] - avgFilter[i-1]) > threshold * stdFilter[i-1]:
            signals[i] = 1
            
            # 动态窗口计算
            start = max(0, idx - fil_len)
            end = min(len(y)-1, idx + fil_len)
            window = np.concatenate([y[start:idx], y[idx+1:end+1]])
            
            if window.size > 0:
                avg = np.mean(window)
                std = np.std(window)
                if y[idx] < thred_std * std + avg:
                    signals[i] = 0
            
            # 更新滤波值
            filteredY[i] = influence * y[idx] + (1-influence)*filteredY[i-1]
        else:
            filteredY[i] = y[idx]
        
        # 更新统计量
        avgFilter[i] = np.mean(filteredY[i-lag:i+1])
        stdFilter[i] = np.std(filteredY[i-lag:i+1])
    
    return signals[lag+1:], avgFilter[lag+1:], stdFilter[lag+1:]

def trace_estimate(t_angle, step=3, threshold=1):
    # 初始化参数 ========================================
    a = t_angle
    N, signal_length = a.shape[0], a.shape[1]
    
    # 系统矩阵
    F = np.array([[1, 1], [0, 1]])  # 状态转移
    H = np.array([[1, 0]])          # 观测矩阵
    Q = 0.01 * np.eye(2)            # 过程噪声
    
    # 算法参数
    lag = 5 * step
    fil_len = 5 * step
    thred_std = 0.5
    # threshold = threshold
    influence = 1
    arg_thred1 = 3 * step
    arg_thred2 = 3 * step
    arg_thred3 = 3 * step

    # 状态初始化 ========================================
    max_tracks = 6000  # 最大跟踪目标数
    x_em = np.zeros((2, N, max_tracks))     # 状态向量 (position, velocity)
    P_em = np.zeros((2, 2, N, max_tracks))  # 协方差矩阵
    z_t = np.zeros((1, N, 3000))            # 观测值
    flag_t1 = np.zeros((1, N, max_tracks), dtype=int)  # 轨迹状态标记
    R1 = np.zeros((1, N, max_tracks))       # 观测噪声
    
    time_thirty_start = np.zeros(6000)       # 轨迹起始时间
    a_record1 = np.zeros_like(a)            # 轨迹记录
    N2 = np.zeros(N, dtype=int)             # 有效轨迹数
    peaks = np.zeros_like(a)                # 峰值标记
    
    # 主处理循环 ========================================
    for k1 in range(N):
        if k1 > 0:
            N2[k1] = N2[k1-1]
        
        # 阶段1: 峰值检测 ================================
        current_signal = a[k1, :]
        
        # 前向检测
        signals1, _, _ = peak_detection_zscore(current_signal, lag, threshold, 
                                             influence, fil_len, thred_std)
        # 反向检测
        signals2, _, _ = peak_detection_zscore(current_signal[::-1], lag, threshold,
                                             influence, fil_len, thred_std)
        signals = signals1 * signals2[::-1]
        signals[signals == 2] = 1
        signals[[0, -1]] = 0
        
        # 精确定位峰值
        mu1, sigma1 = find_peak_sigma(signals, current_signal)
        
        # 附加过滤条件（原始MATLAB第45-48行）
        valid_indices = []
        for idx, pos in enumerate(mu1):
            if current_signal[pos] >= 0.5 * np.mean(current_signal):
                valid_indices.append(idx)
        mu1 = mu1[valid_indices]
        peaks[k1, np.array(mu1, dtype=int)] = 1
        
        # 阶段2: 轨迹管理 ================================
        # 更新轨迹记录（对应MATLAB第52-60行）
        for pos in mu1:
            if k1 == 0:
                a_record1[k1, pos] = 1
            else:
                search_start = max(0, pos - arg_thred1)
                search_end = min(signal_length-1, pos + arg_thred1)
                a_record1[k1, pos] = 1 + np.max(a_record1[k1-1, search_start:search_end+1])
        
        # 创建新轨迹（对应MATLAB第63-81行）
        if len(mu1) > 0:
            for pos in mu1:
                conflict = False
                if N2[k1] > 0:
                    for track_id in range(N2[k1]):
                        if flag_t1[0, k1, track_id] == 2:
                            continue
                        if abs(pos - x_em[0, k1-1, track_id]) < arg_thred3:
                            conflict = True
                            break
                
                if not conflict and a_record1[k1, pos] > 1:
                    # 创建新轨迹
                    N2[k1] += 1
                    new_track = N2[k1] - 1  # Python从0开始索引
                    
                    # 状态初始化（简化版State_init）
                    x = np.array([pos, 0])  # 位置和速度
                    P = np.array([[1, 0], [0, 0.5]])
                    
                    # 回填历史数据（对应MATLAB第76-79行）
                    lookback = 1
                    for kk in range(lookback):
                        hist_k = k1 - 1
                        if hist_k >= 0:
                            x_em[0, hist_k, new_track] = x[0] - 1 * x[1]
                            P_em[:, :, hist_k, new_track] = P
                    
                    time_thirty_start[new_track] = k1
        
        # 阶段3: 轨迹更新 ================================
        if N2[k1] > 0:
            active_tracks = range(N2[k1])
            
            # 调用find_half_band_track_new_find
            mu_current = x_em[0, k1-1, active_tracks]
            flags_current = flag_t1[0, k1, active_tracks]
            x_em_v = x_em[1, k1-1, active_tracks]
            
            # 转换为MATLAB的find_half_band_track_new_find输入格式
            z_t_tmp, R1_tmp, flag_t1_tmp = find_half_band_track_new_find(
                mu1, mu_current, flags_current, arg_thred2, x_em_v, signal_length
            )
            
            # 更新状态（对应MATLAB第84-86行）
            z_t[0, k1, :N2[k1]] = z_t_tmp
            R1[0, k1, :N2[k1]] = R1_tmp
            flag_t1[0, k1, :N2[k1]] = flag_t1_tmp
        
        # 阶段4: 卡尔曼滤波更新 ==========================
        for track_id in range(N2[k1]):
            if flag_t1[0, k1, track_id] == 2:
                continue
            
            # 预测步骤
            x_pred = F @ x_em[:, k1-1, track_id]
            P_pred = F @ P_em[:, :, k1-1, track_id] @ F.T + Q
            
            # 更新步骤
            if z_t[0, k1, track_id] != 0:
                y_res = z_t[0, k1, track_id] - H @ x_pred
                S = H @ P_pred @ H.T + R1[0, k1, track_id]
                K = P_pred @ H.T @ inv(S)
                
                x_em[:, k1, track_id] = x_pred + K @ y_res
                P_em[:, :, k1, track_id] = (np.eye(2) - K @ H) @ P_pred
            else:
                x_em[:, k1, track_id] = x_pred
                P_em[:, :, k1, track_id] = P_pred
        
        
        # 阶段5: 轨迹终止判断 ============================
        for track_id in range(N2[k1]):
            start_time = time_thirty_start[track_id]
            if k1 > start_time:
                window = flag_t1[0, max(0,k1):k1+1, track_id]
                if np.sum(window == 1) >= 1:
                    flag_t1[0, k1:, track_id] = 2
                    x_em[0, k1, track_id] = -1000  # 无效标记
    
    
    return x_em[0, :, :N2[-1]], N2


if __name__ == "__main__":
    # 示例数据测试
    np.random.seed(42)
    test_data = np.random.rand(50, 512)  # 50个样本，每个样本512个点
    
    # 执行轨迹估计
    result_peaks, N2 = trace_estimate(test_data)
    
    print("跟踪结果维度:", result_peaks.shape)
    print("有效通道数:", N2[-1])
