import numpy as np
import scipy.io
from pathlib import Path
import argparse
from scipy.signal import find_peaks

def calculate_pd(btr, theta_true, N, M):
    Pd = np.zeros(N)
    
    for i in range(N):
        # Find peaks (Matlab 的 findpeaks 参数对应)
        peaks_info = find_peaks(btr[i], height=0.001*np.max(btr), distance=3)
        values = peaks_info[1]['peak_heights']  # 获取峰值高度
        peak_positions = peaks_info[0]          # 获取峰值位置索引
        
        # 降序排序峰值（与 Matlab 的 sort 逻辑对应）
        if len(values) > 0:
            sorted_indices = np.argsort(-values)  # 降序排序索引
            sorted_peaks = peak_positions[sorted_indices]
            sorted_values = values[sorted_indices]
        else:
            sorted_peaks = np.array([])
            sorted_values = np.array([])
        
        num = 0
        valid_ids = []
        for j in range(M):
            if len(sorted_peaks) == 0:
                continue
                
            # 计算与真实值的绝对差
            eps = np.abs(sorted_peaks - theta_true[i][j])
            min_eps = np.min(eps)
            
            if min_eps <= 5:
                if sorted_values[np.argmin(eps)] > np.mean(btr[i]):
                    num += 1
                    valid_ids.append(np.argmin(eps))  # 记录有效匹配的索引s
        
        # 处理 Pd 计算（修复原代码中的索引问题）
        if len(sorted_peaks) == 0:
            Pd[i] = 0
        else:
            if i == 0:  # 注意 Python 索引从 0 开始
                Pd[i] = num / len(sorted_peaks)
            else:
                if len(valid_ids) > 0:
                    # 取最后一个匹配的 id（与原 Matlab 代码逻辑对应）
                    weight = 1
                    for op in valid_ids:
                        weight *= sorted_values[op] / np.max(sorted_values)
                    Pd[i] = weight * num / (valid_ids[-1] + 1)  # +1 因 Matlab 索引从 1 开始
                else:
                    Pd[i] = 0
    
    # 最终结果计算
    final_pd = np.sum(Pd) / (N * M)
    return final_pd

def process_mat_files(mat_dir, theta_true, N, M):
    """处理目录下所有 .mat 文件"""
    mat_dir = Path(mat_dir)
    all_pd = []
    
    for mat_file in mat_dir.glob("*.mat"):
        # try:
        mat_data = scipy.io.loadmat(mat_file)['t_angle']
        pd_value = calculate_pd(mat_data, theta_true, N, M)
        all_pd.append(pd_value)
        # print(f"处理文件: {mat_file.name} | Pd: {pd_value:.4f}")
        # except Exception as e:
        #     print(f"错误 {mat_file.name}: {str(e)}")
    
    return np.mean(all_pd) if all_pd else 0.0

def load_matlab_theta(mat_path, key='theta_true'):
    """加载 MATLAB 的 theta_true 数据"""
    mat_data = scipy.io.loadmat(mat_path)
    theta = mat_data[key]
    
    # 确保数据转换为 numpy 数组并调整维度
    if theta.shape[0] == 1:
        theta = theta.T  # 处理 MATLAB 的行列存储差异
    return theta.astype(int)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理 MAT 文件计算平均 Pd')
    parser.add_argument('--mat_dir', required=True, help='包含 .mat 文件的目录')
    parser.add_argument('--truth', required=True, help='truth')
    parser.add_argument('-N', type=int, required=True, help='实验次数')
    parser.add_argument('-M', type=int, required=True, help='目标数量')
    args = parser.parse_args()

    theta_true = load_matlab_theta(args.truth, key='truth')

    final_pd = process_mat_files(args.mat_dir, theta_true, args.N, args.M)
    print(f"\n平均 Pd 值: {final_pd:.4f}")

# python calculatepd_old.py --mat_dir E:\\wyh\\kraken\\beam\\snr1\\MUSIC\\-29dB --truth E:\\wyh\\kraken\\beam\\truth.mat -N 512 -M 1 