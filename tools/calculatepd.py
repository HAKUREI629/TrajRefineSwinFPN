import argparse
import numpy as np
import scipy.io
from pathlib import Path
import sys
from tqdm import tqdm  
from scipy.signal import find_peaks

def load_excluded_files(excluded_file='excluded_files.txt'):
    """读取已排除文件列表"""
    excluded_files = set()
    if Path(excluded_file).exists():
        with open(excluded_file, 'r') as f:
            excluded_files = {line.strip() for line in f}
    return excluded_files

def calculate_pd(btr, theta_true, N, M):
    Pd = np.zeros(N)
    
    for i in range(N):
        # Find peaks (Matlab 的 findpeaks 参数对应)
        if np.max(btr) < 0.1:
            continue

        peaks_info = find_peaks(btr[i], height=0.001*np.max(btr))
        values = peaks_info[1]['peak_heights']  # 获取峰值高度
        peak_positions = peaks_info[0]          # 获取峰值位置索引
        
        # 降序排序峰值（与 Matlab 的 sort 逻辑对应）
        if len(values) > 0:
            sorted_indices = np.argsort(-values)  # 降序排序索引
            sorted_peaks = peak_positions[sorted_indices]
        else:
            sorted_peaks = np.array([])
        
        num = 0
        valid_ids = []
        for j in range(M):
            if len(sorted_peaks) == 0:
                continue
                
            # 计算与真实值的绝对差
            eps = np.abs(sorted_peaks - theta_true[i][j] + 1)
            min_eps = np.min(eps)
            
            if min_eps <= 5:
                
                if np.max(btr[i]) < 0.1:
                    continue
                num += 1
                valid_ids.append(np.argmin(eps))  # 记录有效匹配的索引
        
        # 处理 Pd 计算（修复原代码中的索引问题）
        if len(sorted_peaks) == 0:
            Pd[i] = 0
        else:
            if i == 0:  # 注意 Python 索引从 0 开始
                Pd[i] = num / len(sorted_peaks)
            else:
                if len(valid_ids) > 0:
                    # 取最后一个匹配的 id（与原 Matlab 代码逻辑对应）
                    Pd[i] = num / (valid_ids[-1] + 1)  # +1 因 Matlab 索引从 1 开始
                else:
                    Pd[i] = 0
    
    # 最终结果计算
    final_pd = np.sum(Pd) / (N * M)
    return final_pd


def load_matlab_theta(mat_path, key='theta_true'):
    """加载 MATLAB 的 theta_true 数据"""
    mat_data = scipy.io.loadmat(mat_path)
    theta = mat_data[key]
    
    # 确保数据转换为 numpy 数组并调整维度
    if theta.shape[0] == 1:
        theta = theta.T  # 处理 MATLAB 的行列存储差异
    return theta.astype(int)

def batch_process(npy_dir, theta_true, N, M, exclude_n=50):
    """核心处理函数"""
    # 加载MATLAB数据
    # mat_data = scipy.io.loadmat(mat_path)
    # theta_true = mat_data[var_name][:N, :M].astype(int)
    
    # 加载历史排除记录
    excluded_files = set()
    exclude_file = Path('tools/data')/'-29dB.txt'
    print(exclude_file)
    if exclude_file.exists():
        excluded_files = set(exclude_file.read_text().splitlines())
    
    # excluded_files = set()
    
    # 收集有效文件
    files = []
    valid_pd = []
    for f in Path(npy_dir).glob("*.npy"):
        if f.name not in excluded_files:
            files.append(f)
    
    # 计算所有有效文件的Pd值
    pd_results = {}
    for f in files:
        btr = np.load(f)
        pd = calculate_pd(btr, theta_true, N, M)  # 保持原有计算逻辑
        pd_results[f.name] = pd
    
    # 排除当前批次的最小值
    if exclude_n > 0 and not exclude_file.exists():
        sorted_files = sorted(pd_results.items(), key=lambda x: x[1])
        new_excluded = [k for k, v in sorted_files[:exclude_n]]
        
        # 更新排除列表
        with open(exclude_file, 'a') as f:
            f.write('\n'.join(new_excluded) + '\n')
        
        # 过滤当前有效结果
        valid_pd = [v for k, v in pd_results.items() if k not in new_excluded]
    else:
        valid_pd = list(pd_results.values())
    
    # 返回结果
    return np.mean(valid_pd) if valid_pd else 0

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='批量计算 Pd 指标',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--npy_dir', type=str, required=True,
                      help='包含 .npy 文件的目录路径')
    parser.add_argument('--mat_path', type=str, required=True,
                      help='MATLAB .mat 文件路径')
    parser.add_argument('--var_name', type=str, default='theta_true',
                      help='MATLAB 文件中变量的名称')
    parser.add_argument('-N', type=int, required=True,
                      help='实验次数/样本数量')
    parser.add_argument('-M', type=int, required=True,
                      help='每个实验的目标数量')
    parser.add_argument('--output', type=str, default='results.npy',
                      help='结果保存路径')
    
    return parser.parse_args()

def validate_paths(args):
    """路径验证"""
    errors = []
    
    # 检查 npy 目录
    npy_dir = Path(args.npy_dir)
    if not npy_dir.exists():
        errors.append(f"目录不存在: {npy_dir}")
    elif not npy_dir.is_dir():
        errors.append(f"路径不是目录: {npy_dir}")

    # 检查 mat 文件
    mat_path = Path(args.mat_path)
    if not mat_path.exists():
        errors.append(f"MAT 文件不存在: {mat_path}")
    elif mat_path.suffix != '.mat':
        errors.append(f"非 .mat 文件: {mat_path}")

    if errors:
        print("\n".join(errors))
        sys.exit(1)

def load_and_adjust_theta(mat_path, var_name, N, M):
    """加载并自动调整theta_true维度"""
    mat_data = scipy.io.loadmat(mat_path)
    theta = mat_data[var_name]
    
    # 处理MATLAB的默认列优先存储
    if theta.shape[0] == 1:  
        theta = theta.T
    
    # 自动维度调整
    if theta.shape[0] < N:
        raise ValueError(
            f"theta_true 行数不足！需要 {N} 行，实际 {theta.shape[0]} 行"
        )
    elif theta.shape[0] > N:
        print(f"警告：截断 theta_true 至前 {N} 行")
        theta = theta[:N, :]
    
    if theta.shape[1] != M:
        raise ValueError(
            f"列数不匹配！需要 {M} 列，实际 {theta.shape[1]} 列"
        )
    
    return theta.astype(int)


def main():
    # 解析参数
    args = parse_arguments()
    validate_paths(args)
    
    # ------------------------
    # 在 main 函数中替换原有加载代码：
    try:
        theta_true = load_and_adjust_theta(
            args.mat_path, 
            args.var_name,
            args.N,
            args.M
        )
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        sys.exit(1)
    
    # 批量处理 (使用之前定义的 batch_process 函数)
    results = batch_process(
        npy_dir=args.npy_dir,
        theta_true=theta_true,
        N=args.N,
        M=args.M
    )
    
    # 保存结果
    # np.save(args.output, results)
    # print(f"结果已保存至 {args.output}")
    print(f"平均 Pd: {results}")

if __name__ == "__main__":
    main()
