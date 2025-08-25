import numpy as np
import scipy.io
from scipy.optimize import linear_sum_assignment
import argparse
from pathlib import Path
from trace_estimate import trace_estimate
from calculatepd_old import calculate_pd
import re
import pandas as pd
from tqdm import tqdm
from scipy.signal import find_peaks

def compute_sosnr(btr, theta_true, N, M):
    SOSNR = 0.0
    threshold = round(1 * 512 / 180)  # 计算固定阈值
    
    for i in range(N):
        btr_row = btr[i, :]
        idx, _ = find_peaks(btr_row)  # 找到峰值的索引
        idx = np.array(idx)  # 确保是numpy数组
        
        di = 0.0
        for j in range(M):
            target = theta_true[i, j]  # 注意索引是否从0开始
            
            # 计算所有峰值位置与目标的绝对差
            eps = np.abs(idx - target)
            if len(eps) == 0:
                # 如果没有找到峰值，直接使用目标值
                peak = int(target)
            else:
                min_eps = np.min(eps)
                if min_eps <= threshold:
                    peak_idx = np.argmin(eps)
                    peak = idx[peak_idx]
                else:
                    peak = int(theta_true[i, j])  # 原MATLAB代码可能存在的潜在错误
            
            # 确保peak不越界
            peak = max(0, min(peak, btr_row.shape[0]-1))
            
            if btr_row[peak] == 0:
                di += 1e-10
            else:
                di += btr_row[peak]
        
        total = np.sum(btr_row)
        if di == 0:
            SOSNR += 1e-10
        else:
            SOSNR += 10 * np.log10((512 - M) * di / M / (total - di))
    
    SOSNR = SOSNR / N
    return SOSNR


def evaluate_trace(btr, theta_true, threshold):
    """
    评估 DOA 估计轨迹的连续性和准确性
    参数:
        btr (np.ndarray): 估计的 DOA 数据矩阵 (N x T)
        theta_true (np.ndarray): 真实角度矩阵 (N x M)
    返回:
        tuple: (OSPA, RMSE) 评估指标
    """
    # 调用轨迹估计函数
    result, num = trace_estimate(btr, step=3, threshold=threshold)
    result = np.squeeze(result)  # 调整为 (N x num_tracks)
    result = result[:, :max(num)]  # 保留有效轨迹
    np.save('1.npy', result)
    
    # 定义评估时间窗口
    time_windows = [1, 32, 64, 128, 192, 256, 320, 384, 448]
    
    
    # 初始化评估指标
    ospa_list, error_matrix = [], np.full_like(theta_true, 999999, dtype=np.float32)
    pd_matrix = np.full_like(theta_true, 0.0, dtype=np.float32)

    Pd = np.zeros((len(time_windows), 1))
    PFA = np.zeros((len(time_windows), 1))
    cnt = 0
    
    # 多时间窗口评估
    for win_len in time_windows:
        # 筛选满足长度的轨迹
        valid_tracks = [i for i in range(result.shape[1]) 
                       if np.sum(result[:, i] > 0) > win_len]
        
        if not valid_tracks:
            break
        
        result_win = result[:, valid_tracks]
        result_win[result_win <= 0] = np.inf  # 无效值处理
        
        # 计算 OSPA 和误差矩阵
        ospa, _, pd, err_mat = calculate_ospa(theta_true.T, result_win.T, c=5, p=1)
        pfa = calculate_false_alarm_per_frame(theta_true, result, 4)
        ospa_list.append(ospa)
        error_matrix = np.minimum(error_matrix, err_mat.T)
        if cnt < 1:
            pd_matrix += np.where(err_mat.T < 4, 1, 0)

        # if pd > Pd:
        #     Pd = pd
        Pd[cnt] = pd
        PFA[cnt] = np.mean(pfa)
        cnt += 1
    
    # 综合评估指标
    pd_matrix = pd_matrix / 1
    valid_wins = len(ospa_list)
    # weights = (np.array(time_windows[:valid_wins])) / sum(np.array(time_windows[:valid_wins]))
    # ospa_final = np.dot(ospa_list, weights)
    # ospa_final = min(ospa_list)
    ospa_final = np.mean(ospa_list)
    
    # 计算 RMSE
    valid_errors = error_matrix[error_matrix < 999999]
    rmse_final = np.sqrt(np.mean(valid_errors)) if valid_errors.size > 0 else np.inf

    # DI = compute_sosnr(btr, theta_true, theta_true.shape[0], theta_true.shape[1])

    return result, ospa_final, rmse_final, Pd[0][0], PFA[0][0], pd_matrix

def calculate_false_alarm_per_frame(theta_true, result, c=4):
    """
    逐帧计算跟踪算法的虚警概率 (False Alarm Probability)

    参数:
        theta_true (np.ndarray): 真实目标矩阵，形状 (N, M)
        result     (np.ndarray): 跟踪结果矩阵，形状 (N, K)
        c          (float):     虚警阈值（阈值按 squared error 判断，默认 5）

    返回:
        np.ndarray: Pfa, 形状 (N,), 每帧的虚警概率
    """
    N, M = theta_true.shape
    N2, K = result.shape
    assert N2 == N, "theta_true 与 result 必须具有相同帧数 (行数)"

    Pfa = np.zeros(N, dtype=float)

    for t in range(N):
        # 提取第 t 帧
        true_vals = theta_true[t, :]
        true_vals = true_vals[np.isfinite(true_vals)]

        pred_vals = result[t, :]
        pred_vals = pred_vals[(pred_vals > 0) & np.isfinite(pred_vals)]

        m_t = pred_vals.size
        if m_t == 0:
            Pfa[t] = 0.0
            continue

        FP_t = 0
        for pred in pred_vals:
            if true_vals.size == 0:
                # 无真实目标，所有预测均视为虚警
                FP_t += 1
            else:
                errors = np.abs(pred - true_vals) * (180/512)
                dist = errors**2  # 逐元素平方误差
                # 若与所有真实目标的最小平方误差都 >= c，则视为虚警
                if np.min(dist) >= c:
                    FP_t += 1

        Pfa[t] = FP_t / m_t

    return Pfa

def calculate_ospa(X, Y, c=5, p=1):
    """
    计算 OSPA 距离和 RMSE
    参数:
        X (np.ndarray): 真实目标集 (n x d)
        Y (np.ndarray): 预测目标集 (m x d)
        c (float): 截断距离
        p (int): 范数阶数
    返回:
        tuple: (ospa_dist, rmse_dist, error_matrix)
    """
    n, m = X.shape[0], Y.shape[0]
    k = max(n, m)
    
    # 构建代价矩阵
    cost_matrix = np.full((k, k), c**p, dtype=np.float64)
    error_matrix = np.full((X.shape[0], X.shape[1]), 999999.0)
    
    for i in range(n):
        for j in range(m):
            # # 计算有效索引
            # out = X[i].copy()
            # # 找到正、负无穷的位置
            # mask = np.isinf(out)
            # # 在这些位置赋值
            # X[i, mask] = 1024
            valid_X = ~np.isinf(X[i])
            valid_Y = ~np.isinf(Y[j])
            valid_idx = valid_X & valid_Y

            # errors = np.full(len(X[i]), c**p, dtype=np.float64)
            if np.any(valid_idx):
                # 角度转换 (假设 512 点对应 180 度)
                errors = (X[i, valid_idx] - Y[j, valid_idx]) * 180/512
                dist = np.linalg.norm(errors, ord=p) / len(errors)
                cost_matrix[i, j] = min(dist, c)**p
                
                # 更新误差矩阵
                sq_errors = errors**2
                update_mask = sq_errors < error_matrix[i, valid_idx]
                error_matrix[i, valid_idx] = np.where(update_mask, sq_errors, error_matrix[i, valid_idx])
    # print(cost_matrix)
    # 匈牙利算法匹配
    # row_idx, col_idx = linear_sum_assignment(cost_matrix)
    # total_cost = cost_matrix[row_idx, col_idx].sum() + c**p * (k - len(row_idx))

    total_cost = 0
    for i in range(k):
        total_cost += np.min(cost_matrix[:n, i])
    
    # 计算 OSPA
    ospa_dist = (total_cost / k) ** (1/p)
    
    # 计算 RMSE
    valid_errors = error_matrix[error_matrix < 999999]
    rmse_dist = np.sqrt(np.mean(valid_errors)) if valid_errors.size > 0 else np.inf

    # 计算 Pd
    valid_errors = error_matrix[error_matrix < 4]
    pd_dist = len(valid_errors) / (X.shape[0] * X.shape[1]) if valid_errors.size > 0 else 0
    
    return ospa_dist, rmse_dist, pd_dist, error_matrix

def match_input_truth(input_dir, truth_dir, truth_single_file=None):
    """匹配输入文件和真值文件"""
    # 获取所有输入文件
    input_files = []
    for ext in ['*.npy', 't_angle*.mat']:
        input_files.extend(input_dir.glob(ext))
    
    # 构建文件匹配字典
    file_pairs = {}
    pattern = re.compile(r'(\d+)([a-zA-Z])')
    for input_file in input_files:
        if 'track' in str(input_file):
            continue
        # 提取文件编号
        if str(input_file).endswith('.mat'):
            file_id = int(input_file.name.split('.')[0][7:])
        else:
            match = pattern.search(input_file.name)
            if not match:
                continue
            file_id = match.group(1)
        
        # 查找对应的真值文件
        if truth_single_file:
            # 单个真值文件模式
            truth_path = truth_single_file
        else:
            # 多个真值文件模式
            truth_path = truth_dir / f'truth{file_id}.mat'
        
        if truth_path.exists():
            file_pairs[input_file] = truth_path
    
    return file_pairs

def load_single_pair(input_path, truth_path):
    """加载单个输入-真值对"""
    # 加载输入数据
    if input_path.suffix == '.mat':
        btr = scipy.io.loadmat(input_path)['t_angle']
    else:
        btr = np.load(input_path)
    
    # 加载真值数据
    truth_data = scipy.io.loadmat(truth_path)
    theta_true = truth_data['truth']  # 假设mat文件中变量名为theta_true
    
    # 维度对齐
    if theta_true.shape[0] > btr.shape[0]:
        theta_true = theta_true[:btr.shape[0], :]
    elif theta_true.shape[0] < btr.shape[0]:
        raise ValueError(f"真值数据样本数不足: {input_path.name}")
    
    return btr, theta_true

def batch_evaluate(input_dir, truth_dir, output_file, single_truth=False, threshold=1.0):
    """批量评估主函数"""
    # 解析文件匹配
    truth_single = truth_dir if single_truth else None
    file_pairs = match_input_truth(input_dir, truth_dir, truth_single)
    
    # 准备结果存储
    results = []
    PD = []
    total_ospa, total_rmse, total_pd, total_pfa, count = 0.0, 0.0, 0.0, 0.0, 0
    total_pd_mat = np.zeros((512,2))
    
    # 进度条
    pbar = tqdm(file_pairs.items(), desc='Processing files')
    for input_path, truth_path in pbar:
        # 加载数据
        btr, theta_true = load_single_pair(input_path, truth_path)
        name = input_path.stem
        
        # 执行评估
        track_result, ospa, rmse, pd, pfa, pd_matrix = evaluate_trace(btr, theta_true, threshold)  # 需要实现evaluate_trace
        # pd = calculate_pd(btr, theta_true, 512, theta_true.shape[1])
        
        # 记录结果
        results.append({
            'file_id': input_path.stem,
            'input_file': input_path.name,
            'truth_file': truth_path.name,
            'OSPA': ospa,
            'RMSE': rmse,
            'Pd' : pd,
            'PFA' : pfa
        })
        total_ospa += ospa
        total_rmse += rmse
        total_pd += pd
        total_pfa += pfa
        total_pd_mat += pd_matrix
        count += 1
        pbar.set_postfix_str(f"{input_path.name}/{truth_path.name}: OSPA={ospa:.2f}, PFA={pfa:.2f}, Pd={pd:.3f}")
        PD.append(pd)

        np.save(str(input_dir) + '/' + name + '_track.npy', track_result)
    
    N = 70
    if N > count:
        N = len(PD) // 2
    for _ in range(min(N, len(PD))):
        PD.remove(min(PD))

    PD = np.mean(np.array(PD))
    total_pd_mat /= count
    np.save("./tools/swellpd/" + output_file + '.npy', total_pd_mat)
    # 保存结果
    # df = pd.DataFrame(results)
    # df.to_excel(output_file, index=False)
    # print(f"\n评估结果已保存至: {output_file}")
    avg_results = {
        'File': 'Average',
        'OSPA': total_ospa / count,
        'RMSE': total_rmse / count,
        'Pd': total_pd / count,
        'PFA': total_pfa / count
    }

    print("\nFinal Average Values:")
    print(f"OSPA: {avg_results['OSPA']:.4f}")
    print(f"RMSE: {avg_results['RMSE']:.4f}")
    print(f"Pd: {avg_results['Pd']:.4f}")
    print(f"pfa: {avg_results['PFA']:.4f}")
    print(f"Pd: {PD:.4f}")

    line = (
        f"OSPA: {avg_results['OSPA']:.4f}, "
        f"RMSE: {avg_results['RMSE']:.4f}, "
        f"Pd_avg: {avg_results['Pd']:.4f}, "
        f"PFA: {avg_results['PFA']:.4f}, "
        f"Pd: {PD:.4f}"
        )

    # write to a .txt in one line
    with open(str(input_dir) + '/results.txt', 'a') as f:
        f.write(line + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='批量DOA轨迹评估')
    parser.add_argument('--input', required=True, help='输入文件目录')
    parser.add_argument('--truth', required=True, help='真值文件目录或单个文件')
    parser.add_argument('--output', default='results.xlsx', help='输出结果文件')
    parser.add_argument('--single', type=int, default=1, 
                      help='是否使用单个真值文件（需包含所有样本的真值）')
    parser.add_argument('--threshold', type=float, default=1, 
                      help='阈值控制虚警率')
    
    args = parser.parse_args()
    
    # 路径处理
    input_dir = Path(args.input)
    truth_path = Path(args.truth)
    
    # 验证路径
    if not input_dir.exists():
        raise FileNotFoundError(f"234: {input_dir}")
    
    if args.single and not truth_path.is_file():
        raise FileNotFoundError(f"123: {truth_path}")
    
    if not args.single and not truth_path.is_dir():
        raise NotADirectoryError(f"456: {truth_path}")
    
    net_name = args.input.split('/')[3]
    number = args.input.split('/')[-3][-5:]

    out_name = net_name + number
    print(out_name)
    # 运行批处理
    batch_evaluate(
        input_dir=input_dir,
        truth_dir=truth_path if args.single else truth_path,
        output_file=out_name,#args.output,
        single_truth=args.single,
        threshold=args.threshold
    )

# python calculatetrace.py --input /wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/SBL/-20dB/ --truth truth.mat --single 1 --threshold 2