import numpy as np
import scipy.io
import re
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def load_truth(theta_dir: Path, file_id: str, single_truth: bool):
    """
    加载真值矩阵 theta_true.
    如果 single_truth=True，则 theta_dir 本身即为真值文件路径。
    否则在 theta_dir 下寻找 truth{file_id}.mat 或 .npy。
    """
    if single_truth:
        truth_path = theta_dir
    else:
        mat_path = theta_dir / f"truth{file_id}.mat"
        npy_path = theta_dir / f"truth{file_id}.npy"
        if mat_path.exists():
            truth_path = mat_path
        elif npy_path.exists():
            truth_path = npy_path
        else:
            raise FileNotFoundError(f"找不到真值文件 truth{file_id}.mat 或 .npy")
    if truth_path.suffix == '.mat':
        data = scipy.io.loadmat(truth_path)
        # 支持变量名 'truth' 或 'theta_true'
        theta_true = data.get('truth', data.get('theta_true'))
    else:
        theta_true = np.load(truth_path)
    theta_true = theta_true[:512, :]
    return theta_true

def count_false_alarms(theta_true: np.ndarray, result: np.ndarray, c: float):
    """
    逐帧统计虚警数（FP）——预测点到任一真值点的最小平方误差 >= c 即视为该预测为虚警。
    返回累计的 FP 数量。
    """
    result = result[:512, :]
    N, M = theta_true.shape
    # 确保 result 是 N×K
    if result.ndim == 1:
        result = result.reshape(-1, 1)
    N2, K = result.shape
    assert N2 == N, "theta_true 与 result 帧数不一致"

    valid_tracks = [i for i in range(result.shape[1]) 
                       if np.sum(result[:, i] > 0) > 1]
        
    if not valid_tracks:
        return 0
    
    result = result[:, valid_tracks]

    total_FP_all = []
    for t in range(N):
        true_vals = theta_true[t, :]
        true_vals = true_vals[np.isfinite(true_vals)]
        pred_vals = result[t, :]
        pred_vals = pred_vals[(pred_vals > 0) & np.isfinite(pred_vals)]
        total_FP = 0
        for pred in pred_vals:
            if true_vals.size == 0:
                total_FP += 1
            else:
                errors = np.abs(pred - true_vals) * (180/512)
                dist2 = errors**2
                if np.min(dist2) >= c:
                    total_FP += 1
        total_FP_all.append(total_FP)
    # print(total_FP_all)
    return np.mean(total_FP_all)

def main():
    parser = argparse.ArgumentParser(description="统计各 dB 子文件夹下 track.npy 的虚警数量")
    parser.add_argument("input_root", help="包含 -20~-29dB 子文件夹的根目录")
    parser.add_argument("truth_dir", help="真值文件目录或单个真值文件")
    parser.add_argument("--single", action="store_true",
                        help="如果真值聚合在一个文件，请加此标志")
    parser.add_argument("--c", type=float, default=4.0,
                        help="判断虚警的平方误差阈值 c (默认 4)")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    truth_dir = Path(args.truth_dir)
    single_truth = args.single
    c = args.c

    report = []
    results_all = []
    db_folders = [f"{i}dB" for i in range(20, 30)]
    for db_name in db_folders:
        db_folder = input_root / db_name
        if not db_folder.is_dir(): 
            continue
        # 遍历每个子文件夹里的 *_track.npy 文件
        results = []
        for track_file in db_folder.glob("*_track.npy"):
            stem = track_file.stem  # e.g. "001_track"
            m = re.search(r"(\d+)", stem)
            if not m:
                print(f"跳过未识别 ID 的文件: {track_file.name}")
                continue
            file_id = m.group(1)
            try:
                theta_true = load_truth(truth_dir, file_id, single_truth)
            except FileNotFoundError as e:
                print(e)
                continue

            result = np.load(track_file)
            fp_count = count_false_alarms(theta_true, result, c)

            # report.append({
            #     "dB_folder": db_folder.name,
            #     "track_file": track_file.name,
            #     "false_alarms": fp_count
            # })
            results.append(fp_count)
        results_all.append(np.mean(results))
    print(results_all)


if __name__ == "__main__":
    main()

