#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""批量时序异常检测脚本

基于 temporal_anomaly_detector.py 实现批量的时序异常检测
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from temporal_anomaly_detector import TemporalAnomalyDetector
import json
from typing import Dict, List, Tuple

# 配置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

# 定义数据目录和输出文件
# 使用当前工作目录而不是硬编码的路径
data_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
fixed_data_path = os.path.join(data_dir, "australia_batch_data_fixed.npy")
metadata_path = os.path.join(data_dir, "australia_batch_data_fixed_metadata.npy")
output_dir = os.path.join(data_dir, "temporal_anomaly_results")

# 创建输出目录
def ensure_output_dir():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

# 加载数据和元数据
def load_data_and_metadata():
    """加载修复后的数据和元数据"""
    try:
        # 加载数据和元数据
        data = np.load(fixed_data_path)
        metadata = np.load(metadata_path, allow_pickle=True).item()
        
        print(f"成功加载数据：")
        print(f"  数据形状: {data.shape}")
        print(f"  特征数量: {len(metadata['feature_names'])}")
        print(f"  网格大小: {metadata['width']}x{metadata['height']}")
        print(f"  时间序列长度: {metadata['num_time_steps']}")
        
        return data, metadata
    except Exception as e:
        print(f"加载数据失败: {e}")
        raise

# 获取时间范围内的数据索引
def get_time_indices(timestamps, start_time, end_time):
    """根据开始时间和结束时间获取数据索引"""
    # 将字符串时间转换为numpy datetime64
    start_dt = np.datetime64(start_time)
    end_dt = np.datetime64(end_time)
    
    # 将时间戳转换为numpy datetime64
    time_array = np.array(timestamps, dtype='datetime64')
    
    # 找到时间范围内的索引
    mask = (time_array >= start_dt) & (time_array <= end_dt)
    indices = np.where(mask)[0]
    
    if len(indices) == 0:
        print(f"警告：在指定的时间范围内没有找到数据点")
    else:
        print(f"找到 {len(indices)} 个时间点在 {start_time} 到 {end_time} 之间")
        print(f"  开始索引: {indices[0]}, 结束索引: {indices[-1]}")
    
    return indices

# 为指定网格点的时序数据进行异常检测
def detect_anomalies_for_grid_point(
    data: np.ndarray,
    metadata: Dict,
    grid_x: int,
    grid_y: int,
    feature_name: str,
    time_indices: np.ndarray,
    detection_method: str = '3sigma',
    threshold_factor: float = 1.0,
    initial_window_size: int = 100,
    threshold: float = 3.0,
    threshold_method: str = 'sigma',
    percentile_value: float = 95.0,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """为指定网格点的指定特征进行时序异常检测"""
    # 获取特征索引
    if feature_name not in metadata['feature_names']:
        raise ValueError(f"特征 '{feature_name}' 不存在于数据中")
    feature_idx = metadata['feature_names'].index(feature_name)
    
    try:
        # 提取完整的时序数据用于异常检测（考虑完整历史数据）
        full_time_series_data = data[grid_y, grid_x, feature_idx, :]
        
        # 但只返回指定时间范围内的结果
        time_series_data = full_time_series_data[time_indices]
        
        # 初始化异常检测器
        detector_kwargs = kwargs.copy()
        if detection_method == '3sigma':
            # 输出window_size参数值以进行调试
            print(f"网格点 ({grid_x}, {grid_y}) 的window_size参数值: {detector_kwargs.get('window_size', '未设置')}")
            detector_kwargs['threshold_method'] = threshold_method
            if threshold_method == 'percentile':
                detector_kwargs['percentile'] = percentile_value
        elif detection_method == 'percentile':
            # 可以调整百分位数和窗口大小
            detector_kwargs.setdefault('percentile', percentile_value)
            detector_kwargs.setdefault('window_size', 50)
        elif detection_method == 'spot':
            # 可以调整q值和窗口大小
            detector_kwargs.setdefault('q', kwargs.get('spot_q_value', 0.98))
            detector_kwargs.setdefault('window_size', 100)
        elif detection_method == 'fixed_window_once':
            # 设置初始窗口大小和固定阈值
            detector_kwargs['initial_window_size'] = initial_window_size
            detector_kwargs['threshold'] = threshold
            detector_kwargs['threshold_method'] = threshold_method
            if threshold_method == 'percentile':
                detector_kwargs['percentile'] = percentile_value
            print(f"网格点 ({grid_x}, {grid_y}) 使用fixed_window_once方法: 初始窗口大小={initial_window_size}, 固定阈值={threshold}, 阈值方法={threshold_method}")
        
        detector = TemporalAnomalyDetector(method=detection_method, **detector_kwargs)
        
        # 进行异常检测 - 使用完整的时序数据进行训练/计算统计量
        try:
            # 先对完整数据进行异常检测以确保算法考虑了所有历史数据
            full_anomaly_labels = detector.detect(full_time_series_data)
            
            # 然后只提取指定时间范围内的异常标签
            # 添加边界检查以避免索引越界错误
            if len(full_anomaly_labels) >= max(time_indices) + 1:
                anomaly_labels = full_anomaly_labels[time_indices]
            else:
                print(f"警告: 异常标签数组长度 ({len(full_anomaly_labels)}) 小于最大时间索引 ({max(time_indices)})")
                # 创建一个与时间序列数据长度相同的全0数组
                anomaly_labels = np.zeros(len(time_series_data), dtype=int)
            
            # 如果指定了阈值因子，可以调整异常检测的灵敏度
            if threshold_factor != 1.0 and detection_method == '3sigma':
                # 对于3sigma方法，可以重新计算阈值并调整标签
                mean = np.mean(full_time_series_data)  # 使用完整数据计算统计量
                std = np.std(full_time_series_data)
                threshold = 3 * std * threshold_factor
                adjusted_labels = np.zeros_like(anomaly_labels)
                adjusted_labels[abs(time_series_data - mean) > threshold] = 1
                anomaly_labels = adjusted_labels
            
            return time_series_data, anomaly_labels, time_indices
        except Exception as e:
            print(f"网格点 ({grid_x}, {grid_y}) 的异常检测失败: {e}")
            # 记录详细的错误信息，包括时间序列数据的长度
            print(f"  时间序列数据长度: {len(full_time_series_data)}")
            print(f"  时间索引范围: {min(time_indices)} - {max(time_indices)}")
            # 返回原始数据和全0的异常标签
            return time_series_data, np.zeros(len(time_series_data), dtype=int), time_indices
    except Exception as e:
        print(f"网格点 ({grid_x}, {grid_y}) 的数据处理失败: {e}")
        # 返回空数据和全0的异常标签
        return np.array([]), np.array([], dtype=int), time_indices

# 可视化单个网格点的时序异常检测结果
def visualize_single_grid_point_anomalies(
    time_series_data: np.ndarray,
    anomaly_labels: np.ndarray,
    timestamps: np.ndarray,
    grid_x: int,
    grid_y: int,
    feature_name: str,
    detection_method: str
):
    """可视化单个网格点的时序异常检测结果"""
    # 创建图形
    plt.figure(figsize=(12, 6))
    
    # 绘制时序数据
    plt.plot(timestamps, time_series_data, 'b-', label='原始数据')
    
    # 标记异常点
    anomaly_indices = np.where(anomaly_labels == 1)[0]
    if len(anomaly_indices) > 0:
        plt.scatter(
            timestamps[anomaly_indices], 
            time_series_data[anomaly_indices], 
            color='r', 
            marker='o', 
            label=f'异常点 ({len(anomaly_indices)})'
        )
    
    # 设置图形属性
    plt.title(f'网格点 ({grid_x}, {grid_y}) - {feature_name} 时序异常检测')
    plt.xlabel('时间')
    plt.ylabel('特征值')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # 保存图形
    output_dir = ensure_output_dir()
    save_path = os.path.join(output_dir, f'temporal_anomaly_{feature_name}_{grid_x}_{grid_y}_{detection_method}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return save_path

# 批量处理所有网格点的时序异常检测
def batch_temporal_anomaly_detection(
    feature_name: str = 'albedo_03_difference',   
    start_time: str = '2021-02-01T02:00:00',
    end_time: str = '2021-02-01T12:00:00',
    detection_method: str = '3sigma',
    threshold_factor: float = 1.0,
    threshold_method: str = 'sigma',
    percentile_value: float = 95.0,
    visualize: bool = True,
    sample_grid_points: bool = False,
    num_samples: int = 5,
    visualize_center_points: bool = False,
    center_radius: int = 2,
    window_size: int = 50,
    spot_q_value: float = 0.98,  # SPOT算法的q值参数
    initial_window_size: int = 100,  # 固定窗口一次方法的初始窗口大小
    threshold: float = 3.0  # 固定窗口一次方法的固定阈值
):
    """批量处理所有网格点的时序异常检测"""
    try:
        # 加载数据和元数据
        data, metadata = load_data_and_metadata()
        
        # 获取时间范围内的索引
        time_indices = get_time_indices(metadata['timestamps'], start_time, end_time)
        if len(time_indices) == 0:
            print("没有找到符合条件的时间点，无法进行异常检测")
            return
        
        # 获取时间戳
        timestamps = np.array(metadata['timestamps'], dtype='datetime64')[time_indices]
        
        # 获取网格大小
        width = metadata['width']
        height = metadata['height']
        
        # 存储所有网格点的异常检测结果
        all_results = {}
        
        # 确定要处理的网格点
        grid_points = []
        if sample_grid_points:
            # 随机采样一些网格点
            np.random.seed(42)
            sample_indices = np.random.choice(width * height, min(num_samples, width * height), replace=False)
            for idx in sample_indices:
                grid_y = idx // width
                grid_x = idx % width  
                grid_points.append((grid_x, grid_y))
        else:
            # 处理所有网格点
            for grid_y in range(height):
                for grid_x in range(width):
                    grid_points.append((grid_x, grid_y))
        
        # 存储中心附近的网格点（用于可视化）
        center_grid_points = []
        if visualize_center_points:
            center_x = width // 2
            center_y = height // 2
            for grid_y in range(max(0, center_y - center_radius), min(height, center_y + center_radius + 1)):
                for grid_x in range(max(0, center_x - center_radius), min(width, center_x + center_radius + 1)):
                    center_grid_points.append((grid_x, grid_y))
            print(f"已选择中心({center_x}, {center_y})附近半径{center_radius}范围内的{len(center_grid_points)}个网格点进行可视化")
        
        # 为每个网格点进行异常检测
        total_points = len(grid_points)
        for i, (grid_x, grid_y) in enumerate(grid_points):
            if (i + 1) % 10 == 0 or i == total_points - 1:
                print(f"进度: {i+1}/{total_points} 网格点")
            
            # 进行异常检测
            time_series_data, anomaly_labels, _ = detect_anomalies_for_grid_point(
                data=data,
                metadata=metadata,
                grid_x=grid_x,
                grid_y=grid_y,
                feature_name=feature_name,
                time_indices=time_indices,
                detection_method=detection_method,
                threshold_factor=threshold_factor,
                window_size=window_size,
                spot_q_value=spot_q_value,  # 传递SPOT算法的q值参数
                initial_window_size=initial_window_size,  # 传递固定窗口一次方法的初始窗口大小
                threshold=threshold  # 传递固定窗口一次方法的固定阈值
            )
            
            # 统计异常信息
            anomaly_count = np.sum(anomaly_labels)
            result = {
                'grid_x': grid_x,
                'grid_y': grid_y,
                'time_series_data': time_series_data.tolist(),
                'anomaly_labels': anomaly_labels.tolist(),
                'anomaly_count': int(anomaly_count),
                'anomaly_ratio': float(anomaly_count / len(anomaly_labels)) if len(anomaly_labels) > 0 else 0.0
            }
            all_results[f"{grid_x}_{grid_y}"] = result
            
            # 可视化结果
            if visualize and ((visualize_center_points and (grid_x, grid_y) in center_grid_points) or (not visualize_center_points and i < 10)):
                save_path = visualize_single_grid_point_anomalies(
                    time_series_data,
                    anomaly_labels,
                    timestamps,
                    grid_x,
                    grid_y,
                    feature_name,
                    detection_method
                )
                print(f"已保存网格点 ({grid_x}, {grid_y}) 的可视化结果: {save_path}")
        
        # 保存所有结果
        output_dir = ensure_output_dir()
        results_file = os.path.join(output_dir, f'batch_temporal_anomaly_results_{feature_name}_{detection_method}.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        # 提取特定时间点的结果并保存为npy文件
        # 需要提取的时间点：0201那天的0350、0400、0410、0420、0500、0600、0700、0800、0900、1000、1100时刻
        target_times = ['03:50', '04:00', '04:10', '04:20', '05:00', '06:00', 
                        '07:00', '08:00', '09:00', '10:00', '11:00', '12:00']
        
        # 创建空的异常热力图数组
        specific_time_anomalies = {}
        
        # 创建热力图数据结构
        height = metadata['height']
        width = metadata['width']
        
        # 打印一些时间戳用于调试
        print(f"\n可用的时间戳示例 (前10个):")
        for i, ts in enumerate(timestamps[:10]):
            print(f"  {i}: {ts}")
        
        # 对每个目标时间点，提取异常结果
        for target_time in target_times:
            # 使用精确的时间匹配
            matched_indices = []
            matched_timestamps = []
            
            # 构建完整的目标时间戳格式
            if target_time == '10:00':
                # 特殊处理10:00，避免匹配到02:10
                target_timestamp = '2021-02-01T10:00:00'
                for idx, ts in enumerate(timestamps):
                    ts_str = str(ts)
                    if target_timestamp in ts_str:
                        matched_indices.append(idx)
                        matched_timestamps.append(ts_str)
            else:
                # 对于其他时间，使用精确匹配
                for idx, ts in enumerate(timestamps):
                    ts_str = str(ts)
                    # 使用T分隔符确保只匹配时间部分
                    if ('T' + target_time + ':') in ts_str and '2021-02-01' in ts_str:
                        matched_indices.append(idx)
                        matched_timestamps.append(ts_str)
            
            if matched_indices:
                # 取第一个匹配的时间点
                time_idx = matched_indices[0]
                time_str = matched_timestamps[0]
                
                # 创建一个二维数组存储该时间点的异常情况
                anomaly_map = np.zeros((height, width), dtype=int)
                
                for point_id, result in all_results.items():
                    grid_x, grid_y = map(int, point_id.split('_'))
                    # 检查该网格点是否在该时间点有异常
                    if len(result['anomaly_labels']) > time_idx:
                        anomaly_map[grid_y, grid_x] = result['anomaly_labels'][time_idx]
                
                specific_time_anomalies[time_str] = anomaly_map
                print(f"已提取 {time_str} 的异常热力图数据")
            else:
                print(f"警告: 未找到包含 {target_time} 的时间点")
        
        # 保存特定时间点的异常结果为npy文件
        specific_time_file = os.path.join(output_dir, f'specific_time_anomalies_{feature_name}_{detection_method}.npy')
        np.save(specific_time_file, specific_time_anomalies)
        print(f"特定时间点的异常结果已保存至: {specific_time_file}")
        
        # 绘制特定时间点的热力图
        if specific_time_anomalies:
            draw_time_anomaly_heatmaps(specific_time_anomalies, feature_name, detection_method)
        
        print(f"批量时序异常检测完成！")
        print(f"所有结果已保存至: {results_file}")
        
        # 计算整体统计信息
        total_anomalies = sum(result['anomaly_count'] for result in all_results.values())
        total_data_points = sum(len(result['time_series_data']) for result in all_results.values())
        overall_anomaly_ratio = total_anomalies / total_data_points if total_data_points > 0 else 0.0
        
        print(f"\n整体统计信息:")
        print(f"  处理的网格点数量: {len(all_results)}")
        print(f"  总数据点数: {total_data_points}")
        print(f"  总异常点数: {total_anomalies}")
        print(f"  整体异常比例: {overall_anomaly_ratio * 100:.2f}%")
        
        # 找出异常最严重的网格点
        worst_points = sorted(all_results.items(), key=lambda x: x[1]['anomaly_ratio'], reverse=True)[:5]
        print(f"\n异常比例最高的5个网格点:")
        for i, (point_id, result) in enumerate(worst_points):
            grid_x, grid_y = map(int, point_id.split('_'))
            print(f"  {i+1}. ({grid_x}, {grid_y}): {result['anomaly_ratio']*100:.2f}% ({result['anomaly_count']}/{len(result['time_series_data'])})")
            
        return all_results
        
    except Exception as e:
        print(f"批量时序异常检测失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def draw_time_anomaly_heatmaps(anomaly_data, feature_name, detection_method):
    """绘制特定时间点的异常热力图"""
    # 计算子图的行数和列数
    num_times = len(anomaly_data)
    n_cols = 3  # 每行3个子图
    n_rows = (num_times + n_cols - 1) // n_cols  # 向上取整
    
    # 创建大图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    
    # 为每个时间点绘制热力图
    for idx, (time_str, anomaly_map) in enumerate(anomaly_data.items()):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        
        # 跳过超出范围的子图
        if row_idx >= len(axes) or col_idx >= len(axes[0]):
            continue
        
        ax = axes[row_idx][col_idx]
        
        # 绘制热力图
        im = ax.imshow(anomaly_map, cmap='hot', interpolation='none')
        
        # 设置标题（只显示时间部分）
        time_display = time_str.split('T')[1]  # 提取时间部分
        ax.set_title(f'时间: {time_display}')
        
        # 添加颜色条
        fig.colorbar(im, ax=ax, label='异常值')
    
    # 删除多余的子图
    for idx in range(num_times, n_rows * n_cols):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        if row_idx < len(axes) and col_idx < len(axes[0]):
            fig.delaxes(axes[row_idx][col_idx])
    
    # 调整布局
    plt.suptitle(f'{feature_name} - 特定时间点异常热力图 ({detection_method})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # 调整suptitle的位置
    
    # 保存图形
    output_dir = ensure_output_dir()
    save_path = os.path.join(output_dir, f'specific_time_anomalies_heatmap_{feature_name}_{detection_method}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"特定时间点异常热力图已保存至: {save_path}")

# 主函数
if __name__ == "__main__":
    # 设置参数（根据用户要求）
    feature_name = 'tbb_07_difference'  # 用户指定的特征波段
    start_time = '2021-02-01T02:00:00'       # 用户指定的开始时间
    end_time = '2021-02-01T12:00:00'         # 用户指定的结束时间
    detection_method = 'fixed_window_once'   # 使用新的固定窗口一次方法
    threshold_factor = 1                     # 阈值因子，用于调整异常检测的灵敏度
    threshold_method = 'percentile'               # 阈值计算方法: 'sigma'(三西格玛), 'percentile'(百分位数), 'fixed'(固定阈值)
    percentile_value = 95                    # 百分位数值（当threshold_method='percentile'时使用）
    window_size = 100                        # 滑动窗口大小，增大窗口以捕获更多历史信息
    spot_q_value = 0.90                      # 降低q值以提高异常检测灵敏度（默认0.98）
    initial_window_size = 50               # 固定窗口一次方法的初始窗口大小（选取开始时刻前100个时相）
    threshold = 3.0                          # 固定窗口一次方法的固定阈值
    
    print("=== 批量时序异常检测任务开始 ===")
    print(f"特征波段: {feature_name}")
    print(f"时间范围: {start_time} 到 {end_time}")
    print(f"检测方法: {detection_method}")
    print(f"阈值因子: {threshold_factor}")
    print(f"阈值计算方法: {threshold_method}")
    print(f"百分位数值: {percentile_value}")
    if detection_method == 'fixed_window_once':
        print(f"初始窗口大小: {initial_window_size}（只使用开始时刻前{initial_window_size}个时相）")
        print(f"固定阈值: {threshold}")
    else:
        print(f"窗口大小: {window_size}")
    
    # 执行批量时序异常检测
    results = batch_temporal_anomaly_detection(
        feature_name=feature_name,
        start_time=start_time,
        end_time=end_time,
        detection_method=detection_method,
        threshold_factor=threshold_factor,
        threshold_method=threshold_method,
        percentile_value=percentile_value,
        visualize=True,
        sample_grid_points=False,  # 处理所有网格点
        visualize_center_points=True,  # 可视化中心附近的网格点
        center_radius=1,  # 中心附近的半径
        window_size=window_size,  # 传递窗口大小参数
        spot_q_value=spot_q_value,  # 传递spot算法的q值参数
        initial_window_size=initial_window_size,  # 传递固定窗口一次方法的初始窗口大小
        threshold=threshold  # 传递固定窗口一次方法的固定阈值
    )
    
    print("\n=== 批量时序异常检测任务完成 ===")