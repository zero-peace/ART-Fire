import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
from .spatial_anomaly_detector import SpatialAnomalyDetector

class TimeRangeSpatialAnomalyDetector:
    """时间段空间异常检测器
    
    允许用户指定时间段和特征，对该时间段内的数据进行空间异常检测
    """
    
    def __init__(self, data_path='australia_batch_data.npy', 
                 metadata_path='australia_batch_data_metadata.npy'):
        """初始化检测器
        
        Args:
            data_path: numpy数据文件路径
            metadata_path: 元数据文件路径
        """
        # 加载数据和元数据
        self.data = np.load(data_path)
        self.metadata = np.load(metadata_path, allow_pickle=True).item()
        
        # 提取基本信息
        self.width = self.metadata['width']
        self.height = self.metadata['height']
        self.feature_names = self.metadata['feature_names']
        self.timestamps = self.metadata['timestamps']
        
        # 转换时间戳为datetime类型便于处理
        self.datetimes = np.array([
            datetime.utcfromtimestamp(timestamp.astype('datetime64[s]').astype(int))
            if isinstance(timestamp, np.datetime64) or isinstance(timestamp, int)
            else timestamp
            for timestamp in self.timestamps
        ])
        
        print(f"成功加载数据：")
        print(f"  网格大小: {self.width}x{self.height}")
        print(f"  特征数量: {len(self.feature_names)}")
        print(f"  时间范围: {self.datetimes[0]} 到 {self.datetimes[-1]}")
        print(f"  时间步数: {len(self.datetimes)}")
    
    def get_time_indices(self, start_time, end_time):
        """获取指定时间段内的时间索引
        
        Args:
            start_time: 开始时间（datetime对象或字符串）
            end_time: 结束时间（datetime对象或字符串）
            
        Returns:
            时间索引数组
        """
        # 转换字符串时间为datetime对象
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        if isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            
        # 找到时间范围内的索引
        mask = (self.datetimes >= start_time) & (self.datetimes <= end_time)
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            print(f"警告：在指定的时间段内没有找到数据。\n可用时间范围：{self.datetimes[0]} 到 {self.datetimes[-1]}")
            
        return indices
    
    def detect_anomalies(self, feature_name, start_time, end_time, 
                         detection_method='otsu', threshold_factor=1.0, sigma_threshold=3.0, **method_kwargs):
        """对指定特征和时间段进行空间异常检测
        
        Args:
            feature_name: 特征名称
            start_time: 开始时间
            end_time: 结束时间
            detection_method: 异常检测方法 ('otsu', '3sigma', 'lof')
            threshold_factor: Otsu方法的阈值调整因子，大于1使阈值更高更严格，小于1使阈值更低更宽松
            sigma_threshold: 3σ方法的标准差倍数，用于调整异常检测的严格程度
            **method_kwargs: 检测方法的额外参数
            
        Returns:
            检测结果字典
        """
        # 验证特征名称
        if feature_name not in self.feature_names:
            raise ValueError(f"无效的特征名称：{feature_name}\n可用特征：{self.feature_names}")
        
        # 获取特征索引
        feature_idx = self.feature_names.index(feature_name)
        
        # 检查特征数据是否全部为NaN
        feature_data = self.data[:, :, feature_idx, :]
        if np.isnan(feature_data).all():
            print(f"错误：特征 '{feature_name}' 的所有值都是NaN，无法进行异常检测")
            return {'status': 'error', 'message': f'特征 {feature_name} 的所有值都是NaN，无法进行异常检测'}
        
        # 检查特征数据是否同质化严重（标准差为0）
        non_nan_data = feature_data[~np.isnan(feature_data)]
        if len(non_nan_data) > 0 and np.std(non_nan_data) == 0:
            print(f"警告：特征 '{feature_name}' 的数据同质化严重（所有非NaN值相同），异常检测结果可能不准确")
        
        # 检查特征数据是否全部为NaN
        feature_data = self.data[:, :, feature_idx, :]
        if np.isnan(feature_data).all():
            print(f"错误：特征 '{feature_name}' 的所有值都是NaN，无法进行异常检测")
            return {'status': 'error', 'message': f'特征 {feature_name} 的所有值都是NaN，无法进行异常检测'}
        
        # 检查特征数据是否同质化严重（标准差为0）
        non_nan_data = feature_data[~np.isnan(feature_data)]
        if len(non_nan_data) > 0 and np.std(non_nan_data) == 0:
            print(f"警告：特征 '{feature_name}' 的数据同质化严重（所有非NaN值相同），异常检测结果可能不准确")
        
        # 获取时间索引
        time_indices = self.get_time_indices(start_time, end_time)
        
        if len(time_indices) == 0:
            return {'status': 'error', 'message': '没有找到匹配的时间数据'}
        
        # 初始化检测器并传递相应的参数
        if detection_method == 'otsu':
            detector = SpatialAnomalyDetector(method=detection_method, threshold_factor=threshold_factor, **method_kwargs)
        elif detection_method == '3sigma':
            detector = SpatialAnomalyDetector(method=detection_method, sigma_threshold=sigma_threshold, **method_kwargs)
        else:
            detector = SpatialAnomalyDetector(method=detection_method, **method_kwargs)
        
        # 计算时间范围内的平均特征值，形成空间分布图
        spatial_data = np.nanmean(self.data[:, :, feature_idx, time_indices], axis=2)
        
        # 处理NaN值
        # 将NaN值替换为该特征的全局平均值（忽略NaN）
        feature_data = self.data[:, :, feature_idx, :]
        if np.isnan(feature_data).all():
            # 如果该特征所有值都是NaN，使用0填充
            print(f"警告：特征 '{feature_name}' 的所有值都是NaN，使用0填充")
            global_mean = 0
        else:
            global_mean = np.nanmean(feature_data)
            
        # 再次检查global_mean是否为NaN
        if np.isnan(global_mean):
            global_mean = 0
            print(f"警告：无法计算特征 '{feature_name}' 的全局平均值，使用0填充")
            
        spatial_data_filled = np.where(np.isnan(spatial_data), global_mean, spatial_data)
        
        # 执行异常检测前再次检查数据是否有效
        if np.isnan(spatial_data_filled).any():
            print(f"警告：特征 '{feature_name}' 仍存在NaN值，使用0替换")
            spatial_data_filled = np.nan_to_num(spatial_data_filled, nan=0)
        
        # 执行异常检测
        anomaly_labels = detector.detect(spatial_data_filled)
        
        # 统计异常点
        total_anomalies = np.sum(anomaly_labels)
        anomaly_positions = np.argwhere(anomaly_labels == 1)
        
        # 准备结果
        results = {
            'status': 'success',
            'feature_name': feature_name,
            'detection_method': detection_method,
            'time_range': {
                'start': start_time,
                'end': end_time,
                'time_points': len(time_indices)
            },
            'statistics': {
                'total_anomalies': total_anomalies,
                'anomaly_ratio': total_anomalies / (self.width * self.height),
                'data_mean': np.mean(spatial_data_filled),
                'data_std': np.std(spatial_data_filled)
            },
            'anomaly_positions': anomaly_positions,
            'anomaly_labels': anomaly_labels,
            'spatial_data': spatial_data
        }
        
        return results
    
    def visualize_results(self, results, save_path=None):
        """可视化异常检测结果
        
        Args:
            results: detect_anomalies方法返回的结果字典
            save_path: 图像保存路径，为None时直接显示
        """
        if results['status'] != 'success':
            print(f"无法可视化结果：{results.get('message', '未知错误')}")
            return
        
        # 设置中文显示
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 绘制原始空间数据
        spatial_data = results['spatial_data']
        im1 = ax1.imshow(spatial_data, cmap='viridis')
        ax1.set_title(f"特征 '{results['feature_name']}' 的空间分布")
        ax1.set_xlabel('Y坐标')
        ax1.set_ylabel('X坐标')
        plt.colorbar(im1, ax=ax1)
        
        # 绘制异常检测结果
        anomaly_labels = results['anomaly_labels']
        im2 = ax2.imshow(anomaly_labels, cmap='Reds')
        
        # 在异常点上标记位置
        anomaly_positions = results['anomaly_positions']
        for pos in anomaly_positions:
            ax2.plot(pos[1], pos[0], 'bo', markersize=5, fillstyle='none')
            ax2.annotate(f'({pos[0]},{pos[1]})', 
                         (pos[1], pos[0]), 
                         textcoords="offset points", 
                         xytext=(0,10), 
                         ha='center',
                         fontsize=8)
        
        ax2.set_title(f"使用{results['detection_method']}方法检测到的异常点\n总数: {results['statistics']['total_anomalies']}")
        ax2.set_xlabel('Y坐标')
        ax2.set_ylabel('X坐标')
        plt.colorbar(im2, ax=ax2, ticks=[0, 1], label='异常标签')
        
        plt.suptitle(f"时间段: {results['time_range']['start']} 到 {results['time_range']['end']}")
        plt.tight_layout()
        
    def detect_anomalies_for_each_time(self, feature_name, start_time, end_time,
                                      detection_method='otsu', threshold_factor=1.0, sigma_threshold=3.0, **method_kwargs):
        """对指定特征和时间段内的每个时刻进行空间异常检测
        
        Args:
            feature_name: 特征名称
            start_time: 开始时间
            end_time: 结束时间
            detection_method: 异常检测方法 ('otsu', '3sigma', 'lof')
            threshold_factor: Otsu方法的阈值调整因子，大于1使阈值更高更严格，小于1使阈值更低更宽松
            sigma_threshold: 3σ方法的标准差倍数，用于调整异常检测的严格程度
            **method_kwargs: 检测方法的额外参数
            
        Returns:
            检测结果字典
        """
        # 验证特征名称
        if feature_name not in self.feature_names:
            raise ValueError(f"无效的特征名称：{feature_name}\n可用特征：{self.feature_names}")
        
        # 获取特征索引
        feature_idx = self.feature_names.index(feature_name)
        
        # 检查特征数据是否全部为NaN
        feature_data = self.data[:, :, feature_idx, :]
        if np.isnan(feature_data).all():
            print(f"错误：特征 '{feature_name}' 的所有值都是NaN，无法进行异常检测")
            return {'status': 'error', 'message': f'特征 {feature_name} 的所有值都是NaN，无法进行异常检测'}
        
        # 检查特征数据是否同质化严重（标准差为0）
        non_nan_data = feature_data[~np.isnan(feature_data)]
        if len(non_nan_data) > 0 and np.std(non_nan_data) == 0:
            print(f"警告：特征 '{feature_name}' 的数据同质化严重（所有非NaN值相同），异常检测结果可能不准确")
        
        # 获取时间索引
        time_indices = self.get_time_indices(start_time, end_time)
        
        if len(time_indices) == 0:
            return {'status': 'error', 'message': '没有找到匹配的时间数据'}
        
        # 初始化检测器并传递相应的参数
        if detection_method == 'otsu':
            detector = SpatialAnomalyDetector(method=detection_method, threshold_factor=threshold_factor, **method_kwargs)
        elif detection_method == '3sigma':
            detector = SpatialAnomalyDetector(method=detection_method, sigma_threshold=sigma_threshold, **method_kwargs)
        else:
            detector = SpatialAnomalyDetector(method=detection_method, **method_kwargs)
        
        # 初始化结果存储
        all_anomaly_labels = []
        all_spatial_data = []
        all_anomaly_counts = []
        
        # 对每个时间点进行异常检测
        for time_idx in time_indices:
            # 获取该时间点的空间数据
            spatial_data = self.data[:, :, feature_idx, time_idx]
            
            # 处理NaN值
            # 将NaN值替换为该特征的全局平均值（忽略NaN）
            feature_data = self.data[:, :, feature_idx, :]
            if np.isnan(feature_data).all():
                # 如果该特征所有值都是NaN，使用0填充
                print(f"警告：特征 '{feature_name}' 的所有值都是NaN，使用0填充")
                global_mean = 0
            else:
                global_mean = np.nanmean(feature_data)
                
            # 再次检查global_mean是否为NaN
            if np.isnan(global_mean):
                global_mean = 0
                print(f"警告：无法计算特征 '{feature_name}' 的全局平均值，使用0填充")
                
            spatial_data_filled = np.where(np.isnan(spatial_data), global_mean, spatial_data)
            
            # 执行异常检测前再次检查数据是否有效
            if np.isnan(spatial_data_filled).any():
                print(f"警告：特征 '{feature_name}' 在时间点 {time_idx} 仍存在NaN值，使用0替换")
                spatial_data_filled = np.nan_to_num(spatial_data_filled, nan=0)
            
            # 执行异常检测
            anomaly_labels = detector.detect(spatial_data_filled)
            
            # 统计异常点数量
            anomaly_count = np.sum(anomaly_labels)
            
            # 存储结果
            all_anomaly_labels.append(anomaly_labels)
            all_spatial_data.append(spatial_data)
            all_anomaly_counts.append(anomaly_count)
        
        # 将列表转换为numpy数组
        all_anomaly_labels = np.array(all_anomaly_labels)
        all_spatial_data = np.array(all_spatial_data)
        all_anomaly_counts = np.array(all_anomaly_counts)
        
        # 准备结果
        results = {
            'status': 'success',
            'feature_name': feature_name,
            'detection_method': detection_method,
            'threshold_factor': threshold_factor if detection_method == 'otsu' else None,
            'sigma_threshold': sigma_threshold if detection_method == '3sigma' else None,
            'time_range': {
                'start': start_time,
                'end': end_time,
                'time_points': len(time_indices),
                'time_indices': time_indices,
                'timestamps': self.timestamps[time_indices]
            },
            'statistics': {
                'total_time_points': len(time_indices),
                'mean_anomaly_count': np.mean(all_anomaly_counts),
                'max_anomaly_count': np.max(all_anomaly_counts),
                'min_anomaly_count': np.min(all_anomaly_counts),
                'time_points_with_anomalies': np.sum(all_anomaly_counts > 0)
            },
            'anomaly_labels_per_time': all_anomaly_labels,  # 形状: (时间点数量, width, height)
            'spatial_data_per_time': all_spatial_data,      # 形状: (时间点数量, width, height)
            'anomaly_counts_per_time': all_anomaly_counts
        }
        
        return results
        
    def visualize_time_series_results(self, results, save_path=None):
        """可视化按时间序列的异常检测结果
        
        Args:
            results: detect_anomalies_for_each_time方法返回的结果字典
            save_path: 图像保存路径，为None时直接显示
        """
        if results['status'] != 'success':
            print(f"无法可视化结果：{results.get('message', '未知错误')}")
            return
        
        # 设置中文显示
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图形
        fig = plt.figure(figsize=(16, 10))
        
        # 第一个子图：异常点数量随时间变化
        ax1 = fig.add_subplot(221)
        timestamps = results['time_range']['timestamps']
        anomaly_counts = results['anomaly_counts_per_time']
        
        # 将numpy datetime64转换为matplotlib可识别的格式
        dates = [datetime.utcfromtimestamp(ts.astype('datetime64[s]').astype(int))
                 if isinstance(ts, np.datetime64) or isinstance(ts, int) else ts
                 for ts in timestamps]
        
        ax1.plot(dates, anomaly_counts, 'o-', color='red')
        ax1.set_title('异常点数量随时间变化')
        ax1.set_xlabel('时间')
        ax1.set_ylabel('异常点数量')
        ax1.grid(True)
        
        # 自动调整时间标签
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 找到异常点数量最多的三个时间点
        top3_indices = np.argsort(anomaly_counts)[-3:][::-1]  # 获取异常点最多的三个时间点索引
        
        # 绘制这三个时间点的空间分布和异常检测结果
        for i, top_idx in enumerate(top3_indices):
            # 第二个子图开始是空间分布图
            ax_spatial = fig.add_subplot(2, 3, 3 + i + 1)
            
            # 绘制空间数据
            spatial_data = results['spatial_data_per_time'][top_idx]
            im = ax_spatial.imshow(spatial_data, cmap='viridis')
            ax_spatial.set_title(f"时间点 {top_idx} 的空间分布\n异常点: {anomaly_counts[top_idx]}")
            ax_spatial.set_xlabel('Y坐标')
            ax_spatial.set_ylabel('X坐标')
            plt.colorbar(im, ax=ax_spatial)
            
            # 在异常点上标记位置
            anomaly_positions = np.argwhere(results['anomaly_labels_per_time'][top_idx] == 1)
            for pos in anomaly_positions:
                ax_spatial.plot(pos[1], pos[0], 'ro', markersize=5, fillstyle='none')
        
        # 统计信息子图
        ax_stats = fig.add_subplot(222)
        stats = results['statistics']
        
        # 创建统计信息文本
        stats_text = (
            f"总时间点数: {stats['total_time_points']}\n" \
            f"平均异常点数: {stats['mean_anomaly_count']:.2f}\n" \
            f"最大异常点数: {stats['max_anomaly_count']}\n" \
            f"最小异常点数: {stats['min_anomaly_count']}\n" \
            f"存在异常的时间点数: {stats['time_points_with_anomalies']}"
        )
        
        ax_stats.text(0.5, 0.5, stats_text, fontsize=12, 
                     horizontalalignment='center', verticalalignment='center')
        ax_stats.set_title('统计信息')
        ax_stats.axis('off')
        
        plt.suptitle(f"特征: {results['feature_name']}\n时间段: {results['time_range']['start']} 到 {results['time_range']['end']}")
        plt.tight_layout()
        
        if save_path:
            # 检查是否需要创建目录
            dir_name = os.path.dirname(save_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"时间序列结果图像已保存至：{save_path}")
        else:
            plt.show()

    def save_anomaly_labels_to_file(self, results, output_file):
        """保存每个时间点的异常标签到文件
        
        Args:
            results: detect_anomalies_for_each_time方法返回的结果字典
            output_file: 输出文件路径
        """
        if results['status'] != 'success':
            print(f"无法保存结果：{results.get('message', '未知错误')}")
            return
        
        # 准备数据
        anomaly_labels = results['anomaly_labels_per_time']
        timestamps = results['time_range']['timestamps']
        
        # 创建一个结构化数组来存储时间戳和异常标签
        data_to_save = {
            'timestamps': timestamps,
            'anomaly_labels': anomaly_labels,
            'feature_name': results['feature_name'],
            'detection_method': results['detection_method']
        }
        
        # 保存到文件
        np.savez_compressed(output_file, **data_to_save)
        print(f"每个时间点的异常标签已保存至：{output_file}")

# 命令行接口
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='时间段空间异常检测器')
    parser.add_argument('--feature', type=str, required=True, help='要检测的特征名称')
    parser.add_argument('--start-time', type=str, required=True, help='开始时间 (格式: YYYY-MM-DDTHH:MM:SS)')
    parser.add_argument('--end-time', type=str, required=True, help='结束时间 (格式: YYYY-MM-DDTHH:MM:SS)')
    parser.add_argument('--method', type=str, default='otsu', choices=['otsu', '3sigma', 'lof'], help='异常检测方法')
    parser.add_argument('--threshold-factor', type=float, default=1.0, help='Otsu方法的阈值调整因子，大于1使阈值更高更严格，小于1使阈值更低更宽松')
    parser.add_argument('--sigma-threshold', type=float, default=3.0, help='3σ方法的标准差倍数，用于调整异常检测的严格程度')
    parser.add_argument('--output', type=str, default=None, help='结果图像保存路径')
    parser.add_argument('--per-time', action='store_true', help='是否对每个时间点单独进行检测')
    parser.add_argument('--save-labels', type=str, default=None, help='保存异常标签的文件路径 (.npz格式)')
    
    args = parser.parse_args()
    
    try:
        # 初始化检测器
        detector = TimeRangeSpatialAnomalyDetector()
        
        if args.per_time:
            # 对每个时间点单独进行检测
            print(f"\n对每个时间点单独进行异常检测...")
            results = detector.detect_anomalies_for_each_time(
                feature_name=args.feature,
                start_time=args.start_time,
                end_time=args.end_time,
                detection_method=args.method,
                threshold_factor=args.threshold_factor,
                sigma_threshold=args.sigma_threshold
            )
            
            if results['status'] == 'success':
                # 打印结果统计信息
                print(f"\n异常检测结果:")
                print(f"  特征: {results['feature_name']}")
                print(f"  检测方法: {results['detection_method']}")
                print(f"  时间段: {results['time_range']['start']} 到 {results['time_range']['end']}")
                print(f"  总时间点数: {results['statistics']['total_time_points']}")
                print(f"  平均异常点数: {results['statistics']['mean_anomaly_count']:.2f}")
                print(f"  最大异常点数: {results['statistics']['max_anomaly_count']}")
                print(f"  最小异常点数: {results['statistics']['min_anomaly_count']}")
                print(f"  存在异常的时间点数: {results['statistics']['time_points_with_anomalies']}")
                
                # 可视化时间序列结果
                detector.visualize_time_series_results(results, save_path=args.output)
                
                # 保存异常标签
                if args.save_labels:
                    detector.save_anomaly_labels_to_file(results, args.save_labels)
        else:
            # 对时间段内的数据进行平均后检测
            results = detector.detect_anomalies(
                feature_name=args.feature,
                start_time=args.start_time,
                end_time=args.end_time,
                detection_method=args.method,
                threshold_factor=args.threshold_factor,
                sigma_threshold=args.sigma_threshold
            )
            
            if results['status'] == 'success':
                # 打印结果统计信息
                print(f"\n异常检测结果:")
                print(f"  特征: {results['feature_name']}")
                print(f"  检测方法: {results['detection_method']}")
                print(f"  时间段: {results['time_range']['start']} 到 {results['time_range']['end']}")
                print(f"  包含 {results['time_range']['time_points']} 个时间点")
                print(f"  异常点数量: {results['statistics']['total_anomalies']}")
                print(f"  异常点比例: {results['statistics']['anomaly_ratio']*100:.2f}%")
                print(f"  数据平均值: {results['statistics']['data_mean']:.4f}")
                print(f"  数据标准差: {results['statistics']['data_std']:.4f}")
                
                if len(results['anomaly_positions']) > 0:
                    print(f"\n异常点位置 (x,y):")
                    for pos in results['anomaly_positions']:
                        print(f"    ({pos[0]}, {pos[1]})")
                
                # 可视化结果
                detector.visualize_results(results, save_path=args.output)
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()