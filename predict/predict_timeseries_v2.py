#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间序列预测工具 V2
根据用户需求重新设计的预测逻辑
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesPredictorV2:
    def __init__(self, model_path='Maple728/TimeMoE-200M', device='cuda'):
        """
        初始化时间序列预测器
        
        Args:
            model_path: 预训练模型路径
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")
        
        # 加载模型
        print("正在加载模型...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=self.device,
            trust_remote_code=True,
        )
        print("模型加载完成!")
        
    def load_data(self, csv_path, timestamp_col='timestamp'):
        """
        加载CSV数据
        
        Args:
            csv_path: CSV文件路径
            timestamp_col: 时间戳列名
            
        Returns:
            pandas.DataFrame: 加载的数据
        """
        print(f"正在加载数据: {csv_path}")
        try:
            # 尝试解析时间戳
            df = pd.read_csv(csv_path, parse_dates=[timestamp_col])
        except:
            # 如果解析失败，先加载再转换
            df = pd.read_csv(csv_path)
            if timestamp_col in df.columns:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        print(f"数据加载完成，共 {len(df)} 行")
        print(f"列名: {list(df.columns)}")
        return df
    
    def find_time_indices(self, df, start_timestamp, end_timestamp, timestamp_col='timestamp'):
        """
        找到时间戳对应的索引
        
        Args:
            df: 数据框
            start_timestamp: 开始时间戳
            end_timestamp: 结束时间戳
            timestamp_col: 时间戳列名
            
        Returns:
            tuple: (开始索引, 结束索引)
        """
        # 确保时间戳列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # 找到时间戳对应的索引
        start_idx = df[df[timestamp_col] >= start_timestamp].index[0] if len(df[df[timestamp_col] >= start_timestamp]) > 0 else 0
        end_idx = df[df[timestamp_col] <= end_timestamp].index[-1] if len(df[df[timestamp_col] <= end_timestamp]) > 0 else len(df) - 1
        
        print(f"时间范围索引: {start_idx} 到 {end_idx}")
        print(f"对应时间: {df[timestamp_col].iloc[start_idx]} 到 {df[timestamp_col].iloc[end_idx]}")
        
        return start_idx, end_idx
    
    def prepare_sequence(self, data, band, normalize=True):
        """
        准备序列数据
        
        Args:
            data: 数据框或数组
            band: 预测波段列名
            normalize: 是否标准化
            
        Returns:
            tuple: (标准化后的序列, 均值, 标准差)
        """
        if isinstance(data, pd.DataFrame):
            seqs = torch.tensor(data[band].values, dtype=torch.float32)
        else:
            seqs = torch.tensor(data, dtype=torch.float32)
        
        seqs = seqs.to(self.device)
        seqs = seqs.reshape(1, len(seqs))
        
        if normalize:
            mean = seqs.mean(dim=-1, keepdim=True)
            std = seqs.std(dim=-1, keepdim=True)
            normed_seqs = (seqs - mean) / std
            return normed_seqs, mean, std
        else:
            return seqs, None, None
    
    def predict_single_step(self, input_data, prediction_steps=1):
        """
        单次预测
        
        Args:
            input_data: 输入数据 (标准化后的)
            prediction_steps: 预测步数
            
        Returns:
            torch.Tensor: 预测结果（只取最后一步）
        """
        with torch.no_grad():
            output = self.model.generate(input_data, max_new_tokens=prediction_steps)
        
        # 只取最后一步的预测结果
        predictions = output[:, -1:]
        return predictions
    
    def predict_time_range(self, df, band, start_idx, end_idx, use_len, prediction_steps):
        """
        预测指定时间范围
        
        Args:
            df: 数据框
            band: 预测波段
            start_idx: 预测开始索引
            end_idx: 预测结束索引
            use_len: 用于预测的长度
            prediction_steps: 预测步数
            
        Returns:
            tuple: (预测结果, 原始数据, 时间戳)
        """
        print(f"开始预测波段: {band}")
        print(f"预测范围: 索引 {start_idx} 到 {end_idx}")
        print(f"使用长度: {use_len}, 预测步数: {prediction_steps}")
        
        # 计算预测起始位置
        predict_start_idx = start_idx - (use_len + prediction_steps - 1)
        if predict_start_idx < 0:
            predict_start_idx = 0
            print(f"警告: 预测起始位置调整为 0")
        
        print(f"预测起始位置: {predict_start_idx}")
        
        # 准备序列数据
        normed_seqs, mean, std = self.prepare_sequence(df, band)
        
        # 存储预测结果
        all_predictions = []
        prediction_timestamps = []
        
        # 循环预测
        current_idx = start_idx
        while current_idx <= end_idx:
            # 计算输入数据的起始和结束索引
            input_start_idx = current_idx - use_len
            input_end_idx = current_idx
            
            if input_start_idx < 0:
                input_start_idx = 0
            
            # 提取输入数据
            input_data = normed_seqs[:, input_start_idx:input_end_idx]
            
            # 进行预测
            prediction = self.predict_single_step(input_data, prediction_steps)
            
            # 反标准化
            if mean is not None and std is not None:
                prediction = prediction * std + mean
            
            # 添加到结果列表
            all_predictions.append(prediction[0, 0].cpu().numpy())
            prediction_timestamps.append(df['timestamp'].iloc[current_idx])
            
            # 移动到下一个位置
            current_idx += 1
        
        # 转换为numpy数组
        predictions = np.array(all_predictions)
        original_data = df[band].iloc[start_idx:end_idx+1].values
        
        print(f"预测完成，共预测 {len(predictions)} 个时间点")
        
        return predictions, original_data, prediction_timestamps
    
    def calculate_metrics(self, y_true, y_pred):
        """
        计算评估指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            dict: 评估指标字典
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # 确保长度一致
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # 计算各种指标
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # 改进的MAPE计算 (忽略0值和极小值)
        # 使用更严格的阈值，只考虑有意义的非零值
        threshold = 0.001  # 固定阈值，忽略小于0.001的值
        mask = y_true > threshold
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else 0
        
        # 改进的SMAPE计算 (忽略0值)
        # 只考虑真实值和预测值都不为0的情况
        mask_smape = (y_true > threshold) & (y_pred > threshold)
        smape = np.mean(2 * np.abs(y_true[mask_smape] - y_pred[mask_smape]) / (y_true[mask_smape] + y_pred[mask_smape])) * 100 if np.any(mask_smape) else 0
        
        # 计算R²
        r2 = r2_score(y_true, y_pred)
        
        # 最大误差
        max_error = np.max(np.abs(y_true - y_pred))
        
        # 添加数据统计信息用于调试
        zero_count = np.sum(y_true == 0)
        small_count = np.sum(y_true <= threshold)
        
        data_stats = {
            'y_true_min': y_true.min(),
            'y_true_max': y_true.max(),
            'y_true_mean': y_true.mean(),
            'y_pred_min': y_pred.min(),
            'y_pred_max': y_pred.max(),
            'y_pred_mean': y_pred.mean(),
            'abs_error_min': np.abs(y_true - y_pred).min(),
            'abs_error_max': np.abs(y_true - y_pred).max(),
            'abs_error_mean': np.abs(y_true - y_pred).mean(),
            'zero_count': zero_count,
            'small_count': small_count,
            'valid_points_mape': np.sum(mask),
            'valid_points_smape': np.sum(mask_smape),
            'total_points': len(y_true),
            'threshold_used': threshold
        }
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'SMAPE': smape,
            'R2': r2,
            'Max_Error': max_error,
            'Data_Stats': data_stats
        }
        
        return metrics
    
    def plot_results(self, original_data, predictions, timestamps, band, save_path=None):
        """
        绘制预测结果
        
        Args:
            original_data: 原始数据
            predictions: 预测数据
            timestamps: 时间戳
            band: 波段名称
            save_path: 保存路径
        """
        plt.figure(figsize=(15, 8))
        
        # 绘制原始数据
        plt.plot(original_data, label='原始数据', color='blue', alpha=0.7)
        
        # 绘制预测数据
        plt.plot(predictions, label='预测数据', color='red', alpha=0.8)
        
        plt.title(f'{band} 时间序列预测结果')
        plt.xlabel('时间步')
        plt.ylabel('值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            # 确保输出目录存在
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        #plt.show()
    
    def run_prediction(self, csv_path, bands, start_timestamp, end_timestamp, 
                      use_len=288, prediction_steps=1, timestamp_col='timestamp', 
                      save_plot=True, output_dir='./'):
        """
        运行完整的预测流程
        
        Args:
            csv_path: CSV文件路径
            bands: 预测波段列表
            start_timestamp: 开始时间戳
            end_timestamp: 结束时间戳
            use_len: 用于预测的长度
            prediction_steps: 预测步数
            timestamp_col: 时间戳列名
            save_plot: 是否保存图表
            output_dir: 输出目录
            
        Returns:
            dict: 预测结果和指标
        """
        # 加载数据
        df = self.load_data(csv_path, timestamp_col)
        
        # 找到时间索引
        start_idx, end_idx = self.find_time_indices(df, start_timestamp, end_timestamp, timestamp_col)
        
        if start_idx >= end_idx:
            raise ValueError("预测时间范围无效")
        
        # 创建输出目录
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        for band in bands:
            if band not in df.columns:
                print(f"警告: 波段 {band} 不在数据列中，跳过")
                continue
            
            print(f"\n开始预测波段: {band}")
            
            # 进行预测
            predictions, original_data, timestamps = self.predict_time_range(
                df, band, start_idx, end_idx, use_len, prediction_steps
            )
            
            # 计算指标
            metrics = self.calculate_metrics(original_data, predictions)
            
            # 保存结果
            results[band] = {
                'predictions': predictions,
                'original_data': original_data,
                'timestamps': timestamps,
                'metrics': metrics
            }
            
            # 打印指标
            print(f"\n{band} 预测指标:")
            for metric_name, value in metrics.items():
                if metric_name == 'Data_Stats':
                    print(f"\n{metric_name}:")
                    for stat_name, stat_value in value.items():
                        if stat_name in ['zero_count', 'small_count', 'valid_points_mape', 'valid_points_smape', 'total_points']:
                            print(f"  {stat_name}: {stat_value}")
                        elif stat_name == 'threshold_used':
                            print(f"  {stat_name}: {stat_value:.6f}")
                        else:
                            print(f"  {stat_name}: {stat_value:.6f}")
                elif metric_name in ['MSE']:
                    print(f"{metric_name}: {value:.10f}")
                elif metric_name in ['MAE', 'RMSE', 'Max_Error']:
                    print(f"{metric_name}: {value:.8f}")
                elif metric_name in ['MAPE', 'SMAPE']:
                    print(f"{metric_name}: {value:.4f}%")
                elif metric_name == 'R2':
                    print(f"{metric_name}: {value:.4f}")
                else:
                    print(f"{metric_name}: {value:.4f}")
            
            # 绘制结果
            if save_plot:
                plot_path = f"{output_dir}/{band}_prediction.png"
                self.plot_results(original_data, predictions, timestamps, band, plot_path)
            
            # 保存预测结果到CSV
            result_df = pd.DataFrame({
                'timestamp': timestamps,
                'original': original_data[:len(predictions)],
                'predicted': predictions
            })
            csv_path = f"{output_dir}/{band}_prediction_results.csv"
            result_df.to_csv(csv_path, index=False)
            print(f"预测结果已保存到: {csv_path}")
        
        return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='时间序列预测工具 V2')
    parser.add_argument('--csv_path', type=str, required=True, help='CSV文件路径')
    parser.add_argument('--bands', type=str, nargs='+', required=True, help='预测波段列表')
    parser.add_argument('--start_timestamp', type=str, required=True, help='开始时间戳 (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--end_timestamp', type=str, required=True, help='结束时间戳 (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--use_len', type=int, default=288, help='用于预测的长度')
    parser.add_argument('--prediction_steps', type=int, default=1, help='预测步数')
    parser.add_argument('--timestamp_col', type=str, default='timestamp', help='时间戳列名')
    parser.add_argument('--output_dir', type=str, default='./predictions', help='输出目录')
    parser.add_argument('--model_path', type=str, default='Maple728/TimeMoE-200M', help='模型路径')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    
    args = parser.parse_args()
    
    # 转换时间戳
    start_timestamp = pd.to_datetime(args.start_timestamp)
    end_timestamp = pd.to_datetime(args.end_timestamp)
    
    # 创建预测器
    predictor = TimeSeriesPredictorV2(args.model_path, args.device)
    
    # 运行预测
    try:
        results = predictor.run_prediction(
            csv_path=args.csv_path,
            bands=args.bands,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            use_len=args.use_len,
            prediction_steps=args.prediction_steps,
            timestamp_col=args.timestamp_col,
            save_plot=True,
            output_dir=args.output_dir
        )
        
        print(f"\n预测完成! 结果保存在: {args.output_dir}")
        
    except Exception as e:
        print(f"预测过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    main() 