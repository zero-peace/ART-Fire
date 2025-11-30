#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量时间序列预测脚本
处理文件夹中的所有CSV文件，进行批量预测
"""

import os
import glob
import pandas as pd
from datetime import datetime
import argparse
import torch
from predict_timeseries_v2 import TimeSeriesPredictorV2
import warnings
warnings.filterwarnings('ignore')

class BatchPredictor:
    def __init__(self, model_path='Maple728/TimeMoE-200M', device='cuda'):
        """
        初始化批量预测器
        
        Args:
            model_path: 预训练模型路径
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")
        
        # 初始化时清理内存
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            print(f"已清理GPU缓存")
        
        # 创建预测器
        print("正在加载模型...")
        self.predictor = TimeSeriesPredictorV2(model_path=model_path, device=self.device)
        print("模型加载完成!")
        
    def find_csv_files(self, folder_path):
        """
        查找文件夹中的所有CSV文件
        
        Args:
            folder_path: 文件夹路径
            
        Returns:
            list: CSV文件路径列表
        """
        # 支持多种CSV文件模式
        patterns = ['*.csv', '*.CSV']
        csv_files = []
        
        for pattern in patterns:
            csv_files.extend(glob.glob(os.path.join(folder_path, pattern)))
            csv_files.extend(glob.glob(os.path.join(folder_path, '**', pattern), recursive=True))
        
        # 去重并排序
        csv_files = sorted(list(set(csv_files)))
        
        print(f"在文件夹 {folder_path} 中找到 {len(csv_files)} 个CSV文件:")
        for file in csv_files:
            print(f"  - {os.path.basename(file)}")
        
        return csv_files
    
    def get_output_dir(self, csv_path, base_output_dir):
        """
        根据CSV文件路径生成输出目录
        
        Args:
            csv_path: CSV文件路径
            base_output_dir: 基础输出目录
            
        Returns:
            str: 输出目录路径
        """
        # 获取文件名（不含扩展名）
        file_name = os.path.splitext(os.path.basename(csv_path))[0]
        
        # 创建输出目录：原文件名 + fire
        output_dir = os.path.join(base_output_dir, f"{file_name}_fire")
        
        return output_dir
    
    def validate_csv_file(self, csv_path, bands, timestamp_col='timestamp'):
        """
        验证CSV文件是否有效
        
        Args:
            csv_path: CSV文件路径
            bands: 预测波段列表
            timestamp_col: 时间戳列名
            
        Returns:
            bool: 是否有效
        """
        try:
            # 尝试加载数据
            df = pd.read_csv(csv_path)
            
            # 检查必需的列
            if timestamp_col not in df.columns:
                print(f"  警告: 文件 {os.path.basename(csv_path)} 缺少时间戳列 '{timestamp_col}'")
                return False
            
            # 检查预测波段
            missing_bands = [band for band in bands if band not in df.columns]
            if missing_bands:
                print(f"  警告: 文件 {os.path.basename(csv_path)} 缺少波段: {missing_bands}")
                return False
            
            # 检查数据行数
            if len(df) < 10:
                print(f"  警告: 文件 {os.path.basename(csv_path)} 数据行数过少: {len(df)}")
                return False
            
            return True
            
        except Exception as e:
            print(f"  错误: 无法读取文件 {os.path.basename(csv_path)}: {str(e)}")
            return False
    
    def predict_single_file(self, csv_path, bands, start_timestamp, end_timestamp, 
                           use_len, prediction_steps, output_dir, timestamp_col='timestamp'):
        """
        预测单个文件
        
        Args:
            csv_path: CSV文件路径
            bands: 预测波段列表
            start_timestamp: 开始时间戳
            end_timestamp: 结束时间戳
            use_len: 用于预测的长度
            prediction_steps: 预测步数
            output_dir: 输出目录
            timestamp_col: 时间戳列名
            
        Returns:
            dict: 预测结果
        """
        print(f"\n开始预测文件: {os.path.basename(csv_path)}")
        print(f"输出目录: {output_dir}")
        
        try:
            # 清理GPU缓存
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                print(f"  已清理GPU缓存")
            
            # 运行预测
            results = self.predictor.run_prediction(
                csv_path=csv_path,
                bands=bands,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                use_len=use_len,
                prediction_steps=prediction_steps,
                timestamp_col=timestamp_col,
                save_plot=True,
                output_dir=output_dir
            )
            
            # 预测完成后再次清理缓存
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            print(f"文件 {os.path.basename(csv_path)} 预测完成!")
            return results
            
        except Exception as e:
            print(f"文件 {os.path.basename(csv_path)} 预测失败: {str(e)}")
            # 发生错误时也清理缓存
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            return None
    
    def run_batch_prediction(self, folder_path, bands, start_timestamp, end_timestamp,
                           use_len=144, prediction_steps=1, timestamp_col='timestamp',
                           base_output_dir='./batch_predictions', skip_existing=True):
        """
        运行批量预测
        
        Args:
            folder_path: 输入文件夹路径
            bands: 预测波段列表
            start_timestamp: 开始时间戳
            end_timestamp: 结束时间戳
            use_len: 用于预测的长度
            prediction_steps: 预测步数
            timestamp_col: 时间戳列名
            base_output_dir: 基础输出目录
            
        Returns:
            dict: 批量预测结果统计
        """
        print("=" * 80)
        print("批量时间序列预测")
        print("=" * 80)
        print(f"输入文件夹: {folder_path}")
        print(f"预测波段: {bands}")
        print(f"时间范围: {start_timestamp} 到 {end_timestamp}")
        print(f"使用长度: {use_len}, 预测步数: {prediction_steps}")
        print(f"基础输出目录: {base_output_dir}")
        
        # 查找CSV文件
        csv_files = self.find_csv_files(folder_path)
        
        if not csv_files:
            print("未找到任何CSV文件!")
            return {}
        
        # 创建基础输出目录
        os.makedirs(base_output_dir, exist_ok=True)
        
        # 统计信息
        total_files = len(csv_files)
        success_count = 0
        failed_count = 0
        skipped_count = 0
        results_summary = {}
        
        print(f"\n开始批量预测，共 {total_files} 个文件...")
        
        # 逐个处理文件
        for i, csv_path in enumerate(csv_files, 1):
            print(f"\n[{i}/{total_files}] 处理文件: {os.path.basename(csv_path)}")
            
            # 验证文件
            if not self.validate_csv_file(csv_path, bands, timestamp_col):
                print(f"  跳过无效文件: {os.path.basename(csv_path)}")
                failed_count += 1
                continue
            
            # 生成输出目录
            output_dir = self.get_output_dir(csv_path, base_output_dir)
            
            # 检查是否需要跳过该文件
            if skip_existing and self.should_skip_file(output_dir, bands):
                print(f"  跳过已完成的文件: {os.path.basename(csv_path)}")
                skipped_count += 1
                results_summary[os.path.basename(csv_path)] = {
                    'status': 'skipped',
                    'output_dir': output_dir,
                    'bands': bands
                }
                continue
            
            # 检查缺失的波段
            if skip_existing:
                missing_bands = self.get_missing_bands(output_dir, bands)
                if missing_bands:
                    print(f"  文件 {os.path.basename(csv_path)} 缺失波段: {missing_bands}")
                    print(f"  将预测缺失的波段: {missing_bands}")
                    # 只预测缺失的波段
                    bands_to_predict = missing_bands
                else:
                    bands_to_predict = bands
            else:
                bands_to_predict = bands
            
            # 预测单个文件
            result = self.predict_single_file(
                csv_path=csv_path,
                bands=bands_to_predict,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                use_len=use_len,
                prediction_steps=prediction_steps,
                output_dir=output_dir,
                timestamp_col=timestamp_col
            )
            
            if result is not None:
                success_count += 1
                results_summary[os.path.basename(csv_path)] = {
                    'status': 'success',
                    'output_dir': output_dir,
                    'bands': list(result.keys())
                }
            else:
                failed_count += 1
                results_summary[os.path.basename(csv_path)] = {
                    'status': 'failed',
                    'output_dir': output_dir
                }
            
            # 每处理几个文件后强制清理一次内存
            if i % 5 == 0:
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                    print(f"  已清理GPU缓存 (处理了 {i} 个文件)")
                import gc
                gc.collect()
                print(f"  已清理Python内存")
        
        # 打印总结
        print("\n" + "=" * 80)
        print("批量预测完成!")
        print("=" * 80)
        print(f"总文件数: {total_files}")
        print(f"成功: {success_count}")
        print(f"跳过: {skipped_count}")
        print(f"失败: {failed_count}")
        print(f"成功率: {(success_count + skipped_count)/total_files*100:.1f}%")
        print(f"结果保存在: {base_output_dir}")
        
        # 保存批量预测总结
        summary_file = os.path.join(base_output_dir, 'batch_prediction_summary.csv')
        summary_data = []
        for file_name, info in results_summary.items():
            summary_data.append({
                'file_name': file_name,
                'status': info['status'],
                'output_dir': info.get('output_dir', ''),
                'bands': ', '.join(info.get('bands', []))
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)
        print(f"预测总结已保存到: {summary_file}")
        
        return results_summary
    
    def check_prediction_files_exist(self, output_dir, bands):
        """
        检查预测结果文件是否已存在
        
        Args:
            output_dir: 输出目录路径
            bands: 预测波段列表
            
        Returns:
            dict: 每个波段的文件存在状态
        """
        if not os.path.exists(output_dir):
            return {band: False for band in bands}
        
        file_status = {}
        for band in bands:
            # 检查两个必需的文件：结果CSV和预测图
            csv_file = os.path.join(output_dir, f"{band}_prediction_results.csv")
            png_file = os.path.join(output_dir, f"{band}_prediction.png")
            
            csv_exists = os.path.exists(csv_file) and os.path.getsize(csv_file) > 0
            png_exists = os.path.exists(png_file) and os.path.getsize(png_file) > 0
            
            file_status[band] = csv_exists and png_exists
        
        return file_status
    
    def get_missing_bands(self, output_dir, bands):
        """
        获取缺失的波段列表
        
        Args:
            output_dir: 输出目录路径
            bands: 预测波段列表
            
        Returns:
            list: 缺失的波段列表
        """
        file_status = self.check_prediction_files_exist(output_dir, bands)
        missing_bands = [band for band, exists in file_status.items() if not exists]
        return missing_bands
    
    def should_skip_file(self, output_dir, bands):
        """
        判断是否应该跳过该文件的预测
        
        Args:
            output_dir: 输出目录路径
            bands: 预测波段列表
            
        Returns:
            bool: 是否应该跳过
        """
        missing_bands = self.get_missing_bands(output_dir, bands)
        return len(missing_bands) == 0  # 如果没有缺失的波段，则跳过

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量时间序列预测工具')
    parser.add_argument('--folder_path', type=str, required=True, help='输入文件夹路径')
    parser.add_argument('--bands', type=str, nargs='+', required=True, help='预测波段列表')
    parser.add_argument('--start_timestamp', type=str, required=True, help='开始时间戳 (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--end_timestamp', type=str, required=True, help='结束时间戳 (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--use_len', type=int, default=288, help='用于预测的长度')
    parser.add_argument('--prediction_steps', type=int, default=1, help='预测步数')
    parser.add_argument('--timestamp_col', type=str, default='timestamp', help='时间戳列名')
    parser.add_argument('--output_dir', type=str, default='./batch_predictions', help='输出目录')
    parser.add_argument('--model_path', type=str, default='Maple728/TimeMoE-200M', help='模型路径')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    parser.add_argument('--skip_existing', action='store_true', default=True, help='跳过已存在的文件')
    parser.add_argument('--no_skip_existing', action='store_false', dest='skip_existing', help='不跳过已存在的文件')
    
    args = parser.parse_args()
    
    # 检查输入文件夹是否存在
    if not os.path.exists(args.folder_path):
        print(f"错误: 输入文件夹不存在: {args.folder_path}")
        return
    
    # 转换时间戳
    try:
        start_timestamp = pd.to_datetime(args.start_timestamp)
        end_timestamp = pd.to_datetime(args.end_timestamp)
    except Exception as e:
        print(f"错误: 时间戳格式无效: {str(e)}")
        return
    
    # 创建批量预测器
    batch_predictor = BatchPredictor(args.model_path, args.device)
    
    # 运行批量预测
    try:
        results = batch_predictor.run_batch_prediction(
            folder_path=args.folder_path,
            bands=args.bands,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            use_len=args.use_len,
            prediction_steps=args.prediction_steps,
            timestamp_col=args.timestamp_col,
            base_output_dir=args.output_dir,
            skip_existing=args.skip_existing
        )
        
        print(f"\n批量预测完成! 结果保存在: {args.output_dir}")
        
    except Exception as e:
        print(f"批量预测过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    import torch
    main() 