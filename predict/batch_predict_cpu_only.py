#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU专用批量预测脚本
专门用于GPU内存不足的情况，强制使用CPU进行预测
"""

import os
import sys
import torch
import gc
import time
from batch_predict import BatchPredictor

def get_memory_info():
    """获取内存使用情况"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        return f"GPU: {gpu_memory:.2f}GB (已分配) / {gpu_memory_reserved:.2f}GB (已保留)"
    else:
        return "CPU模式"

def force_memory_cleanup():
    """强制内存清理"""
    print("🧹 执行内存清理...")
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            print("  ✓ GPU缓存已清理")
        except Exception as e:
            print(f"  ⚠ GPU缓存清理失败: {e}")
    
    # 清理Python内存
    try:
        gc.collect()
        print("  ✓ Python内存已清理")
    except Exception as e:
        print(f"  ⚠ Python内存清理失败: {e}")

class CPUOnlyBatchPredictor:
    def __init__(self, model_path='Maple728/TimeMoE-200M'):
        """
        CPU专用批量预测器
        
        Args:
            model_path: 预训练模型路径
        """
        print(f"🔧 初始化CPU专用预测器...")
        print(f"💻 强制使用CPU设备")
        
        # 强制清理内存
        force_memory_cleanup()
        
        # 创建预测器，强制使用CPU
        print("📥 正在加载模型到CPU...")
        self.predictor = None
        self._load_model(model_path)
        print("✅ 模型加载完成!")
    
    def _load_model(self, model_path):
        """加载模型到CPU"""
        try:
            # 如果之前有模型，先清理
            if self.predictor is not None:
                del self.predictor
                force_memory_cleanup()
            
            from predict_timeseries_v2 import TimeSeriesPredictorV2
            
            # 强制使用CPU
            self.predictor = TimeSeriesPredictorV2(model_path=model_path, device='cpu')
            print("  ✓ 模型已加载到CPU")
                
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def predict_single_file_cpu(self, csv_path, bands, start_timestamp, end_timestamp, 
                               use_len, prediction_steps, output_dir, timestamp_col='timestamp'):
        """
        CPU单文件预测
        
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
        print(f"\n📁 开始预测文件: {os.path.basename(csv_path)}")
        print(f"📂 输出目录: {output_dir}")
        print(f"💻 使用设备: CPU")
        
        try:
            # 预测前清理内存
            force_memory_cleanup()
            
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
            
            # 预测后清理内存
            force_memory_cleanup()
            
            print(f"✅ 文件 {os.path.basename(csv_path)} 预测完成!")
            return results
            
        except Exception as e:
            print(f"❌ 文件 {os.path.basename(csv_path)} 预测失败: {str(e)}")
            # 发生错误时强制清理内存
            force_memory_cleanup()
            return None
    
    def run_cpu_batch_prediction(self, folder_path, bands, start_timestamp, end_timestamp,
                                use_len=144, prediction_steps=1, timestamp_col='timestamp',
                                base_output_dir='./cpu_only_predictions',
                                batch_size=2, max_retries=2):
        """
        CPU批量预测
        
        Args:
            folder_path: 输入文件夹路径
            bands: 预测波段列表
            start_timestamp: 开始时间戳
            end_timestamp: 结束时间戳
            use_len: 用于预测的长度
            prediction_steps: 预测步数
            timestamp_col: 时间戳列名
            base_output_dir: 基础输出目录
            batch_size: 批处理大小（CPU可以处理更多文件）
            max_retries: 最大重试次数
            
        Returns:
            dict: 批量预测结果统计
        """
        print("🚀 CPU专用批量预测启动")
        print("=" * 60)
        print(f"📁 输入文件夹: {folder_path}")
        print(f"🎯 预测波段: {bands}")
        print(f"⏰ 时间范围: {start_timestamp} 到 {end_timestamp}")
        print(f"📊 使用长度: {use_len}, 预测步数: {prediction_steps}")
        print(f"📦 批处理大小: {batch_size}")
        print(f"🔄 最大重试次数: {max_retries}")
        print(f"💻 强制使用CPU设备")
        
        # 查找CSV文件
        import glob
        csv_files = []
        for pattern in ['*.csv', '*.CSV']:
            csv_files.extend(glob.glob(os.path.join(folder_path, pattern)))
            csv_files.extend(glob.glob(os.path.join(folder_path, '**', pattern), recursive=True))
        csv_files = sorted(list(set(csv_files)))
        
        if not csv_files:
            print("❌ 未找到任何CSV文件!")
            return {}
        
        print(f"📋 找到 {len(csv_files)} 个CSV文件")
        
        # 创建输出目录
        os.makedirs(base_output_dir, exist_ok=True)
        
        # 统计信息
        total_files = len(csv_files)
        success_count = 0
        failed_count = 0
        results_summary = {}
        
        # 分批处理文件
        for batch_start in range(0, total_files, batch_size):
            batch_end = min(batch_start + batch_size, total_files)
            batch_files = csv_files[batch_start:batch_end]
            
            print(f"\n🔄 处理批次 {batch_start//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}")
            print(f"   文件 {batch_start + 1}-{batch_end}/{total_files}")
            
            # 批次开始前清理内存
            force_memory_cleanup()
            
            # 处理当前批次的文件
            for i, csv_path in enumerate(batch_files):
                file_index = batch_start + i + 1
                file_name = os.path.basename(csv_path)
                print(f"\n[{file_index}/{total_files}] 处理文件: {file_name}")
                
                # 验证文件
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_path)
                    
                    # 检查必需的列
                    if timestamp_col not in df.columns:
                        print(f"  ⚠ 跳过无效文件: 缺少时间戳列 '{timestamp_col}'")
                        failed_count += 1
                        results_summary[file_name] = {
                            'status': 'failed',
                            'reason': f'缺少时间戳列 {timestamp_col}'
                        }
                        continue
                    
                    # 检查预测波段
                    missing_bands = [band for band in bands if band not in df.columns]
                    if missing_bands:
                        print(f"  ⚠ 跳过无效文件: 缺少波段 {missing_bands}")
                        failed_count += 1
                        results_summary[file_name] = {
                            'status': 'failed',
                            'reason': f'缺少波段 {missing_bands}'
                        }
                        continue
                    
                    # 检查数据行数
                    if len(df) < 10:
                        print(f"  ⚠ 跳过无效文件: 数据行数过少 ({len(df)})")
                        failed_count += 1
                        results_summary[file_name] = {
                            'status': 'failed',
                            'reason': f'数据行数过少 ({len(df)})'
                        }
                        continue
                        
                except Exception as e:
                    print(f"  ⚠ 跳过无效文件: 无法读取文件 ({str(e)})")
                    failed_count += 1
                    results_summary[file_name] = {
                        'status': 'failed',
                        'reason': f'无法读取文件: {str(e)}'
                    }
                    continue
                
                # 生成输出目录
                file_name_no_ext = os.path.splitext(file_name)[0]
                output_subdir = os.path.join(base_output_dir, f"{file_name_no_ext}_fire")
                
                # 重试机制
                success = False
                for retry in range(max_retries):
                    try:
                        print(f"  🔄 尝试 {retry + 1}/{max_retries}")
                        
                        # 预测单个文件
                        result = self.predict_single_file_cpu(
                            csv_path=csv_path,
                            bands=bands,
                            start_timestamp=start_timestamp,
                            end_timestamp=end_timestamp,
                            use_len=use_len,
                            prediction_steps=prediction_steps,
                            output_dir=output_subdir,
                            timestamp_col=timestamp_col
                        )
                        
                        if result is not None:
                            success_count += 1
                            results_summary[file_name] = {
                                'status': 'success',
                                'output_dir': output_subdir,
                                'bands': list(result.keys()),
                                'retries': retry + 1
                            }
                            success = True
                            break
                        else:
                            print(f"    ⚠ 预测返回空结果")
                            
                    except Exception as e:
                        print(f"    ❌ 尝试 {retry + 1} 失败: {str(e)}")
                        if retry < max_retries - 1:
                            print(f"    ⏳ 等待 3 秒后重试...")
                            time.sleep(3)
                            force_memory_cleanup()
                        else:
                            print(f"    💀 所有重试都失败了")
                
                if not success:
                    failed_count += 1
                    results_summary[file_name] = {
                        'status': 'failed',
                        'reason': f'所有 {max_retries} 次重试都失败',
                        'output_dir': output_subdir
                    }
            
            # 批次处理完成后清理内存
            print(f"\n🧹 批次 {batch_start//batch_size + 1} 完成，清理内存...")
            force_memory_cleanup()
            
            # 显示当前进度
            current_success_rate = success_count / (success_count + failed_count) * 100 if (success_count + failed_count) > 0 else 0
            print(f"📊 当前进度: {success_count + failed_count}/{total_files} ({current_success_rate:.1f}% 成功率)")
        
        # 显示最终结果
        print("\n" + "=" * 60)
        print("✅ CPU专用批量预测完成!")
        print("=" * 60)
        print(f"📈 成功: {success_count}/{total_files}")
        print(f"📊 成功率: {success_count/total_files*100:.1f}%")
        print(f"📁 结果保存在: {base_output_dir}")
        
        # 保存总结报告
        import pandas as pd
        summary_data = []
        for file_name, info in results_summary.items():
            summary_data.append({
                'file_name': file_name,
                'status': info['status'],
                'output_dir': info.get('output_dir', ''),
                'bands': ', '.join(info.get('bands', [])),
                'reason': info.get('reason', ''),
                'retries': info.get('retries', 0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(base_output_dir, 'cpu_only_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"📋 总结报告已保存到: {summary_file}")
        
        return results_summary

def main():
    """主函数"""
    print("🔥 CPU专用时间序列批量预测工具")
    print("=" * 60)
    print("💡 此工具专门用于GPU内存不足的情况")
    print("💻 强制使用CPU进行预测，避免GPU内存问题")
    print("=" * 60)
    
    # 获取用户输入
    print("\n请输入预测参数:")
    
    # 输入文件夹路径
    folder_path = input("📁 输入文件夹路径: ").strip()
    if not folder_path:
        print("❌ 请输入有效的文件夹路径")
        return
    
    # 预测波段
    bands_input = input("🎯 预测波段 (用空格分隔，如: albedo_01 tbb_07): ").strip()
    if not bands_input:
        print("❌ 请输入预测波段")
        return
    bands = bands_input.split()
    
    # 时间范围
    start_time = input("⏰ 开始时间 (格式: YYYY-MM-DD HH:MM:SS): ").strip()
    if not start_time:
        print("❌ 请输入开始时间")
        return
    
    end_time = input("⏰ 结束时间 (格式: YYYY-MM-DD HH:MM:SS): ").strip()
    if not end_time:
        print("❌ 请输入结束时间")
        return
    
    # 可选参数
    use_len_input = input("📊 使用长度 (默认144，按回车使用默认值): ").strip()
    use_len = int(use_len_input) if use_len_input else 144
    
    prediction_steps_input = input("🔮 预测步数 (默认1，按回车使用默认值): ").strip()
    prediction_steps = int(prediction_steps_input) if prediction_steps_input else 1
    
    batch_size_input = input("📦 批处理大小 (默认2，按回车使用默认值): ").strip()
    batch_size = int(batch_size_input) if batch_size_input else 2
    
    max_retries_input = input("🔄 最大重试次数 (默认2，按回车使用默认值): ").strip()
    max_retries = int(max_retries_input) if max_retries_input else 2
    
    # 确认参数
    print("\n📋 预测参数确认:")
    print(f"  输入文件夹: {folder_path}")
    print(f"  预测波段: {', '.join(bands)}")
    print(f"  时间范围: {start_time} 到 {end_time}")
    print(f"  使用长度: {use_len}")
    print(f"  预测步数: {prediction_steps}")
    print(f"  计算设备: CPU (强制)")
    print(f"  批处理大小: {batch_size}")
    print(f"  最大重试次数: {max_retries}")
    
    confirm = input("\n确认开始预测? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ 用户取消预测")
        return
    
    # 创建CPU专用预测器
    predictor = CPUOnlyBatchPredictor()
    
    # 运行预测
    results = predictor.run_cpu_batch_prediction(
        folder_path=folder_path,
        bands=bands,
        start_timestamp=start_time,
        end_timestamp=end_time,
        use_len=use_len,
        prediction_steps=prediction_steps,
        batch_size=batch_size,
        max_retries=max_retries
    )
    
    if results:
        print("\n🎉 预测完成! 请查看输出目录中的结果文件。")

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1:
        # 命令行模式
        if len(sys.argv) < 5:
            print("用法: python batch_predict_cpu_only.py <文件夹路径> <波段1 波段2...> <开始时间> <结束时间> [使用长度] [预测步数] [批处理大小] [重试次数]")
            print("示例: python batch_predict_cpu_only.py './data' 'albedo_01 tbb_07' '2022-10-15 01:00:00' '2022-10-16 01:00:00' 144 1 2 2")
            sys.exit(1)
        
        folder_path = sys.argv[1]
        bands = sys.argv[2].split()
        start_time = sys.argv[3]
        end_time = sys.argv[4]
        use_len = int(sys.argv[5]) if len(sys.argv) > 5 else 144
        prediction_steps = int(sys.argv[6]) if len(sys.argv) > 6 else 1
        batch_size = int(sys.argv[7]) if len(sys.argv) > 7 else 2
        max_retries = int(sys.argv[8]) if len(sys.argv) > 8 else 2
        
        # 创建CPU专用预测器
        predictor = CPUOnlyBatchPredictor()
        
        # 运行预测
        results = predictor.run_cpu_batch_prediction(
            folder_path=folder_path,
            bands=bands,
            start_timestamp=start_time,
            end_timestamp=end_time,
            use_len=use_len,
            prediction_steps=prediction_steps,
            batch_size=batch_size,
            max_retries=max_retries
        )
    else:
        # 交互式模式
        main() 