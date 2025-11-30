#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速批量预测脚本
提供简化的批量预测接口，适合快速使用
"""

import os
import sys
import torch
from batch_predict import BatchPredictor

def quick_batch_predict(folder_path, bands, start_time, end_time, 
                       use_len=144, prediction_steps=3, device='auto'):
    """
    快速批量预测函数
    
    Args:
        folder_path: 输入文件夹路径
        bands: 预测波段列表
        start_time: 开始时间 (格式: 'YYYY-MM-DD HH:MM:SS')
        end_time: 结束时间 (格式: 'YYYY-MM-DD HH:MM:SS')
        use_len: 用于预测的长度，默认144（24小时）
        prediction_steps: 预测步数，默认3
        device: 计算设备，'auto'自动选择，'cpu'或'cuda'
    
    Returns:
        dict: 预测结果统计
    """
    
    print("🚀 快速批量预测启动")
    print("=" * 50)
    
    # 检查输入文件夹
    if not os.path.exists(folder_path):
        print(f"❌ 错误: 输入文件夹不存在: {folder_path}")
        return None
    
    # 自动选择设备
    if device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"📁 输入文件夹: {folder_path}")
    print(f"🎯 预测波段: {', '.join(bands)}")
    print(f"⏰ 时间范围: {start_time} 到 {end_time}")
    print(f"📊 使用长度: {use_len} ({(use_len*10)/60:.1f}小时)")
    print(f"🔮 预测步数: {prediction_steps}")
    print(f"💻 使用设备: {device}")
    
    # 生成输出目录
    folder_name = os.path.basename(folder_path)
    output_dir = f'./{folder_name}_batch_predictions'
    
    try:
        # 创建批量预测器
        print("\n🔧 初始化预测器...")
        batch_predictor = BatchPredictor(device=device)
        
        # 运行批量预测
        print("🚀 开始批量预测...")
        results = batch_predictor.run_batch_prediction(
            folder_path=folder_path,
            bands=bands,
            start_timestamp=start_time,
            end_timestamp=end_time,
            use_len=use_len,
            prediction_steps=prediction_steps,
            base_output_dir=output_dir
        )
        
        # 显示结果
        success_count = sum(1 for info in results.values() if info['status'] == 'success')
        total_count = len(results)
        
        print("\n" + "=" * 50)
        print("✅ 批量预测完成!")
        print("=" * 50)
        print(f"📈 成功: {success_count}/{total_count}")
        print(f"📊 成功率: {success_count/total_count*100:.1f}%")
        print(f"📁 结果保存在: {output_dir}")
        
        return results
        
    except Exception as e:
        print(f"❌ 预测过程中出现错误: {str(e)}")
        return None

def main():
    """主函数 - 交互式使用"""
    
    print("🔥 时间序列批量预测工具")
    print("=" * 50)
    
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
    
    prediction_steps_input = input("🔮 预测步数 (默认3，按回车使用默认值): ").strip()
    prediction_steps = int(prediction_steps_input) if prediction_steps_input else 3
    
    device_input = input("💻 计算设备 (auto/cpu/cuda，默认auto): ").strip()
    device = device_input if device_input else 'auto'
    
    # 确认参数
    print("\n📋 预测参数确认:")
    print(f"  输入文件夹: {folder_path}")
    print(f"  预测波段: {', '.join(bands)}")
    print(f"  时间范围: {start_time} 到 {end_time}")
    print(f"  使用长度: {use_len}")
    print(f"  预测步数: {prediction_steps}")
    print(f"  计算设备: {device}")
    
    confirm = input("\n确认开始预测? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ 用户取消预测")
        return
    
    # 运行预测
    results = quick_batch_predict(
        folder_path=folder_path,
        bands=bands,
        start_time=start_time,
        end_time=end_time,
        use_len=use_len,
        prediction_steps=prediction_steps,
        device=device
    )
    
    if results:
        print("\n🎉 预测完成! 请查看输出目录中的结果文件。")

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1:
        # 命令行模式
        if len(sys.argv) < 5:
            print("用法: python quick_batch_predict.py <文件夹路径> <波段1 波段2...> <开始时间> <结束时间> [使用长度] [预测步数] [设备]")
            print("示例: python quick_batch_predict.py './data' 'albedo_01 tbb_07' '2022-10-15 01:00:00' '2022-10-16 01:00:00' 144 3 cuda")
            sys.exit(1)
        
        folder_path = sys.argv[1]
        bands = sys.argv[2].split()
        start_time = sys.argv[3]
        end_time = sys.argv[4]
        use_len = int(sys.argv[5]) if len(sys.argv) > 5 else 144
        prediction_steps = int(sys.argv[6]) if len(sys.argv) > 6 else 3
        device = sys.argv[7] if len(sys.argv) > 7 else 'auto'
        
        results = quick_batch_predict(
            folder_path=folder_path,
            bands=bands,
            start_time=start_time,
            end_time=end_time,
            use_len=use_len,
            prediction_steps=prediction_steps,
            device=device
        )
    else:
        # 交互式模式
        main() 