from matplotlib import pyplot as plt
import numpy as np
from streamad.util import StreamGenerator, CustomDS
from streamad.model import SpotDetector

class WildfireFeatureProcessor:
    @staticmethod
    def calculate_spectral_features(data):
        BT7 = data[3]
        BT13 = data[4]
        BT14 = data[5]
        return [BT7 - BT13, BT7 - BT14]

    @staticmethod
    def calculate_temporal_features(current_data, history_data):
        if len(history_data) < 2:
            return [0.0]
        current_rate = current_data[3] - history_data[-1][3]
        history_rate = history_data[-1][3] - history_data[-2][3]
        return [current_rate - history_rate]

    @staticmethod
    def calculate_spatial_features(current_data, neighbor_data):
        if not neighbor_data:
            return [0.0]
        neighbor_BT7 = [n[3] for n in neighbor_data]
        avg_neighbor_BT7 = np.mean(neighbor_BT7)
        return [current_data[3] - avg_neighbor_BT7]

def extract_all_features(data, history, neighbor_data):
    spectral = WildfireFeatureProcessor.calculate_spectral_features(data)
    temporal = WildfireFeatureProcessor.calculate_temporal_features(data, history)
    spatial = WildfireFeatureProcessor.calculate_spatial_features(data, neighbor_data)
    return spectral + temporal + spatial

if __name__ == "__main__":
    num_samples = 500
    data = [
        (
            np.random.normal(0.2, 0.05),
            np.random.normal(0.3, 0.05),
            np.random.normal(0.25, 0.05),
            np.random.normal(320, 10) if i % 50 == 0 else np.random.normal(290, 5),
            np.random.normal(290, 5) if i % 50 == 0 else np.random.normal(285, 5),
            np.random.normal(288, 5) if i % 50 == 0 else np.random.normal(283, 5)
        ) for i in range(num_samples)
    ]

    # 1. 构建所有特征流
    history_data = []
    feature_streams = [[] for _ in range(4)]  # 4个特征

    for i in range(num_samples):
        main_sample = data[i]
        neighbor_samples = [
            data[j % num_samples]
            for j in range(i+1, i+4)
            if j < num_samples
        ]
        features = extract_all_features(main_sample, history_data, neighbor_samples)
        for j, val in enumerate(features):
            feature_streams[j].append(val)
        history_data.append(main_sample)
        if len(history_data) > 50:
            history_data.pop(0)

    # 2. 用streamad的CustomDS和StreamGenerator处理每个特征流
    feature_names = ['spectral_delta_BT7_13', 'spectral_delta_BT7_14', 'temporal_delta_rate_BT7', 'spatial_delta_BT7']
    all_scores = {}

    for idx, feat_name in enumerate(feature_names):
        feat_arr = np.array(feature_streams[idx])
        ds = CustomDS(feat_arr, np.zeros_like(feat_arr))
        stream = StreamGenerator(ds.data)
        model = SpotDetector(window_len=20)  # 可以调整窗口
        scores = []
        for x in stream.iter_item():
            score = model.fit_score(x)
            scores.append(score)
        all_scores[feat_name] = scores

    # 3. 打印每个特征的异常分数示例
    for i in range(num_samples):
        print(f"样本[{i:03d}]", end=' ')
        for feat_name in feature_names:
            score = all_scores[feat_name][i]
            score = 0.0 if score is None else score  # 防止None报错
            print(f"{feat_name}:{score:.4f}", end=' ')
        print()
    # for idx, feat_name in enumerate(feature_names):
    #     scores = [0.0 if s is None else s for s in all_scores[feat_name]]
    #     x = np.arange(len(scores))
    #     plt.plot(x, scores, label=feat_name)
    #     plt.title(f"{feat_name} 异常分数曲线")
    #     plt.xlabel("样本序号")
    #     plt.ylabel("异常分数")
    #     plt.legend()
    #     plt.show()
    for idx, feat_name in enumerate(feature_names):
        # 获取原始特征和异常分数
        raw_values = feature_streams[idx]
        scores = [0.0 if s is None else s for s in all_scores[feat_name]]
        x = np.arange(len(scores))

        fig, ax1 = plt.subplots(figsize=(12, 5))
        
        # 左轴：原始特征
        color = 'tab:blue'
        ax1.set_xlabel('样本序号')
        ax1.set_ylabel('原始特征值', color=color)
        ax1.plot(x, raw_values, color=color, label=f'{feat_name} 原始值')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')
        
        # 右轴：异常分数
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('异常分数', color=color)
        ax2.plot(x, scores, color=color, label=f'{feat_name} 异常分数')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')
        
        plt.title(f"{feat_name} 原始特征+异常分数")
        plt.tight_layout()
        plt.show()