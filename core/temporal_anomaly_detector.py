import numpy as np
from typing import Union, Optional
from scipy import stats

class TemporalAnomalyDetector:
    """时序异常检测器
    
    支持多种阈值方法：
    1. SPOT (Streaming POT)
    2. 3σ (三西格玛)
    3. 百分位数
    """
    
    def __init__(self, method: str = '3sigma', **kwargs):
        """
        初始化时序异常检测器
        
        Args:
            method: 异常检测方法 ('spot', '3sigma', 'percentile', 'fixed_window_once')
            **kwargs: 方法特定参数
                - spot: q(默认0.95), window_size(默认50), score_threshold(默认0.5)
                - 3sigma: window_size(默认50)
                - percentile: percentile(默认95), window_size(默认50)
                - fixed_window_once: initial_window_size(默认100), threshold(默认3.0), threshold_method(默认'sigma')
        """
        self.method = method.lower()
        self.kwargs = kwargs
        
        # 初始化方法特定参数
        if self.method == 'spot':
            self.q = kwargs.get('q', 0.95)  # 降低默认值以提高检测灵敏度
            self.window_size = kwargs.get('window_size', 50)  # 滑动窗口大小
            self.score_threshold = kwargs.get('score_threshold', 0.5)  # 异常分数阈值
            self._init_spot_detector()
        elif self.method == '3sigma':
            self.window_size = kwargs.get('window_size', 50)  # 滑动窗口大小
        elif self.method == 'percentile':
            self.percentile = kwargs.get('percentile', 95)  # 百分位数阈值
            self.window_size = kwargs.get('window_size', 50)  # 滑动窗口大小
        elif self.method == 'fixed_window_once':
            self.initial_window_size = kwargs.get('initial_window_size', 100)  # 初始窗口大小
            self.threshold = kwargs.get('threshold', 3.0)  # 固定阈值
            self.threshold_method = kwargs.get('threshold_method', 'sigma')  # 阈值计算方法: 'sigma', 'percentile', 'fixed'
            # 百分位数方法的额外参数
            self.percentile_value = kwargs.get('percentile_value', 95)  # 百分位数值
        else:
            raise ValueError(f"Unsupported method: {method}. Choose from 'spot', '3sigma', 'percentile', 'fixed_window_once'")
            
        # 存储历史数据，用于滑动窗口计算
        self.history_data = []
        self.spot_detector = None  # SPOT检测器实例
    
    def _init_spot_detector(self):
        """初始化SPOT检测器"""
        try:
            from streamad.model import SpotDetector
            self.spot_detector = SpotDetector(
                window_len=self.window_size  # 设置SPOT的窗口长度为我们的window_size
            )
            # 设置q参数
            self.spot_detector.q = self.q
        except ImportError:
            raise ImportError("streamad library is required for SPOT method. Install it with: pip install streamad")
    
    def detect(self, data: np.ndarray) -> np.ndarray:
        """
        对时序数据进行异常检测
        
        Args:
            data: 一维时序数据，形状为 (len,)
            
        Returns:
            异常标签数组，1表示异常，0表示正常
        """
        if data.ndim != 1:
            raise ValueError("Input data must be 1-dimensional")
            
        # 记录数据基本信息
        if self.method == 'fixed_window_once':
            print(f"开始异常检测: 数据长度={len(data)}, 方法={self.method}, 初始窗口大小={self.initial_window_size}")
        else:
            print(f"开始异常检测: 数据长度={len(data)}, 方法={self.method}, 窗口大小={self.window_size}")
        
        # 重置历史数据
        self.history_data = []
        
        if self.method == 'spot':
            return self._detect_with_spot(data)
        elif self.method == '3sigma':
            return self._detect_with_3sigma(data)
        elif self.method == 'percentile':
            return self._detect_with_percentile(data)
        elif self.method == 'fixed_window_once':
            return self._detect_with_fixed_window_once(data)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
    
    def _detect_with_spot(self, data: np.ndarray) -> np.ndarray:
        """使用SPOT方法进行异常检测 - 基于指定窗口大小的历史数据"""
        labels = np.zeros(len(data), dtype=int)
        
        try:
            # 确保数据不为空
            if len(data) == 0:
                print("警告: 输入数据为空")
                return labels
                
            # 确保SPOT检测器已初始化
            if self.spot_detector is None:
                self._init_spot_detector()
                
            # 遍历数据点，使用滑动窗口进行异常检测
            for i, value in enumerate(data):
                # 维护滑动窗口
                self.history_data.append(value)
                if len(self.history_data) > self.window_size:
                    self.history_data.pop(0)
                    
                # 当窗口数据足够时进行检测
                if len(self.history_data) >= self.window_size // 2:  # 至少需要一半窗口大小的数据
                    try:
                        # 使用SPOT检测器检测当前点是否异常
                        # 注意：这里使用当前窗口的所有数据进行异常检测
                        # 这样确保检测是基于历史窗口的
                        score = 0
                        for hist_value in self.history_data:
                            current_score = self.spot_detector.fit_score(np.array([hist_value]))
                            if current_score is not None and isinstance(current_score, (int, float)):
                                score = max(score, current_score)  # 使用窗口内的最高分数
                        
                        # 根据配置的阈值判断当前点是否为异常
                        if score > self.score_threshold:
                            labels[i] = 1
                            
                    except Exception as e:
                        # 记录错误但继续处理
                        print(f"处理数据点 {i} 时出错: {e}")
                        pass
                        
                # 调试信息
                if i % 50 == 0:
                    print(f"已处理数据点 {i}/{len(data)}, 当前窗口大小: {len(self.history_data)}")
        except Exception as e:
            # 发生严重错误时，返回全0标签
            print(f"SPOT检测过程中发生严重错误: {e}")
        
        print(f"SPOT方法总共检测到 {np.sum(labels)} 个异常点")
        return labels
    
    def _detect_with_3sigma(self, data: np.ndarray) -> np.ndarray:
        """使用3σ方法进行异常检测 - 基于指定窗口大小的历史数据"""
        labels = np.zeros(len(data), dtype=int)
        
        for i, value in enumerate(data):
            # 维护滑动窗口
            self.history_data.append(value)
            if len(self.history_data) > self.window_size:
                self.history_data.pop(0)
                
            # 当历史数据足够时进行检测
            if len(self.history_data) >= 5:  # 至少需要5个点来估计分布
                # 使用当前滑动窗口内的历史数据计算统计量
                mean = np.mean(self.history_data)
                std = np.std(self.history_data)
                
                # 3σ规则：超出μ±3σ范围为异常
                if abs(value - mean) > 3 * std:
                    labels[i] = 1
                    
            # 调试信息
            if i % 50 == 0:
                print(f"已处理数据点 {i}/{len(data)}, 当前窗口大小: {len(self.history_data)}")
        
        print(f"3σ方法总共检测到 {np.sum(labels)} 个异常点")
        return labels
    
    def _detect_with_percentile(self, data: np.ndarray) -> np.ndarray:
        """使用百分位数方法进行异常检测 - 基于指定窗口大小的历史数据"""
        labels = np.zeros(len(data), dtype=int)
        
        for i, value in enumerate(data):
            # 维护滑动窗口
            self.history_data.append(value)
            if len(self.history_data) > self.window_size:
                self.history_data.pop(0)
                
            # 当历史数据足够时进行检测
            if len(self.history_data) >= 10:  # 至少需要10个点来估计分布
                # 使用当前滑动窗口内的历史数据计算百分位数阈值
                lower_percentile = (100 - self.percentile) / 2
                upper_percentile = 100 - lower_percentile
                
                lower_threshold = np.percentile(self.history_data, lower_percentile)
                upper_threshold = np.percentile(self.history_data, upper_percentile)
                
                # 超出阈值范围为异常
                if value < lower_threshold or value > upper_threshold:
                    labels[i] = 1
                    
            # 调试信息
            if i % 50 == 0:
                print(f"已处理数据点 {i}/{len(data)}, 当前窗口大小: {len(self.history_data)}")
        
        print(f"百分位数方法总共检测到 {np.sum(labels)} 个异常点")
        return labels
    
    def _detect_with_fixed_window_once(self, data: np.ndarray) -> np.ndarray:
        """
        使用固定窗口一次方法进行异常检测
        只使用开始时刻前initial_window_size个时相作为窗口，之后不再滑动，使用固定阈值
        
        Args:
            data: 一维时序数据，形状为 (len,)
            
        Returns:
            异常标签数组，1表示异常，0表示正常
        """
        labels = np.zeros(len(data), dtype=int)
        
        # 确保数据长度足够
        if len(data) < self.initial_window_size:
            print(f"警告: 数据长度({len(data)})小于初始窗口大小({self.initial_window_size})")
            # 使用所有可用数据计算统计量
            if len(data) >= 5:  # 至少需要5个点来估计分布
                if self.threshold_method == 'sigma':
                    mean = np.mean(data)
                    std = np.std(data)
                    # 对所有数据点应用阈值
                    labels[abs(data - mean) > self.threshold * std] = 1
                elif self.threshold_method == 'percentile':
                    # 计算上界和下界
                    lower_bound = np.percentile(data, (100 - self.percentile_value) / 2)
                    upper_bound = np.percentile(data, 100 - (100 - self.percentile_value) / 2)
                    labels[(data < lower_bound) | (data > upper_bound)] = 1
                else:  # fixed方法
                    mean = np.mean(data)
                    std = np.std(data)
                    labels[abs(data - mean) > self.threshold * std] = 1
            return labels
        
        # 使用前initial_window_size个点计算统计量
        initial_window = data[:self.initial_window_size]
        mean = np.mean(initial_window)
        std = np.std(initial_window)
        
        # 根据阈值计算方法应用不同的检测逻辑
        if self.threshold_method == 'sigma':
            print(f"使用前{self.initial_window_size}个数据点计算统计量: 均值={mean:.4f}, 标准差={std:.4f}")
            print(f"使用三西格玛方法，阈值系数={self.threshold}")
            # 对所有数据点应用阈值
            labels[abs(data - mean) > self.threshold * std] = 1
        elif self.threshold_method == 'percentile':
            # 使用百分位数方法
            lower_bound = np.percentile(initial_window, (100 - self.percentile_value) / 2)
            upper_bound = np.percentile(initial_window, 100 - (100 - self.percentile_value) / 2)
            print(f"使用前{self.initial_window_size}个数据点计算统计量: 均值={mean:.4f}, 标准差={std:.4f}")
            print(f"使用百分位数方法，百分位={self.percentile_value}, 下界={lower_bound:.4f}, 上界={upper_bound:.4f}")
            labels[(data < lower_bound) | (data > upper_bound)] = 1
        else:  # fixed方法
            print(f"使用前{self.initial_window_size}个数据点计算统计量: 均值={mean:.4f}, 标准差={std:.4f}")
            print(f"使用固定阈值方法，阈值系数={self.threshold}")
            labels[abs(data - mean) > self.threshold * std] = 1
        
        # 确保初始窗口内的数据点不被标记为异常
        # 这是为了保证基线的稳定性
        labels[:self.initial_window_size] = 0
        
        print(f"固定窗口一次方法总共检测到 {np.sum(labels)} 个异常点")
        return labels
    
    def get_best_params_suggestion(self, data_pattern: str = 'general') -> dict:
        """获取推荐的最佳参数配置"""
        params_suggestions = {
            'spot': {
                'general': {'q': 0.95, 'window_size': 50, 'score_threshold': 0.5},
                'high_variability': {'q': 0.90, 'window_size': 30, 'score_threshold': 0.4},
                'low_variability': {'q': 0.98, 'window_size': 100, 'score_threshold': 0.7},
                'seasonal': {'q': 0.95, 'window_size': 100, 'score_threshold': 0.6}
            },
            '3sigma': {
                'general': {'window_size': 50},
                'high_variability': {'window_size': 30},
                'low_variability': {'window_size': 100},
                'seasonal': {'window_size': 100}
            },
            'percentile': {
                'general': {'percentile': 95, 'window_size': 50},
                'high_variability': {'percentile': 90, 'window_size': 30},
                'low_variability': {'percentile': 98, 'window_size': 100},
                'seasonal': {'percentile': 95, 'window_size': 100}
            },
            'fixed_window_once': {
                'general': {'initial_window_size': 100, 'threshold': 3.0},
                'high_variability': {'initial_window_size': 100, 'threshold': 2.5},
                'low_variability': {'initial_window_size': 100, 'threshold': 3.5},
                'seasonal': {'initial_window_size': 100, 'threshold': 3.0}
            }
        }
        
        # 返回对应方法和数据模式的推荐参数
        if data_pattern not in params_suggestions[self.method]:
            print(f"警告: 未知的数据模式 '{data_pattern}'，使用默认参数")
            return params_suggestions[self.method]['general']
            
        return params_suggestions[self.method][data_pattern]

# 示例用法
if __name__ == "__main__":
    # 生成测试数据
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, 200)
    # 插入一些异常值
    anomaly_indices = [50, 100, 150]
    test_data = normal_data.copy()
    test_data[anomaly_indices] = 10  # 明显的异常值
    
    print("测试不同的时序异常检测方法:")
    
    # 测试3σ方法
    detector_3sigma = TemporalAnomalyDetector(method='3sigma', window_size=50)
    labels_3sigma = detector_3sigma.detect(test_data)
    print(f"3σ方法检测到的异常点索引: {np.where(labels_3sigma == 1)[0]}")
    
    # 测试百分位数方法
    detector_percentile = TemporalAnomalyDetector(method='percentile', percentile=95, window_size=50)
    labels_percentile = detector_percentile.detect(test_data)
    print(f"百分位数方法检测到的异常点索引: {np.where(labels_percentile == 1)[0]}")
    
    # 如果安装了streamad，测试SPOT方法
    try:
        detector_spot = TemporalAnomalyDetector(method='spot', q=0.95, window_size=50)
        labels_spot = detector_spot.detect(test_data)
        print(f"SPOT方法检测到的异常点索引: {np.where(labels_spot == 1)[0]}")
    except ImportError as e:
        print(f"无法测试SPOT方法: {e}")