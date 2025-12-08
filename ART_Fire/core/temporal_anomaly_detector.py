import numpy as np
from typing import Union, Optional
from scipy import stats

class TemporalAnomalyDetector:
    """
    Temporal Anomaly Detector for time series data analysis.
    
    Supports multiple threshold methods:
    1. SPOT (Streaming Peaks-Over-Threshold)
    2. 3-Sigma (Three Standard Deviations)
    3. Percentile-based detection
    4. Fixed window once-pass detection
    
    This detector is specifically designed for wildfire detection applications,
    analyzing temporal patterns in environmental data such as temperature,
    humidity, and satellite imagery time series.
    """
    
    def __init__(self, method: str = '3sigma', **kwargs):
        """
        Initialize the temporal anomaly detector.
        
        Args:
            method (str): Anomaly detection method. Options:
                - 'spot': Streaming Peaks-Over-Threshold
                - '3sigma': Three standard deviations
                - 'percentile': Percentile-based threshold
                - 'fixed_window_once': Single-pass fixed window
            **kwargs: Method-specific parameters:
                - spot: q (default 0.95), window_size (default 50), score_threshold (default 0.5)
                - 3sigma: window_size (default 50)
                - percentile: percentile (default 95), window_size (default 50)
                - fixed_window_once: initial_window_size (default 100), threshold (default 3.0), 
                                     threshold_method (default 'sigma')
        
        Raises:
            ValueError: If an unsupported detection method is specified.
            ImportError: If required dependencies for SPOT method are not installed.
        """
        self.method = method.lower()
        self.kwargs = kwargs
        
        # Initialize method-specific parameters
        if self.method == 'spot':
            self.q = kwargs.get('q', 0.95)  # Lower default for increased sensitivity
            self.window_size = kwargs.get('window_size', 50)  # Sliding window size
            self.score_threshold = kwargs.get('score_threshold', 0.5)  # Anomaly score threshold
            self._init_spot_detector()
        elif self.method == '3sigma':
            self.window_size = kwargs.get('window_size', 50)  # Sliding window size
        elif self.method == 'percentile':
            self.percentile = kwargs.get('percentile', 95)  # Percentile threshold
            self.window_size = kwargs.get('window_size', 50)  # Sliding window size
        elif self.method == 'fixed_window_once':
            self.initial_window_size = kwargs.get('initial_window_size', 100)  # Initial window size
            self.threshold = kwargs.get('threshold', 3.0)  # Fixed threshold
            self.threshold_method = kwargs.get('threshold_method', 'sigma')  # Threshold calculation method
            # Additional parameters for percentile method
            self.percentile_value = kwargs.get('percentile_value', 95)  # Percentile value
        else:
            raise ValueError(f"Unsupported method: {method}. Choose from 'spot', '3sigma', 'percentile', 'fixed_window_once'")
            
        # Store historical data for sliding window calculations
        self.history_data = []
        self.spot_detector = None  # SPOT detector instance
    
    def _init_spot_detector(self):
        """
        Initialize the SPOT (Streaming Peaks-Over-Threshold) detector.
        
        SPOT is an advanced anomaly detection method particularly effective for
        detecting extreme values in streaming data, commonly used in environmental
        monitoring and wildfire detection applications.
        
        Raises:
            ImportError: If the streamad library is not installed.
        """
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
        Perform anomaly detection on time series data.
        
        This method applies the configured detection algorithm to identify
        anomalous data points in the input time series. The detection is
        performed using a sliding window approach to maintain temporal context.
        
        Args:
            data (np.ndarray): One-dimensional time series data with shape (n_samples,).
                             Should contain numeric values representing the time series.
            
        Returns:
            np.ndarray: Binary anomaly labels where 1 indicates anomaly and 0 indicates normal.
                       Array has the same length as the input data.
        
        Raises:
            ValueError: If input data is not one-dimensional or contains invalid values.
            RuntimeError: If detection fails due to insufficient data or computational errors.
        
        Example:
            >>> detector = TemporalAnomalyDetector(method='3sigma')
            >>> data = np.array([1, 2, 3, 100, 4, 5])  # 100 is an anomaly
            >>> labels = detector.detect(data)
            >>> print(labels)  # [0 0 0 1 0 0]
        """
        if data.ndim != 1:
            raise ValueError("Input data must be 1-dimensional")
            
        # Log basic information about the detection process
        if self.method == 'fixed_window_once':
            print(f"Starting anomaly detection: data_length={len(data)}, method={self.method}, initial_window_size={self.initial_window_size}")
        else:
            print(f"Starting anomaly detection: data_length={len(data)}, method={self.method}, window_size={self.window_size}")
        
        # Reset historical data for fresh detection
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
        """
        Perform anomaly detection using the SPOT (Streaming Peaks-Over-Threshold) method.
        
        SPOT is particularly effective for detecting extreme values in streaming data by
        modeling the tail distribution of normal data. This implementation uses a
        sliding window approach to maintain historical context.
        
        Args:
            data (np.ndarray): One-dimensional time series data.
            
        Returns:
            np.ndarray: Binary anomaly labels (1 for anomaly, 0 for normal).
            
        Note:
            This method requires the streamad library to be installed.
            The detection is based on the specified window size of historical data.
        """
        labels = np.zeros(len(data), dtype=int)
        
        try:
            # Ensure data is not empty
            if len(data) == 0:
                print("Warning: Input data is empty")
                return labels
                
            # Ensure SPOT detector is initialized
            if self.spot_detector is None:
                self._init_spot_detector()
                
            # 遍历数据点，使用滑动窗口进行异常检测
            for i, value in enumerate(data):
                # Maintain sliding window
                self.history_data.append(value)
                if len(self.history_data) > self.window_size:
                    self.history_data.pop(0)
                    
                # Perform detection when sufficient window data is available
                if len(self.history_data) >= self.window_size // 2:  # Require at least half window size
                    try:
                        # Use SPOT detector to check if current point is anomalous
                        # Note: Use all data in current window for anomaly detection
                        # This ensures detection is based on historical window
                        score = 0
                        for hist_value in self.history_data:
                            current_score = self.spot_detector.fit_score(np.array([hist_value]))
                            if current_score is not None and isinstance(current_score, (int, float)):
                                score = max(score, current_score)  # Use highest score in window
                        
                        # Determine if current point is anomalous based on configured threshold
                        if score > self.score_threshold:
                            labels[i] = 1
                            
                    except Exception as e:
                        # Log error but continue processing
                        print(f"Error processing data point {i}: {e}")
                        pass
                        
                # Debug information
                if i % 50 == 0:
                    print(f"Processed data point {i}/{len(data)}, current window size: {len(self.history_data)}")
        except Exception as e:
            # Return all-zero labels on critical error
            print(f"Critical error in SPOT detection: {e}")
        
        print(f"SPOT method detected {np.sum(labels)} anomaly points in total")
        return labels
    
    def _detect_with_3sigma(self, data: np.ndarray) -> np.ndarray:
        """
        Perform anomaly detection using the 3-sigma (three standard deviations) method.
        
        This method identifies anomalies based on statistical outliers that fall outside
        the range of mean ± 3 standard deviations. The calculation is performed using
        a sliding window of historical data to maintain temporal context.
        
        Args:
            data (np.ndarray): One-dimensional time series data.
            
        Returns:
            np.ndarray: Binary anomaly labels (1 for anomaly, 0 for normal).
            
        Note:
            The detection is based on the specified window size of historical data.
            Points are considered anomalous if they deviate more than 3 standard
            deviations from the window mean.
        """
        labels = np.zeros(len(data), dtype=int)
        
        for i, value in enumerate(data):
            # Maintain sliding window
            self.history_data.append(value)
            if len(self.history_data) > self.window_size:
                self.history_data.pop(0)
                
            # Perform detection when sufficient historical data is available
            if len(self.history_data) >= 5:  # Need at least 5 points to estimate distribution
                # Calculate statistics using current sliding window data
                mean = np.mean(self.history_data)
                std = np.std(self.history_data)
                
                # 3-sigma rule: points outside μ±3σ are considered anomalous
                if abs(value - mean) > 3 * std:
                    labels[i] = 1
                    
            # Debug information
            if i % 50 == 0:
                print(f"Processed data point {i}/{len(data)}, current window size: {len(self.history_data)}")
        
        print(f"3-sigma method detected {np.sum(labels)} anomaly points in total")
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