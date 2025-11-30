import numpy as np
from typing import Union, Optional
from scipy import stats
from scipy import ndimage

class SpatialAnomalyDetector:
    """空间异常检测器
    
    支持多种阈值方法：
    1. Otsu阈值
    2. 3σ (三西格玛)
    3. 局部异常因子 (Local Outlier Factor)
    """
    
    def __init__(self, method: str = 'otsu', **kwargs):
        """
        初始化空间异常检测器
        
        Args:
            method: 异常检测方法 ('otsu', '3sigma', 'lof')
            **kwargs: 方法特定参数
                - otsu: threshold_factor(默认1.0) - 阈值调整因子，大于1使阈值更高更严格，小于1使阈值更低更宽松
                - 3sigma: sigma_threshold(默认3.0) - 使用多少倍标准差作为异常边界
                - lof: k_neighbors(默认10) - 近邻数量
        """
        self.method = method.lower()
        self.kwargs = kwargs
        
        # 根据方法类型初始化相应的参数
        if self.method == 'otsu':
            self.threshold_factor = kwargs.get('threshold_factor', 1.0)
        elif self.method == '3sigma':
            self.sigma_threshold = kwargs.get('sigma_threshold', 3.0)
        elif self.method == 'lof':
            self.k_neighbors = kwargs.get('k_neighbors', 10)
            
        if self.method not in ['otsu', '3sigma', 'lof']:
            raise ValueError(f"Unsupported method: {method}. Choose from 'otsu', '3sigma', 'lof'")
    
    def detect(self, data: np.ndarray) -> np.ndarray:
        """
        对空间数据进行异常检测
        
        Args:
            data: 二维空间数据，形状为 (len, len)
            
        Returns:
            异常标签数组，1表示异常，0表示正常
        """
        if data.ndim != 2:
            raise ValueError("Input data must be 2-dimensional")
            
        if data.shape[0] != data.shape[1]:
            raise ValueError("Input data must be square (len, len)")
            
        if self.method == 'otsu':
            return self._detect_with_otsu(data)
        elif self.method == '3sigma':
            return self._detect_with_3sigma(data)
        elif self.method == 'lof':
            return self._detect_with_lof(data)
            
    def _detect_with_otsu(self, data: np.ndarray) -> np.ndarray:
        """使用Otsu方法进行异常检测"""
        # 将二维数据展平为一维进行Otsu阈值计算
        flat_data = data.flatten()
        
        # 计算Otsu阈值
        threshold = self._otsu_threshold(flat_data)
        
        # 根据阈值调整因子调整阈值
        adjusted_threshold = threshold * self.threshold_factor
        
        # 标记高于调整后阈值的点为异常
        labels = (data > adjusted_threshold).astype(int)
        
        return labels
    
    def _otsu_threshold(self, data: np.ndarray) -> float:
        """计算Otsu阈值"""
        # 创建直方图
        hist, bin_edges = np.histogram(data, bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
        
        # 类别概率和均值
        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]
        
        # 避免除零错误
        mean1 = np.cumsum(hist * bin_centers) / weight1
        mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]
        
        # 修正NaN值
        mean1[np.isnan(mean1)] = 0
        mean2[np.isnan(mean2)] = 0
        
        # 计算类间方差
        variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
        
        # 找到最大方差对应的阈值
        idx = np.argmax(variance12)
        threshold = bin_centers[:-1][idx]
        
        return threshold
    
    def _detect_with_3sigma(self, data: np.ndarray) -> np.ndarray:
        """使用3σ方法进行异常检测"""
        mean = np.mean(data)
        std = np.std(data)
        
        # 根据指定的sigma阈值计算异常边界
        lower_bound = mean - self.sigma_threshold * std
        upper_bound = mean + self.sigma_threshold * std
        
        # 标记超出范围的点为异常
        labels = ((data < lower_bound) | (data > upper_bound)).astype(int)
        
        return labels
    
    def _detect_with_lof(self, data: np.ndarray) -> np.ndarray:
        """使用局部异常因子方法进行异常检测"""
        try:
            from sklearn.neighbors import LocalOutlierFactor
        except ImportError:
            raise ImportError("scikit-learn is required for LOF method. Install it with: pip install scikit-learn")
            
        # 将二维数据展平为特征向量
        rows, cols = data.shape
        features = []
        positions = []
        
        # 为每个像素点创建特征向量（包括其邻域信息）
        for i in range(rows):
            for j in range(cols):
                # 获取邻域（3x3窗口）
                row_start = max(0, i - 1)
                row_end = min(rows, i + 2)
                col_start = max(0, j - 1)
                col_end = min(cols, j + 2)
                
                # 提取邻域值
                neighborhood = data[row_start:row_end, col_start:col_end].flatten()
                
                # 添加特征：当前值、邻域均值、邻域标准差
                current_value = data[i, j]
                neighborhood_mean = np.mean(neighborhood)
                neighborhood_std = np.std(neighborhood)
                
                features.append([current_value, neighborhood_mean, neighborhood_std])
                positions.append((i, j))
                
        features = np.array(features)
        
        # 应用LOF算法
        lof = LocalOutlierFactor(n_neighbors=self.k_neighbors)
        lof_labels = lof.fit_predict(features)
        
        # 将结果重构为二维标签数组
        labels = np.zeros_like(data, dtype=int)
        for idx, (i, j) in enumerate(positions):
            # LOF返回-1表示异常，1表示正常
            labels[i, j] = 1 if lof_labels[idx] == -1 else 0
            
        return labels

# 示例用法
if __name__ == "__main__":
    # 生成测试数据 (30x30的正方形)
    np.random.seed(42)
    size = 30
    normal_data = np.random.normal(0, 1, (size, size))
    
    # 插入一些异常区域
    test_data = normal_data.copy()
    test_data[10:15, 10:15] = 5  # 正方形异常区域
    test_data[20, 20] = 10       # 单点异常
    
    print("测试不同的空间异常检测方法:")
    
    # 测试Otsu方法
    detector_otsu = SpatialAnomalyDetector(method='otsu')
    labels_otsu = detector_otsu.detect(test_data)
    print(f"Otsu方法检测到的异常点数量: {np.sum(labels_otsu)}")
    
    # 测试3σ方法
    detector_3sigma = SpatialAnomalyDetector(method='3sigma')
    labels_3sigma = detector_3sigma.detect(test_data)
    print(f"3σ方法检测到的异常点数量: {np.sum(labels_3sigma)}")
    
    # 如果安装了scikit-learn，测试LOF方法
    try:
        detector_lof = SpatialAnomalyDetector(method='lof', k_neighbors=10)
        labels_lof = detector_lof.detect(test_data)
        print(f"LOF方法检测到的异常点数量: {np.sum(labels_lof)}")
    except ImportError as e:
        print(f"无法测试LOF方法: {e}")