# Moe_Fire: Time-MoE for Fire Prediction

This repository contains a specialized implementation of Time-MoE (Time Series Mixture of Experts) for fire prediction and analysis applications.

## Overview

Time-MoE is a billion-scale time series foundation model with Mixture of Experts architecture, adapted here for fire prediction tasks. The model operates in an auto-regressive manner, enabling universal forecasting with arbitrary prediction horizons.

## Key Features

- **Fire-Specific Adaptations**: Specialized for fire-related time series data
- **Flexible Prediction**: Supports arbitrary context lengths up to 4096 time points
- **Multi-Scale Analysis**: Capable of handling various temporal resolutions
- **Multiple Prediction Modes**: CPU-only, quick, and memory-optimized batch prediction options
- **Production Ready**: Includes specialized scripts for different computational environments

## Installation

1. Install Python 3.10+, then install dependencies:

```bash
pip install -r requirements.txt
```

**Note: Requires `transformers==4.40.1`**

2. [Optional] Install flash-attn for faster inference:

```bash
pip install flash-attn==2.6.3
```

## Quick Start

### Basic Usage

```python
import torch
from transformers import AutoModelForCausalLM

# Initialize model
model = AutoModelForCausalLM.from_pretrained(
    'Maple728/TimeMoE-50M',
    device_map="cpu",  # or "cuda" for GPU
    trust_remote_code=True,
)

# Prepare your fire data (context_length: 12, batch_size: 2)
context_length = 12
fire_data = torch.randn(2, context_length)

# Normalize data
mean, std = fire_data.mean(dim=-1, keepdim=True), fire_data.std(dim=-1, keepdim=True)
normed_data = (fire_data - mean) / std

# Make predictions
prediction_length = 6
output = model.generate(normed_data, max_new_tokens=prediction_length)
predictions = output[:, -prediction_length:]

# Denormalize
fire_predictions = predictions * std + mean
```

### Training on Fire Data

For CPU training:
```bash
python main.py -d <your_fire_data_path>
```

For GPU training:
```bash
python torch_dist_run.py main.py -d <your_fire_data_path>
```

### Batch Prediction

For large-scale fire prediction tasks, use the specialized batch prediction scripts:

**CPU-only batch prediction (recommended for large datasets):**
```bash
python batch_predict_cpu_only.py -d <fire_data_path> -m <model_path> -o <output_path>
```

**Standard batch prediction:**
```bash
python batch_predict.py -d <fire_data_path> -m <model_path> -o <output_path>
```

**Quick batch prediction (optimized for speed):**
```bash
python quick_batch_predict.py -d <fire_data_path> -m <model_path> -o <output_path>
```

**Ultra memory-optimized batch prediction:**
```bash
python batch_predict_ultra_memory_optimized.py -d <fire_data_path> -m <model_path> -o <output_path>
```

**Advanced time series prediction:**
```bash
python predict_timeseries_v2.py -d <fire_data_path> -m <model_path> -o <output_path>
```

### Evaluation

```bash
python run_eval.py -d dataset/fire_data.csv -p 96
```

## Data Format

Your fire data should be in JSONL format:

```jsonl
{"sequence": [temperature_1, humidity_1, wind_speed_1, ...]}
{"sequence": [temperature_2, humidity_2, wind_speed_2, ...]}
```

## Model Architecture

The Time-MoE architecture features:
- Decoder-only transformer with Mixture of Experts
- Support for sequences up to 4096 tokens
- Auto-regressive generation for time series forecasting
- Configurable attention mechanisms

## Important Notes

- Maximum sequence length (context + prediction) should not exceed 4096
- For small datasets, use `--stride 1` parameter
- The model supports various normalization methods: none, zero, max

## License

This project is licensed under the Apache-2.0 License.

## Citation

If you use this work, please cite the original Time-MoE paper:

```bibtex
@misc{shi2024timemoe,
      title={Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts}, 
      author={Xiaoming Shi and Shiyu Wang and Yuqi Nie and Dianqi Li and Zhou Ye and Qingsong Wen and Ming Jin},
      year={2024},
      eprint={2409.16040},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2409.16040}, 
}
```