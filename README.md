# SmartStockRL: Reinforcement Learning for Inventory Management

A reinforcement learning approach to inventory management that outperforms traditional rule-based policies by optimizing the trade-off between service level and inventory costs.

## 🎯 Project Overview

This project implements and compares different inventory management strategies:
- **Reinforcement Learning Agent** (PPO/DQN) - Our intelligent approach
- **Naive Policy** - Simple reorder-to-target strategy  
- **sS Policy** - Traditional (s,S) inventory control

The RL agent learns to make optimal ordering decisions by balancing:
- Revenue from sales
- Holding costs for excess inventory
- Stockout costs for unmet demand
- Fixed and variable ordering costs

## 🏆 Key Results

Our PPO agent significantly outperforms traditional inventory policies:

| Policy | Average Return | Fill Rate | Avg Inventory | Stockouts |
|--------|---------------|-----------|---------------|-----------|
| **RL (PPO)** | **4,144.6** ± 223.8 | 99.14% | **9.0** ± 0.3 | 3.9 ± 2.9 |
| Naive(T=50) | 3,769.0 ± 236.2 | 100% | 42.4 ± 0.4 | 0.0 |
| sS(20,80) | 3,797.0 ± 235.2 | 100% | 44.4 ± 1.1 | 0.0 |

**Key Insights:**
- 🚀 **10% higher profits** than baseline policies
- 📉 **78% lower inventory levels** (9 vs 42-44 units)
- 🎯 **Smart risk management** - accepts minimal stockouts for massive cost savings
- 💰 **Superior capital efficiency** - frees up working capital while increasing profits

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/SmartStockRL.git
cd SmartStockRL

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Training an RL Agent
```bash
# Train PPO agent (default)
python train.py --algo ppo --steps 200000 --models-dir models --logs-dir logs

# Train DQN agent
python train.py --algo dqn --steps 200000 --models-dir models --logs-dir logs
```

**Training Parameters:**
- `--algo`: Choose between 'ppo' or 'dqn'
- `--steps`: Total training timesteps (default: 200,000)
- `--seed`: Random seed for reproducibility (default: 42)
- `--eval-every`: Evaluation frequency (default: 10,000)
- `--models-dir`: Directory to save models (default: 'models')
- `--logs-dir`: Directory for logs and TensorBoard (default: 'logs')

### 2. Evaluating Performance
```bash
# Evaluate trained model against baselines
python evaluate.py --model models/best_model.zip --episodes 50 --out runs/eval_metrics.csv

# Evaluate specific model
python evaluate.py --model models/SmartStockRL_PPO_20250906_212858.zip --episodes 100
```

### 3. Monitoring Training
```bash
# Launch TensorBoard to monitor training progress
tensorboard --logdir logs/tb
```

## 📊 Project Structure

```
SmartStockRL/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── requirements-dev.txt   # Development dependencies
├── configs.py            # Environment configuration parameters
├── inv_env.py           # Gymnasium environment for inventory management
├── train.py             # Training script for RL agents
├── evaluate.py          # Evaluation script comparing all policies
├── baselines.py         # Implementation of baseline policies
├── plot_results.py      # Visualization utilities
├── models/              # Saved trained models
├── logs/                # Training logs and TensorBoard data
└── runs/                # Evaluation results and metrics
```

## 🔧 Core Components

### Environment (`inv_env.py`)
- **State Space**: [current_inventory, demand_forecast, day_of_week]
- **Action Space**: Discrete ordering quantities (0 to max_order)
- **Reward Function**: Revenue - Holding Costs - Stockout Costs - Order Costs
- **Dynamics**: Poisson demand with optional seasonality, immediate order fulfillment

### Configuration (`configs.py`)
Easily adjustable parameters for:
- **Economic Parameters**: Prices, costs, margins
- **Demand Patterns**: Mean demand, seasonality factors
- **Simulation Settings**: Horizon, inventory limits, forecasting

### Baseline Policies (`baselines.py`)
- **Naive Policy**: Simple reorder-to-target (T) strategy
- **sS Policy**: Classical (s,S) inventory control with reorder point and target level

## 📈 Performance Analysis

### Why RL Outperforms Traditional Methods

1. **Dynamic Learning**: Adapts to demand patterns and cost structures
2. **Multi-objective Optimization**: Balances competing objectives simultaneously  
3. **Non-linear Decision Making**: Captures complex relationships traditional policies miss
4. **Risk-aware**: Learns optimal service level vs cost trade-offs

### Business Impact
- **Capital Efficiency**: 78% reduction in average inventory
- **Profit Maximization**: 10% increase in returns
- **Scalability**: Easily adaptable to different products/markets
- **Robustness**: Handles demand uncertainty and seasonality

## 🔬 Experimental Setup

### Training Configuration
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Policy Network**: Multi-layer perceptron
- **Training Steps**: 200,000 timesteps
- **Evaluation**: Every 10,000 steps with 5 episodes
- **Environment**: 100-day episodes with Poisson demand

### Hyperparameters
```python
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
    "n_steps": 1024
}
```

## 🚧 Future Enhancements

- [ ] Multi-product inventory management
- [ ] Supply chain delays and lead times
- [ ] Demand forecasting integration
- [ ] Real-world data validation
- [ ] Advanced RL algorithms (SAC, TD3)
- [ ] Hierarchical inventory policies
- [ ] Integration with ERP systems

## 📚 References & Inspiration

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction*
- Silver, E. A., Pyke, D. F., & Peterson, R. (1998). *Inventory management and production planning and scheduling*
- Stable-Baselines3 Documentation: https://stable-baselines3.readthedocs.io/

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Krish Sharma** - krishs0404@gmail.com

Project Link: [https://github.com/yourusername/SmartStockRL](https://github.com/yourusername/SmartStockRL)

---

*Built with ❤️ using Python, Stable-Baselines3, and Gymnasium*
