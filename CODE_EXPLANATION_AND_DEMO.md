# OPHR Code Explanation & Demo Guide

## Complete Codebase Walkthrough

This document explains every module in the OPHR codebase and provides step-by-step instructions for running a demo.

---

## 1. Repository Structure

```
OPHR-MasteringVolatilityTradingwithMultiAgentDeepReinforcementLearning/
|-- config.py                  # Configuration management (YAML loading, dataclasses)
|-- requirements.txt           # Python dependencies
|-- configs/                   # YAML configuration files
|   |-- env_config.yaml        # Environment settings (episode length, fees, data paths)
|   |-- training_config.yaml   # Training hyperparameters for both agents
|   |-- hedger_config.yaml     # Hedger pool configuration
|   |-- evaluation_config.yaml # Evaluation metrics and visualization settings
|-- agents/                    # RL Agent implementations
|   |-- op_agent.py            # Option Position Agent (DQN with n-step TD)
|   |-- hr_agent.py            # Hedger Routing Agent (DQN with hedger pool)
|   |-- replay_buffer.py       # Experience replay and n-step buffer
|-- oracle/
|   |-- oracle_policy.py       # Oracle policy (uses future RV for signal generation)
|-- hedgers/                   # Hedging strategy implementations
|   |-- base_hedger.py         # Abstract base class for all hedgers
|   |-- delta_hedger.py        # Delta-threshold based hedger (baseline)
|   |-- price_move_hedger.py   # Price-movement based hedger
|   |-- deep_hedger.py         # Neural network based hedger (loads pretrained models)
|-- env/                       # Trading environment
|   |-- base_env.py            # Core environment with margin, fees, Greeks
|   |-- rl_env.py              # RL wrapper (state/reward/done interface)
|   |-- config.py              # Fee configuration constants
|   |-- base/                  # Financial primitives
|   |   |-- option.py          # Option contract (Call/Put with Greeks)
|   |   |-- perpetual.py       # Perpetual futures contract
|   |   |-- positions.py       # Position tracking (options + perpetual)
|   |   |-- account.py         # Account management (balance, margin)
|   |   |-- action.py          # Action dataclasses (OptionAction, HedgeAction)
|   |   |-- tick.py            # Market tick (timestamp + all market data)
|   |   |-- options_chain.py   # Full options chain at a given time
|   |   |-- log.py             # Trade and account logging
|   |   |-- trade.py           # Trade record
|   |   |-- call.py / put.py   # Call and Put option specifics
|   |   |-- volatility_tickers.py  # Volatility surface features
|   |-- data/                  # Data loading
|   |   |-- data_handler.py    # Main data handler (loads Parquet/pickle data)
|   |   |-- data_handler_ray.py # Ray-distributed version for parallel training
|   |   |-- utils.py           # Data utilities
|   |-- evaluation/
|       |-- evaluation.py      # Environment-level evaluation
|-- training/                  # Training pipelines
|   |-- phase1_oracle.py       # Phase 1: Oracle experience collection
|   |-- phase1_op_offline.py   # Phase 1: Offline OP-Agent training
|   |-- phase1_hr_warmup.py    # Phase 1: HR-Agent warm-up
|   |-- phase2_iterative.py    # Phase 2: Iterative alternating training
|   |-- checkpoint_utils.py    # Save/load model checkpoints
|-- backtest/
|   |-- backtest.py            # Full backtesting pipeline
|-- evaluation/
|   |-- metrics.py             # Performance metrics (TR, ASR, MDD, etc.)
|   |-- visualize.py           # Plotting functions
|-- sample_data/
    |-- BTC/                   # Sample BTC data (Jan-Apr 2024)
```

---

## 2. Module-by-Module Explanation

### 2.1 Configuration System (`config.py`)

The configuration system uses Python dataclasses and YAML files. The central `ConfigManager` class lazily loads configurations on first access:

```python
class ConfigManager:
    def __init__(self, config_dir='configs', data_root=None, crypto=None):
        # Environment variables resolve ${DATA_ROOT} and ${CRYPTO} in YAML paths
        self._env_vars = {
            'DATA_ROOT': data_root or 'sample_data',
            'CRYPTO': crypto or 'BTC'
        }
```

**Key Config Classes:**
- `EnvConfig`: Episode length (14 days), option trading interval (180 min), hedge interval (60 min), initial capital (10 BTC), PM margin parameters, fee structure, data paths.
- `TrainingConfig`: OP-Agent (hidden_dims=[1024,1024], lr=0.0001, n_step=12, gamma=0.99, batch=512, buffer=100K). HR-Agent (same hidden dims, n_hr=24 hours, buffer=50K). Iterative: 5 iterations, 200 OP + 50 HR episodes per iteration.
- `HedgerPoolConfig`: Defines the pool of available hedgers for the HR-Agent.
- `EvaluationConfig`: Metrics to compute, visualization settings, output directories.

### 2.2 Agents (`agents/`)

#### OP-Agent (`agents/op_agent.py`)

The OP-Agent is a Double DQN with n-step TD learning. Its purpose is volatility timing.

**QNetwork Architecture:**
```python
class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=3):
        # input_dim ~96 (48 volatility + 48 perpetual features)
        # hidden_dims = [1024, 1024] (two fully-connected layers)
        # output_dim = 3 (long=0, neutral=1, short=2)
        # Each hidden layer: Linear -> ReLU
```

**Key Methods:**
- `extract_features(state)`: Concatenates `volatility_tickers` (IV surface features like ATM IV, skew, term structure) and `features` (perpetual futures features like returns, volume, funding rate) into a flat numpy array (~96 dimensions).
- `select_action(state, epsilon)`: Epsilon-greedy action selection. With probability epsilon, picks a random action {0,1,2}; otherwise, feeds features through Q-network and picks argmax.
- `action_to_direction(action)`: Maps action index to trading direction: 0->+1 (long), 1->0 (neutral), 2->-1 (short).
- `store_transition(state, action, reward, next_state, done)`: Pushes transitions through the NStepBuffer. Once n_step transitions accumulate, the discounted n-step return is computed and stored in the main ReplayBuffer.
- `update()`: Samples a batch from the replay buffer, computes Double DQN targets using n-step returns and the target network, and performs gradient descent on MSE loss.

**n-step mechanism (critical for understanding):**
```python
class NStepBuffer:
    def get_n_step_transition(self):
        # Computes: R = r_0 + gamma*r_1 + gamma^2*r_2 + ... + gamma^(n-1)*r_{n-1}
        # Returns: (s_0, a_0, R, s_n, done_n)
        # This allows the OP-Agent to evaluate positions over 12-hour windows
        # rather than judging each hour independently
```

#### HR-Agent (`agents/hr_agent.py`)

The HR-Agent selects which hedger to use from a pool. It uses standard DQN (1-step, not n-step).

**HRQNetwork:** Same MLP architecture as QNetwork but output_dim = number of hedgers in the pool (typically 8: 5 delta-threshold + 3 price-move hedgers).

**Hedger Pool Creation:**
```python
def _create_hedger_pool(self, config):
    # Default pool (when no config provided):
    # DeltaThresholdHedger: thresholds [0.05, 0.10, 0.20, 0.30, 3.0]
    #   - 0.05 = very aggressive (hedges at tiny delta deviations)
    #   - 3.0 = effectively never hedges (extreme risk-seeking)
    # PriceMoveHedger: thresholds [0.01, 0.02, 0.03]
    #   - Hedges based on price movement percentage
```

**Key Methods:**
- `extract_features(state, position, greeks)`: Extends market features with position info (number of options, perpetual quantity) and Greeks (delta, gamma, theta, vega). Total ~102 dimensions.
- `select_hedger(state, position, greeks, epsilon)`: Epsilon-greedy hedger selection.
- `step(state, position, greeks)`: Called every timestep but only makes a new decision every `n_hr` steps (24 hours). Returns the current hedger index.
- `compute_hedge(delta, gamma, theta, vega, ...)`: Delegates to the currently selected hedger.

#### Replay Buffer (`agents/replay_buffer.py`)

Two buffer types:
- `ReplayBuffer`: Standard circular buffer using `deque(maxlen=capacity)`. Stores (state, action, reward, next_state, done) tuples. Supports both 1-step and n-step storage.
- `NStepBuffer`: Sliding window of size n. Computes discounted n-step return: `R = sum(gamma^i * r_i for i in 0..n-1)`. Returns the aggregated transition once full.

### 2.3 Oracle Policy (`oracle/oracle_policy.py`)

The Oracle has access to future information (cheating!) to generate a sub-optimal but instructive policy:

```python
def generate_signal(self, state, info):
    future_rv = self.calculate_future_rv(state, info)  # Looks ahead!
    current_iv = self.get_current_iv(state)             # Current market IV
    
    if future_rv >= current_iv * (1 + self.beta):  # beta=0.1
        return 1   # Long volatility (buy straddle)
    elif future_rv <= current_iv * (1 - self.beta):
        return -1  # Short volatility (sell straddle)
    else:
        return 0   # Neutral (close positions)
```

The `calculate_future_rv` method accesses `info['future_info']` which contains pre-computed realized volatility at horizons of 3h, 6h, 9h, 12h, 18h, and 24h ahead. It selects the horizon closest to the `lookforward_window` (default 168h = 7 days, but effectively uses 24h due to data availability).

The Oracle also handles ATM straddle selection: it finds the option with strike closest to the current underlying price and constructs a call+put pair.

### 2.4 Hedgers (`hedgers/`)

**BaseHedger** (`base_hedger.py`): Abstract class defining the `compute_hedge(delta, gamma, theta, vega, position_info, market_info) -> Decimal` interface.

**DeltaThresholdHedger** (`delta_hedger.py`):
```python
def compute_hedge(self, delta, gamma, theta, vega, position_info, market_info):
    if abs(delta) > self.delta_threshold:
        return -delta * self.hedge_ratio  # Full delta neutralization
    else:
        return Decimal('0')  # No hedge needed
```
This is the classical rule-based approach. The `BaselineHedger` subclass uses threshold=0.1, ratio=1.0.

**PriceMoveHedger** (`price_move_hedger.py`): Hedges based on underlying price movement rather than delta exposure. Triggers when price moves more than `price_move_threshold` (e.g., 1-3%) from the last hedge price.

**DeepHedger** (`deep_hedger.py`): Loads a pre-trained PyTorch model that takes Greeks + market features as input and outputs a hedge ratio. These models are trained separately using actor-critic methods (Murray et al., 2022) with different risk aversion levels. The model is loaded from a `.pt` file and runs inference to determine hedge quantities.

### 2.5 Trading Environment (`env/`)

**BaseEnv** (`env/base_env.py`) is the core simulation engine:

- **Reset:** Randomly selects a start date from available data, initializes account with 10 BTC, creates empty positions and logs.
- **Step:** Processes option trades, hedge trades, handles expirations, computes funding fees, updates net value, calculates portfolio margins.
- **Margin System:** Implements Deribit's Portfolio Margin:
  - Risk Matrix: Simulates PnL across 27 scenarios (9 price moves [-16% to +16%] x 3 vol shocks [0.75x, 1.0x, 1.5x])
  - Extended Table: 8 extreme scenarios [-66% to +500%] with dampening
  - Delta Shock: Additional margin for concentrated directional risk
  - Roll Shock: Margin for expiration-related risk
- **Greeks Tracking:** Aggregates delta, gamma, theta, vega across all option positions and subtracts perpetual delta exposure.
- **State Save/Restore:** Deep-copies entire environment state for twin environment training.

**RLEnv** (`env/rl_env.py`) wraps BaseEnv into a standard RL interface:
- `reset() -> (state, info)`: Returns state dict with timestamp, features, options_chain, Greeks, hedge_history, and position.
- `step(option_action, hedge_action) -> (state, reward, done, info)`: Reward is net value change (`log.get_net_value_change()`). Done when timestamp exceeds end_time.
- State contains: `features` (perpetual), `volatility_tickers` (vol surface), `options_chain` (all active options), `greeks` (portfolio Greeks), `hedge_history` (last 24 hedge actions).

### 2.6 Training Pipeline (`training/`)

**Phase 1.1 - Oracle Collection** (`phase1_oracle.py`):
```
for each episode:
    reset environment
    while not done:
        oracle generates signal using future RV vs IV
        convert signal to OP action
        step environment with oracle's option + hedge actions
        store (state, action, reward, next_state, done) in OP-Agent's buffer
```
Output: Filled replay buffer with Oracle-quality experience.

**Phase 1.2 - OP Offline Training** (`phase1_op_offline.py`):
Train the OP-Agent on the Oracle replay buffer data using batch DQN updates.

**Phase 1.3 - HR Warm-up** (`phase1_hr_warmup.py`):
Initialize HR-Agent with warm-up episodes using the current OP-Agent.

**Phase 2 - Iterative Training** (`phase2_iterative.py`):

The `TwinEnvTrainer` class orchestrates alternating training:

```
for each iteration (1..5):
    # Train OP-Agent (200 episodes)
    for each OP episode:
        HR-Agent selects hedgers (frozen, no learning)
        OP-Agent selects actions with epsilon-greedy
        Store transitions, update OP Q-network
    
    # Train HR-Agent (50 episodes) with Twin Environment
    for each HR episode:
        At each HR decision point (every 24 hours):
            1. Save environment state
            2. Run selected hedger for 24 steps -> get main_net_value
            3. Restore state
            4. Run baseline hedger for 24 steps -> get twin_net_value
            5. HR reward = main_net_value - twin_net_value
            6. Store transition, update HR Q-network
            7. Restore state, advance with selected hedger
```

### 2.7 Backtest (`backtest/backtest.py`)

The `BacktestRunner` class runs trained agents in evaluation mode (epsilon=0, no exploration):

- **Modes:** `'full'` (OP + HR agents), `'op_only'` (OP agent + baseline hedger), `'baseline'` (no agent, neutral position).
- **Tracking:** Records timestamps, net values, rewards, actions, hedger selections, positions, Greeks, and underlying prices at each step.
- **Metrics:** Computes total return, volatility, Sharpe, max drawdown, Calmar, Sortino, position distribution.
- **Report:** Saves CSV files with detailed results and summary metrics.

### 2.8 Evaluation (`evaluation/`)

**metrics.py:** Implements all 8 performance metrics from the paper:
- `calculate_total_return`: (V_final - V_initial) / V_initial
- `calculate_annual_volatility`: std(daily_returns) * sqrt(365)
- `calculate_maximum_drawdown`: max((cummax - value) / cummax)
- `calculate_sharpe_ratio`: (mean_return - rf) / std * sqrt(365)
- `calculate_calmar_ratio`: annualized_return / MDD
- `calculate_sortino_ratio`: excess_return * sqrt(365) / downside_std
- `calculate_win_rate`: % of profitable trades
- `calculate_profit_loss_ratio`: mean_win / mean_loss

**visualize.py:** Creates 6-panel plots showing PnL curve, underlying price, delta/gamma/vega/theta exposure, and hedge positions.

---

## 3. How to Run a Demo

### 3.1 Setup

```bash
# 1. Clone or navigate to the repository
cd OPHR-MasteringVolatilityTradingwithMultiAgentDeepReinforcementLearning

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify sample data exists
ls sample_data/BTC/
# Should show: option_chain/, perp.parquet, volticker.parquet
```

### 3.2 Quick Demo: Run Backtest with Baseline Strategy

This demonstrates the environment works without needing trained models:

```python
# demo_baseline.py
import datetime
from config import ConfigManager
from backtest.backtest import BacktestRunner

config_manager = ConfigManager(data_root='sample_data', crypto='BTC')
data_paths = {
    'option_chain': 'sample_data/BTC/option_chain',
    'perpetual': 'sample_data/BTC/perp.parquet',
    'volatility_ticker': 'sample_data/BTC/volticker.parquet'
}

runner = BacktestRunner(
    crypto='BTC',
    start_date=datetime.datetime(2024, 1, 1),
    end_date=datetime.datetime(2024, 4, 1),
    config_manager=config_manager,
    data_paths=data_paths,
    mode='baseline',
    verbose=True
)

results = runner.run_episode()
metrics = runner.generate_report(results, save_dir='results/baseline')
```

### 3.3 Full Training Pipeline Demo

```python
# demo_training.py
import datetime
from config import ConfigManager
from agents.op_agent import OPAgent
from agents.hr_agent import HRAgent
from training.phase1_oracle import collect_oracle_experience
from training.phase2_iterative import train_iterative

# Configuration
config_manager = ConfigManager(data_root='sample_data', crypto='BTC')
data_paths = {
    'option_chain': 'sample_data/BTC/option_chain',
    'perpetual': 'sample_data/BTC/perp.parquet',
    'volatility_ticker': 'sample_data/BTC/volticker.parquet'
}

start_date = datetime.datetime(2024, 1, 1)
end_date = datetime.datetime(2024, 4, 1)

# Phase 1: Oracle Experience Collection
# (Reduce num_episodes for quick demo, paper uses 1000)
op_agent, oracle_stats = collect_oracle_experience(
    crypto='BTC',
    start_date=start_date,
    end_date=end_date,
    num_episodes=10,  # Use 10 for quick demo, 1000 for full
    config_manager=config_manager,
    data_paths=data_paths,
    save_checkpoint=True,
    verbose=True
)

# Phase 2: Iterative Training
# (The training config specifies 5 iterations, 200+50 episodes each)
# For demo, you can modify configs/training_config.yaml to reduce these
op_agent, hr_agent = train_iterative(
    crypto='BTC',
    start_date=start_date,
    end_date=end_date,
    config_manager=config_manager,
    data_paths=data_paths,
    op_agent=op_agent,
    save_checkpoint=True,
    verbose=True
)

print("Training complete!")
```

### 3.4 Backtest with Trained Agents

```python
# demo_backtest.py
import datetime
from config import ConfigManager
from backtest.backtest import run_backtest
from evaluation.visualize import plot_backtest_results
from evaluation.metrics import calculate_all_metrics

config_manager = ConfigManager(data_root='sample_data', crypto='BTC')
data_paths = {
    'option_chain': 'sample_data/BTC/option_chain',
    'perpetual': 'sample_data/BTC/perp.parquet',
    'volatility_ticker': 'sample_data/BTC/volticker.parquet'
}

# Run with trained agents
results, metrics_df = run_backtest(
    crypto='BTC',
    start_date=datetime.datetime(2024, 1, 1),
    end_date=datetime.datetime(2024, 4, 1),
    config_manager=config_manager,
    data_paths=data_paths,
    op_checkpoint='checkpoints/phase2_iterative/final/op_agent.pt',
    hr_checkpoint='checkpoints/phase2_iterative/final/hr_agent.pt',
    mode='full',
    save_dir='results/ophr_full',
    verbose=True
)

# Visualize results
plot_results = {
    'pnl_history': results['net_values'],
    'delta_history': results['greeks']['delta'],
    'gamma_history': results['greeks']['gamma'],
    'theta_history': results['greeks']['theta'],
    'vega_history': results['greeks']['vega'],
    'underlying_price_history': results['underlying_prices'],
    'position_history': results['perp_positions'],
    'direction_history': results['actions']
}
plot_backtest_results(plot_results, save_path='results/ophr_plots.png')

# Calculate detailed metrics
metrics = calculate_all_metrics(
    pnl_history=results['net_values'],
    direction_history=results['actions'],
    hourly_sampling=24
)
print(metrics)
```

### 3.5 For a Live Presentation Demo

For the viva/presentation, the recommended approach is:

1. **Pre-train the model** before the presentation (even with reduced episodes). Save checkpoints.
2. **Prepare a Jupyter Notebook** that:
   - Loads the pre-trained model checkpoints
   - Runs a backtest on the sample data
   - Displays the PnL curve, Greeks evolution, and position timeline
   - Computes and prints all 8 evaluation metrics
   - Shows a comparison between OPHR (full), OP-only, and baseline
3. **Show the training code** but explain you ran it beforehand to save time.
4. **Key visualizations to show:**
   - PnL curve over the test period
   - Delta exposure showing hedging effectiveness
   - Position actions (long/short/neutral) overlaid on the price chart
   - Hedger selection distribution (which hedgers the HR-Agent preferred)

### 3.6 Quick Demo if Data/Training Issues Arise

If the full pipeline has issues (data format, compute time), you can demonstrate understanding by:

```python
# demo_agents_only.py - Shows agent architecture and decision-making
import torch
import numpy as np
from agents.op_agent import OPAgent
from agents.hr_agent import HRAgent

# Create OP-Agent
op = OPAgent(state_dim=96, hidden_dims=[1024, 1024])
print(f"OP-Agent Q-Network:\n{op.q_network}")
print(f"Parameters: {sum(p.numel() for p in op.q_network.parameters()):,}")

# Simulate a state
fake_state = {
    'volatility_tickers': np.random.randn(48).astype(np.float32),
    'features': np.random.randn(48).astype(np.float32)
}

# Show action selection
action = op.select_action(fake_state, epsilon=0.0)
direction = op.action_to_direction(action)
print(f"\nOP-Agent action: {action}, direction: {['Long', 'Neutral', 'Short'][action]}")

# Q-values
features = op.extract_features(fake_state)
with torch.no_grad():
    q_values = op.q_network(torch.FloatTensor(features).unsqueeze(0))
print(f"Q-values: Long={q_values[0,0]:.4f}, Neutral={q_values[0,1]:.4f}, Short={q_values[0,2]:.4f}")

# Create HR-Agent
hr = HRAgent(state_dim=102, hidden_dims=[1024, 1024])
print(f"\nHR-Agent has {hr.num_hedgers} hedgers in pool:")
for i, h in enumerate(hr.hedgers):
    print(f"  {i}: {h}")
```

---

## 4. Key Concepts for Viva Preparation

### Q: What is the n-step TD update and why is it important here?
**A:** Standard 1-step TD uses `R = r + gamma * V(s')`. With n-step, we use `R = r_0 + gamma*r_1 + ... + gamma^{n-1}*r_{n-1} + gamma^n * V(s_n)`. In options trading, a position decision at hour 0 might not show its value until hour 12 (when theta accumulates or gamma is realized). The 12-step return captures this delayed reward signal, reducing bootstrap bias at the cost of higher variance.

### Q: Why two agents instead of one?
**A:** The action space of "what position to take" and "how to hedge" are fundamentally different decisions with different time scales. OP decides every step, HR decides every 24 hours. A single agent would need to learn a massive joint policy. The decomposition mirrors real trading desks where the portfolio manager and risk manager are separate roles.

### Q: How does the twin environment work?
**A:** At each HR decision point: (1) save environment state, (2) run selected hedger for 24 steps, record net value, (3) restore state, (4) run baseline hedger for 24 steps, record net value, (5) HR reward = difference. This relative reward eliminates the confounding effect of the option position's own PnL and isolates the hedging contribution.

### Q: What makes the Oracle "sub-optimal"?
**A:** The Oracle uses future information (which is impossible in production) but with a blunt threshold (beta=0.1) and a fixed baseline hedger. It doesn't optimize path-dependent PnL, doesn't consider transaction costs in its signal, and uses a simple straddle. The RL agent can learn to improve upon it by discovering better timing, position management, and hedging.

### Q: How are Greeks used in the framework?
**A:** Delta = price sensitivity (what we hedge to zero). Gamma = convexity (the source of profit in long vol). Theta = time decay (the cost of long vol). Vega = IV sensitivity. The HR-Agent receives all four Greeks as state features, enabling it to understand whether the portfolio needs aggressive hedging (high gamma, trending market) or minimal hedging (low gamma, range-bound market).
