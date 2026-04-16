"""
Performance Metrics for OPHR
1. TR - Total Return
2. AVOL - Annual Volatility
3. MDD - Maximum Drawdown
4. ASR - Annual Sharpe Ratio
5. ACR - Annual Calmar Ratio
6. ASoR - Annual Sortino Ratio
7. WR - Win Rate
8. PLR - Profit/Loss Ratio
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    TR: float  # Total Return
    AVOL: float  # Annual Volatility
    MDD: float  # Maximum Drawdown
    ASR: float  # Annual Sharpe Ratio
    ACR: float  # Annual Calmar Ratio
    ASoR: float  # Annual Sortino Ratio
    WR: float  # Win Rate (%)
    PLR: float  # Profit/Loss Ratio
    HP: float = None  # Average Holding Period (hours)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'Total Return (TR)': self.TR,
            'Annual Volatility (AVOL)': self.AVOL,
            'Maximum Drawdown (MDD)': self.MDD,
            'Annual Sharpe Ratio (ASR)': self.ASR,
            'Annual Calmar Ratio (ACR)': self.ACR,
            'Annual Sortino Ratio (ASoR)': self.ASoR,
            'Win Rate (WR)': self.WR,
            'Profit/Loss Ratio (PLR)': self.PLR,
            'Holding Period (HP)': self.HP if self.HP is not None else 0.0
        }
    
    def __str__(self):
        """String representation"""
        lines = [
            f"Total Return (TR): {self.TR:.4f} ({self.TR*100:.2f}%)",
            f"Annual Volatility (AVOL): {self.AVOL:.4f} ({self.AVOL*100:.2f}%)",
            f"Maximum Drawdown (MDD): {self.MDD:.4f} ({self.MDD*100:.2f}%)",
            f"Annual Sharpe Ratio (ASR): {self.ASR:.4f}",
            f"Annual Calmar Ratio (ACR): {self.ACR:.4f}",
            f"Annual Sortino Ratio (ASoR): {self.ASoR:.4f}",
            f"Win Rate (WR): {self.WR:.2f}%",
            f"Profit/Loss Ratio (PLR): {self.PLR:.4f}"
        ]
        if self.HP is not None:
            lines.append(f"Holding Period (HP): {self.HP:.2f} hours")
        return '\n'.join(lines)


def calculate_total_return(pnl_history: np.ndarray) -> float:
    V1 = pnl_history[0]
    Vt = pnl_history[-1]
    return float((Vt - V1) / V1)


def calculate_annual_volatility(daily_returns: np.ndarray, annual_days: int = 365) -> float:
    if daily_returns.size == 0:
        return 0.0
    return float(np.std(daily_returns) * np.sqrt(annual_days))


def calculate_maximum_drawdown(pnl_history: np.ndarray) -> float:
    if pnl_history.size == 0:
        return 0.0
    cummax = np.maximum.accumulate(pnl_history)
    # Avoid divide-by-zero if cummax contains zeros
    safe_cummax = np.where(cummax == 0, 1e-12, cummax)
    drawdown_ratio = (cummax - pnl_history) / safe_cummax
    return float(np.max(drawdown_ratio))


def calculate_sharpe_ratio(
    daily_returns: np.ndarray,
    annual_days: int = 365,
    risk_free_rate: float = 0.0
) -> float:
    if daily_returns.size == 0:
        return 0.0
    mean_return = np.mean(daily_returns)
    std_return = np.std(daily_returns)
    
    if std_return == 0:
        return 0.0
    
    # Daily risk-free rate
    daily_rf = risk_free_rate / annual_days
    
    sharpe = float((mean_return - daily_rf) / std_return * np.sqrt(annual_days))
    return sharpe


def calculate_calmar_ratio(
    daily_returns: np.ndarray,
    mdd: float,
    annual_days: int = 365
) -> float:
    if daily_returns.size == 0:
        return 0.0
    mean_return = np.mean(daily_returns)
    
    if mdd == 0:
        return 0.0
    
    calmar = float(mean_return * annual_days / mdd )
    return calmar


def calculate_sortino_ratio(
    daily_returns: np.ndarray,
    annual_days: int = 365,
    risk_free_rate: float = 0.0
) -> float:
    if daily_returns.size == 0:
        return 0.0
    target_daily = risk_free_rate / annual_days
    excess = daily_returns - target_daily
    downside = excess[excess < 0]
    if downside.size == 0:
        return 0.0
    downside_std = np.std(downside)
    if downside_std == 0:
        return 0.0
    numerator = np.mean(excess) * np.sqrt(annual_days)
    return float(numerator / downside_std)


def extract_trades(
    direction_history: List[int],
    pnl_history: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    trade_returns = []
    trade_directions = []
    trade_entry_pnl = None
    trade_entry_dir = None
    
    direction_arr = np.array(direction_history)
    
    for i in range(1, len(direction_arr)):
        # Detect position change
        if direction_arr[i] != direction_arr[i-1]:
            # Close previous position
            if direction_arr[i-1] != 0 and trade_entry_pnl is not None:
                trade_return = pnl_history[i] - trade_entry_pnl
                trade_returns.append(trade_return)
                trade_directions.append(trade_entry_dir)
            
            # Open new position
            if direction_arr[i] != 0:
                trade_entry_pnl = pnl_history[i]
                trade_entry_dir = direction_arr[i]
            else:
                trade_entry_pnl = None
                trade_entry_dir = None
    
    return np.array(trade_returns), np.array(trade_directions)


def calculate_win_rate(trade_returns: np.ndarray) -> float:

    if len(trade_returns) == 0:
        return 0.0
    
    num_wins = np.sum(trade_returns > 0)
    win_rate = float(num_wins / len(trade_returns) * 100)
    return win_rate


def calculate_profit_loss_ratio(trade_returns: np.ndarray) -> float:

    winning_trades = trade_returns[trade_returns > 0]
    losing_trades = trade_returns[trade_returns < 0]
    
    if len(losing_trades) == 0:
        return np.inf if len(winning_trades) > 0 else 0.0
    
    mean_win = np.abs(np.mean(winning_trades)) if len(winning_trades) > 0 else 0
    mean_loss = np.abs(np.mean(losing_trades))
    
    plr = float(mean_win / mean_loss) if mean_loss > 0 else 0.0
    return plr


def calculate_average_holding_period(
    direction_history: List[int],
    hourly: bool = True
) -> float:

    holding_periods = []
    current_hold_start = None
    current_direction = 0
    
    for i, direction in enumerate(direction_history):
        if direction != 0 and current_direction == 0:
            # Start of new position
            current_hold_start = i
            current_direction = direction
        elif direction == 0 and current_direction != 0:
            # End of position
            if current_hold_start is not None:
                hold_period = i - current_hold_start
                holding_periods.append(hold_period)
            current_hold_start = None
            current_direction = 0
        elif direction != current_direction and direction != 0 and current_direction != 0:
            # Direction change (rare in this implementation but possible)
            if current_hold_start is not None:
                hold_period = i - current_hold_start
                holding_periods.append(hold_period)
            current_hold_start = i
            current_direction = direction
    
    # Handle open position at end
    if current_hold_start is not None and current_direction != 0:
        holding_periods.append(len(direction_history) - current_hold_start)
    
    if len(holding_periods) == 0:
        return 0.0
    
    avg_period = np.mean(holding_periods)
    return float(avg_period) if hourly else float(avg_period)


def calculate_all_metrics(
    pnl_history: List[float],
    direction_history: List[int],
    hourly_sampling: int = 24,
    annual_days: int = 365,
    risk_free_rate: float = 0.0
) -> PerformanceMetrics:

    pnl_arr = np.array(pnl_history)
    
    # Sample at end of each day
    daily_pnl = pnl_arr[hourly_sampling-1::hourly_sampling]
    
    # Calculate daily returns
    daily_returns = np.diff(daily_pnl) / (daily_pnl[:-1] + 1e-12)
    
    # Calculate basic metrics
    TR = calculate_total_return(pnl_arr)
    AVOL = calculate_annual_volatility(daily_returns, annual_days)
    MDD = calculate_maximum_drawdown(pnl_arr)
    
    # Calculate risk-adjusted metrics
    ASR = calculate_sharpe_ratio(daily_returns, annual_days, risk_free_rate)
    ACR = calculate_calmar_ratio(daily_returns, MDD, annual_days)
    ASoR = calculate_sortino_ratio(daily_returns, annual_days, risk_free_rate)
    
    # Extract trades and calculate trade metrics
    trade_returns, trade_directions = extract_trades(direction_history, pnl_arr)
    
    WR = calculate_win_rate(trade_returns)
    PLR = calculate_profit_loss_ratio(trade_returns)
    HP = calculate_average_holding_period(direction_history, hourly=True)
    
    return PerformanceMetrics(
        TR=TR,
        AVOL=AVOL,
        MDD=MDD,
        ASR=ASR,
        ACR=ACR,
        ASoR=ASoR,
        WR=WR,
        PLR=PLR,
        HP=HP
    )


def print_metrics(metrics: PerformanceMetrics):

    print(metrics)


def save_metrics(metrics: PerformanceMetrics, filepath: str):

    with open(filepath, 'w') as f:
        f.write(str(metrics))



