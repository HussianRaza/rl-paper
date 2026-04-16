import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List


def plot_backtest_results(
    results: Dict[str, List],
    save_path: str = None,
    figsize: tuple = (15, 12)
):

    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle('Backtest Results', fontsize=16)
    
    # Extract data
    pnl = results['pnl_history']
    delta = results['delta_history']
    gamma = results['gamma_history']
    vega = results['vega_history']
    theta = results['theta_history']
    underlying = results['underlying_price_history']
    position = results['position_history']
    timestamps = results.get('timestamps_history', range(len(pnl)))
    direction = results.get('direction_history', [])
    
    # Convert to numpy arrays
    pnl = np.array(pnl)
    delta = np.array(delta)
    gamma = np.array(gamma)
    vega = np.array(vega)
    theta = np.array(theta)
    underlying = np.array(underlying)
    
    # 1. PnL Curve
    ax1 = axes[0, 0]
    ax1.plot(pnl, label='PnL', color='blue', linewidth=1.5)
    ax1.set_title('Portfolio Value (PnL)')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Portfolio Value')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Underlying Price
    ax2 = axes[0, 1]
    ax2.plot(underlying, label='Underlying Price', color='orange', linewidth=1.5)
    ax2.set_title('Underlying Asset Price')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Price')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Delta
    ax3 = axes[1, 0]
    ax3.plot(delta, label='Delta', color='green', linewidth=1.5)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax3.set_title('Delta Exposure')
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Delta')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Gamma
    ax4 = axes[1, 1]
    ax4.plot(gamma, label='Gamma', color='red', linewidth=1.5)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax4.set_title('Gamma Exposure')
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Gamma')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 5. Vega and Theta
    ax5 = axes[2, 0]
    ax5_twin = ax5.twinx()
    ax5.plot(vega, label='Vega', color='purple', linewidth=1.5)
    ax5_twin.plot(theta, label='Theta', color='brown', linewidth=1.5, alpha=0.7)
    ax5.set_title('Vega and Theta Exposure')
    ax5.set_xlabel('Time (hours)')
    ax5.set_ylabel('Vega', color='purple')
    ax5_twin.set_ylabel('Theta', color='brown')
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='upper left')
    ax5_twin.legend(loc='upper right')
    
    # 6. Position
    ax6 = axes[2, 1]
    ax6.plot(position, label='Hedge Position', color='teal', linewidth=1.5)
    ax6.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax6.set_title('Hedge Position (Perpetual)')
    ax6.set_xlabel('Time (hours)')
    ax6.set_ylabel('Position')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_trade_analysis(
    trade_returns: np.ndarray,
    trade_directions: np.ndarray,
    save_path: str = None
):

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Separate long and short trades
    long_trades = trade_returns[trade_directions == 1]
    short_trades = trade_returns[trade_directions == -1]
    
    # 1. Trade Return Distribution
    ax1 = axes[0]
    ax1.hist(long_trades, bins=30, alpha=0.6, label='Long', color='green')
    ax1.hist(short_trades, bins=30, alpha=0.6, label='Short', color='red')
    ax1.set_title('Trade Return Distribution')
    ax1.set_xlabel('Return')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative Trade Performance
    ax2 = axes[1]
    cumulative_pnl = np.cumsum(trade_returns)
    ax2.plot(cumulative_pnl, linewidth=2)
    ax2.set_title('Cumulative Trade PnL')
    ax2.set_xlabel('Trade Number')
    ax2.set_ylabel('Cumulative PnL')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()



