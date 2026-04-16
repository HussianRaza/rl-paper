import numpy as np
import decimal
from decimal import Decimal


def calculate_metrics(log, epsilon=Decimal('1e-10')):
    """
    计算回测指标
    
    Args:
        log: 账户日志
        epsilon: 用于防止除零的小数值
    
    Returns:
        dict: 包含各项指标的字典
    """
    account_log = log.total_value_history
    try:
        if not account_log:
            return {
                'total_return': Decimal('0'),
                'annual_return': Decimal('0'),
                'annual_volatility': Decimal('0'),
                'sharpe_ratio': Decimal('0'),
                'max_drawdown': Decimal('0'),
                'calmar_ratio': Decimal('0'),
                'sortino_ratio': Decimal('0'),
                'hedge_count': 0,
                'trade_count': 0
            }
        
        values_list = np.array(account_log)
        returns = np.diff(values_list) / values_list[:-1]
        
        # 总收益计算
        total_return = Decimal(str((values_list[-1] - values_list[0]) / values_list[0]))
        
        # 年化波动率
        annual_volatility = Decimal(str(np.std(returns))) * Decimal(str(np.sqrt(365 * 24 * 12)))
        
        # 最大回撤计算
        cumulative_max = np.maximum.accumulate(values_list)
        drawdowns = (values_list - cumulative_max) / cumulative_max
        max_drawdown = Decimal(str(abs(np.min(drawdowns))))
        
        # 年化收益
        avg_return = Decimal(str(np.mean(returns)))
        annual_return = avg_return * Decimal('365') * Decimal('24') * Decimal('12')
        
        # 夏普比率
        sharpe_ratio = avg_return / annual_volatility if annual_volatility > epsilon else Decimal('0')
        
        # 卡玛比率
        calmar_ratio = annual_return / max_drawdown if max_drawdown > epsilon else Decimal('0')
        
        # 索提诺比率
        downside_returns = returns[returns < 0]
        downside_deviation = Decimal(str(np.std(downside_returns))) if len(downside_returns) > 0 else Decimal('0')
        sortino_ratio = avg_return / downside_deviation * Decimal(str(np.sqrt(365 * 24 * 12))) if downside_deviation > epsilon else Decimal('0')
        
        # 对冲和交易次数
        hedge_count = sum(1 for h in log.hedges if h)
        trade_count = len(log.trades)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'hedge_count': hedge_count,
            'trade_count': trade_count
        }
        
    except Exception as e:
        print(f"计算指标时出错: {str(e)}")
        return {
            'total_return': Decimal('0'),
            'annual_return': Decimal('0'),
            'annual_volatility': Decimal('0'),
            'sharpe_ratio': Decimal('0'),
            'max_drawdown': Decimal('0'),
            'calmar_ratio': Decimal('0'),
            'sortino_ratio': Decimal('0'),
            'hedge_count': 0,
            'trade_count': 0
        }