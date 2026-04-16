import datetime
from typing import Dict, List, Tuple
from decimal import Decimal
from env.base_env import BaseEnv
from env.data.data_handler import DataHandler
from env.base.tick import Tick
from env.base.option import OptionTypes
from env.base.perpetual import Perpetual
from env.base.options_chain import OptionsChain
from env.base.action import Action, OptionAction, HedgeAction
from env.base.positions import Positions, OptionPosition, PerpetualPosition
from env.base.account import Account
from env.base.log import Log
from env.config import fee_config

class RLEnv:
    def __init__(self, env: BaseEnv):
        self.env = env

    def reset(self, start_date: datetime.datetime = None, end_date: datetime.datetime = None) -> Tuple[Dict, Dict]:
        """重置环境
        Args:
            start_date: 可选的开始日期
            end_date: 可选的结束日期
        Returns:
            state: 环境状态
            info: 额外信息
        """
        tick, account, position, log, done = self.env.reset(start_date, end_date)

        state = self.get_state(tick, position)
        info = self.get_info(tick, position, account, log)

        return state, info

    def step(self, option_action: OptionAction, hedge_action: HedgeAction) -> Tuple[Dict, float, bool, Dict]:
        """执行一步交易
        Args:
            option_action: 期权交易动作
            hedge_action: 对冲动作
        Returns:
            state: 新的状态
            reward: 净值变化奖励
            done: 是否结束
            info: 额外信息
        """
        tick, account, position, log, done = self.env.step(option_action, hedge_action)

        state = self.get_state(tick, position)
        # 直接使用净值变化作为奖励
        reward = float(log.get_net_value_change())
        done = self.get_done(position)
        info = self.get_info(tick, position, account, log)

        return state, reward, done, info

    def get_state(self, tick: Tick, position: Positions):
        state = dict()
        state['timestamp'] = tick.timestamp
        state['features'] = tick.perpetual.features
        state['volatility_tickers'] = tick.volatility_tickers.features
        state['options_chain'] = tick.options_chain.puts | tick.options_chain.calls
        state['greeks'] = position.get_greeks()
        state['hedge_history'] = position.get_hedge_history(history_length=24)
        state['position'] = position  # 添加position到state中

        return state

    def get_done(self, positions):
        """只使用时间限制作为结束条件"""
        return self.env._ts >= self.env._end_time

    def get_info(self, tick: Tick, position: Positions, account: Account, log: Log):
        info = dict()
        info['position'] = position
        info['account'] = account
        info['log'] = log
        info['future_info'] = tick.perpetual.future_information

        return info

class OracleEnv(RLEnv):
    def reset(self, start_date: datetime.datetime, end_data: datetime.datetime) -> Tuple[Tick, Dict, Dict]:
        tick, account, position, log, done = self.env.reset(start_date, end_data)

        state = self.get_state(tick, position)
        info = self.get_info(tick, position, account, log)

        return tick, state, info

    def step(self, option_action: OptionAction, hedge_action: HedgeAction) -> Tuple[Tick, Dict, Dict, Decimal, bool, Dict]:
        tick, account, position, log, done = self.env.step(option_action, hedge_action)

        state = self.get_state(tick, position)
        option_reward = self.get_option_reward(position)
        hedge_reward = self.get_hedge_reward(log)
        done = self.get_done(position)
        info = self.get_info(tick, position, account, log)

        return tick, state, option_reward, hedge_reward, done, info

if __name__ == '__main__':
    # 定义开始和结束日期等参数
    start_date = datetime.datetime(2021, 2, 1)
    end_date = datetime.datetime(2024, 6, 30)
    crypto = 'BTC'
    option_chain_path = ''
    perpetual_path = ''
    time_config = {'episode_length': 14, 'option_interval': 180}

    # 启动 DataHandler actor
    data_handler = DataHandler(
        start_date, end_date, crypto, option_chain_path, perpetual_path
    )

    data_handler.reset(start_date)

    # 创建环境，传递 data_handler actor
    env = BaseEnv(data_handler, time_config)

    env = RLEnv(env)

    # 运行回测循环
    done = False
    state, info = env.reset(start_date, end_date)
    while not done:
        # 生成或获取 option_action 和 hedge_action
        option_action = OptionAction(timestamp=state['timestamp'], trades={})
        hedge_action = HedgeAction(timestamp=state['timestamp'], hedge=False)

        state, total_reward, done, info = env.step(option_action, hedge_action)
