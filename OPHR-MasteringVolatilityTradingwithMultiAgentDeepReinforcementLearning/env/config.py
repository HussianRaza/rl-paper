#config of env

#time period config
time_config ={
    'episode_length': 3,  # 14days testing
    'option_interval': 180,  # 3hours
    'hedge_interval': 5  # 5mins
}

# 费用配置
fee_config = {
    'BTC_ETH_Futures_Perpetual': 0.0000,  # 0.05%
    'BTC_ETH_Options': {
        'per_option_contract': 0.0003,  # 0.03%
        'capped_at': 0.125  # capped price
    },
    'Option_Combo_fees': {
        'second_leg_reduction': 1  # second leg free
    }
}

# 文件配置
file_config = {
    'log_file_path': None,  #to add
    'PNL_file_path': None,
    'plt_path': None
}

backtest_config = {
    'init_cash' : 100000
}