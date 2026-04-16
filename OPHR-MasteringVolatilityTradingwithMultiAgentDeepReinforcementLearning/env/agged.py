import pandas as pd
from rv import rv,rv_h,log_return


def agged(df: pd.DataFrame) -> pd.DataFrame:
    # 确保时间戳列为 datetime 类型
    df["timestamp"] = pd.to_datetime(df["timestamp"] * 1000)
    
    # 删除 'local_timestamp' 列
    if 'local_timestamp' in df.columns:
        df.drop('local_timestamp', axis=1, inplace=True)
    
    
    # 重置索引
    df.reset_index(drop=True, inplace=True)
    
    # 设定时间戳为索引，并进行重新采样
    df.set_index('timestamp', inplace=True)
    agged_trades = df.resample('5T').agg(
        {'price': 'ohlc', 'amount': 'sum'})
    
    # 处理多级列索引，保留最后一级
    agged_trades.columns = [col[-1] if isinstance(col, tuple) else col for col in agged_trades.columns]

    agged_trades.bfill(inplace=True)
    agged_trades.ffill(inplace=True)
    
    # 去除缺失值
    agged_trades.dropna(inplace=True)
    
    # 重置索引
    agged_trades.reset_index(inplace=True)
    
    return agged_trades

def aggderi(df: pd.DataFrame) -> pd.DataFrame:
    # 确保时间戳列为 datetime 类型
    df["timestamp"] = pd.to_datetime(df["timestamp"] * 1000)
    
    # 删除 'local_timestamp' 列
    if 'local_timestamp' in df.columns:
        df.drop('local_timestamp', axis=1, inplace=True)
    
    
    # 重置索引
    df.reset_index(drop=True, inplace=True)
    
    # 设定时间戳为索引，并进行重新采样
    df.set_index('timestamp', inplace=True)
    agged_deris = df.resample('5T').agg(
        {'mark_price': 'ohlc', 'funding_rate': 'mean'})
    
    # 处理多级列索引，保留最后一级
    agged_deris.columns = [col[-1] if isinstance(col, tuple) else col for col in agged_deris.columns]

    agged_deris.bfill(inplace=True)
    agged_deris.ffill(inplace=True)
    
    # 去除缺失值
    agged_deris.dropna(inplace=True)
    
    # 重置索引
    agged_deris.reset_index(inplace=True)
    return agged_deris


def aggperpetual(df: pd.DataFrame) -> pd.DataFrame: #perpetual reading
    # 确保时间戳列为 datetime 类型
    df["timestamp"] = pd.to_datetime(df["timestamp"] * 1000)

    # 删除 'local_timestamp' 列
    if 'local_timestamp' in df.columns:
        df.drop('local_timestamp', axis=1, inplace=True)
    
    # 重置索引
    df.reset_index(drop=True, inplace=True)
    
    # 设定时间戳为索引，并进行重新采样
    df.set_index('timestamp', inplace=True)                    
    agged_perpe = df.resample('5T').agg({
        'asks[0].price': 'last', 'asks[0].amount': 'last',
        'asks[1].price': 'last', 'asks[1].amount': 'last',
        'asks[2].price': 'last', 'asks[2].amount': 'last',
        'asks[3].price': 'last', 'asks[3].amount': 'last',
        'asks[4].price': 'last', 'asks[4].amount': 'last',
        'asks[5].price': 'last', 'asks[5].amount': 'last',
        'asks[6].price': 'last', 'asks[6].amount': 'last',
        'asks[7].price': 'last', 'asks[7].amount': 'last',
        'asks[8].price': 'last', 'asks[8].amount': 'last',
        'asks[9].price': 'last', 'asks[9].amount': 'last',
        'asks[10].price': 'last', 'asks[10].amount': 'last',
        'asks[11].price': 'last', 'asks[11].amount': 'last',
        'asks[12].price': 'last', 'asks[12].amount': 'last',
        'asks[13].price': 'last', 'asks[13].amount': 'last',
        'asks[14].price': 'last', 'asks[14].amount': 'last',
        'asks[15].price': 'last', 'asks[15].amount': 'last',
        'asks[16].price': 'last', 'asks[16].amount': 'last',
        'asks[17].price': 'last', 'asks[17].amount': 'last',
        'asks[18].price': 'last', 'asks[18].amount': 'last',
        'asks[19].price': 'last', 'asks[19].amount': 'last',
        'asks[20].price': 'last', 'asks[20].amount': 'last',
        'asks[21].price': 'last', 'asks[21].amount': 'last',
        'asks[22].price': 'last', 'asks[22].amount': 'last',
        'asks[23].price': 'last', 'asks[23].amount': 'last',
        'asks[24].price': 'last', 'asks[24].amount': 'last',
        'bids[0].price': 'last', 'bids[0].amount': 'last',
        'bids[1].price': 'last', 'bids[1].amount': 'last',
        'bids[2].price': 'last', 'bids[2].amount': 'last',
        'bids[3].price': 'last', 'bids[3].amount': 'last',
        'bids[4].price': 'last', 'bids[4].amount': 'last',
        'bids[5].price': 'last', 'bids[5].amount': 'last',
        'bids[6].price': 'last', 'bids[6].amount': 'last',
        'bids[7].price': 'last', 'bids[7].amount': 'last',
        'bids[8].price': 'last', 'bids[8].amount': 'last',
        'bids[9].price': 'last', 'bids[9].amount': 'last',
        'bids[10].price': 'last', 'bids[10].amount': 'last',
        'bids[11].price': 'last', 'bids[11].amount': 'last',
        'bids[12].price': 'last', 'bids[12].amount': 'last',
        'bids[13].price': 'last', 'bids[13].amount': 'last',
        'bids[14].price': 'last', 'bids[14].amount': 'last',
        'bids[15].price': 'last', 'bids[15].amount': 'last',
        'bids[16].price': 'last', 'bids[16].amount': 'last',
        'bids[17].price': 'last', 'bids[17].amount': 'last',
        'bids[18].price': 'last', 'bids[18].amount': 'last',
        'bids[19].price': 'last', 'bids[19].amount': 'last',
        'bids[20].price': 'last', 'bids[20].amount': 'last',
        'bids[21].price': 'last', 'bids[21].amount': 'last',
        'bids[22].price': 'last', 'bids[22].amount': 'last',
        'bids[23].price': 'last', 'bids[23].amount': 'last',
        'bids[24].price': 'last', 'bids[24].amount': 'last'
        })

    # 处理多级列索引，保留最后一级
    agged_perpe.columns = [col[-1] if isinstance(col, tuple) else col for col in agged_perpe.columns]

    agged_perpe.bfill(inplace=True)
    agged_perpe.ffill(inplace=True)
    
    # 去除缺失值
    agged_perpe.dropna(inplace=True)
    
    # 重置索引
    agged_perpe.reset_index(inplace=True)
    return agged_perpe