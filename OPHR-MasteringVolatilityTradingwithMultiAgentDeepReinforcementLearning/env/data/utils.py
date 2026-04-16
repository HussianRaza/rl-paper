import re
import pandas as pd

def load_csv_data(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    elif file_path.endswith('.gz'):
        return pd.read_csv(file_path, compression='gzip')


def extract_date_from_filename(filename):
    """从给定的文件名中提取日期"""
    match = re.search(r'\d{4}-\d{2}-\d{2}', filename)
    return match.group(0) if match else None
