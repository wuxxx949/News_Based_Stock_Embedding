import os
import re
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from src.data.utils import get_path, load_pickled_obj


def date_formatter(date: str) -> str:
    year = date[:4]
    month = date[4:6]
    day = date[6:]

    return f'{year}-{month}-{day}'


def get_news_date_range(data_dir_path: str) -> Tuple[str, str]:
    all_pathes = get_path(data_dir_path)
    result = []
    for path in all_pathes:
        date = [e for e in path.split('/')[1:] if len(re.sub(r'[0-9]', '', e)) == 0][0]
        result.append(date)

    min_date, max_date = min(result), max(result)

    return date_formatter(min_date), date_formatter(max_date)

# TODO: to utils
def get_tickers(dir_path: str, obj_name: str, min_count: int) -> List[str]:
    """fetch pickled tickers that mentioned enough times in the news

    Args:
        dir_path (str): pickle obj dir path
        obj_name (str): pickle obj name
        min_count (int): min number of mentioning

    Returns:
        List[str]: tickers of interest
    """
    ticker_freq = load_pickled_obj(dir_path, obj_name)
    qualified = {k: v for k, v in ticker_freq.items() if v >= min_count}

    return list(qualified.keys())

# TODO: to utils
def sleep_time(loc: float, scale: float):
    out = 0
    while out <= 1:
        out = np.random.normal(loc, scale)

    return out

def fetch_daily_price(tickers: List[str],
                      start: str,
                      end: str,
                      batch_size: int=20) -> pd.DataFrame:
    if len(tickers) % batch_size == 1:
        batch_size += 1

    df_lst = []
    n = -(-len(tickers) // batch_size)
    for i in range(n):
        # print(tickers[i * batch_size: i * batch_size + batch_size])
        ticker_str = ' '.join(tickers[i * batch_size: i * batch_size + batch_size])
        df = yf.download(ticker_str, start=start, end=end)
        df = df.loc[:, df.columns.get_level_values(0) == 'Adj Close']
        df.columns = df.columns.droplevel(0)
        df_lst.append(df.dropna(how='all').reset_index().melt(id_vars='Date', var_name='ticker'))
        time.sleep(sleep_time(3, 1))

    price_df = pd.concat(df_lst).reset_index(drop=True)
    price_df.columns = ['date', 'ticker', 'adj_price']

    return price_df


def daily_return_calc(price_df: pd.DataFrame) -> pd.DataFrame:
    """generate daily return from adjusted price

    Args:
        price_df (pd.DataFrame): daily adjusted price
    Returns:
        pd.DataFrame: daily return with column
    """
    price_df['shifted_price'] = price_df \
        .sort_values(['ticker', 'date']) \
        .groupby('ticker')['adj_price'] \
        .shift()

    price_df['daily_return'] = np.log(price_df['adj_price']) - np.log(price_df['shifted_price'])

    return price_df.dropna(subset=['daily_return'])


def create_target(
    return_df: pd.DataFrame,
    lower_cut: float = -0.0059,
    upper_cut: float = 0.0068
    ) -> pd.DataFrame:
    """generate df with target for DL

    Args:
        return_df (pd.DataFrame): target tickers log return df
        lower_cut (float, optional): threshold for negative. Defaults to -0.0059.
        upper_cut (float, optional): threshold for positive. Defaults to 0.0068.

    Returns:
        pd.DataFrame:
    """

    def _get_target(log_return: float):
        if log_return <= lower_cut: return 0
        if log_return >= upper_cut: return 1
        if lower_cut < log_return < upper_cut: return -1

    output_df = return_df \
        .assign(target = lambda x: x['daily_return'].apply(_get_target)) \
        .query('target >= 0') \
        .loc[:, ['date', 'ticker', 'target']] \
        .reset_index(drop=True)

    return output_df


def main(
    news_dir_path: str,
    processed_data_dir_path: str,
    ticker_obj_name: str,
    min_count: int,
    ) -> pd.DataFrame:
    """generate training target

    Args:
        news_dir_path (str): _description_
        processed_data_dir_path (str): _description_
        ticker_obj_name (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    min_date, max_date = get_news_date_range(data_dir_path=news_dir_path)

    tickers = get_tickers(
        dir_path=processed_data_dir_path,
        obj_name=ticker_obj_name,
        min_count=min_count
        )

    raw_daily_price = fetch_daily_price(
        tickers=tickers,
        start=min_date,
        end=max_date,
        batch_size=10
        )

    return_data = daily_return_calc(raw_daily_price)

    target = create_target(return_data)

    target_save_path = os.path.join(processed_data_dir_path, 'target_df.parquet.gzip')
    target.to_parquet(target_save_path, compression='gzip')

    return target


if __name__ == '__main__':
    output = main(
        news_dir_path= '/home/timnaka123/Documents/financial-news-dataset/ReutersNews106521',
        processed_data_dir_path='/home/timnaka123/Documents/stock_embedding_nlp/src/data/',
        ticker_obj_name='qualified_tickers.pickle',
        min_count=150
        )

    min_date, max_date = get_news_date_range(
        '/home/timnaka123/Documents/financial-news-dataset/ReutersNews106521'
        )

    tickers = get_tickers(
        dir_path='/home/timnaka123/Documents/stock_embedding_nlp/src/data/',
        obj_name='qualified_tickers.pickle'
        )

    raw_daily_price = fetch_daily_price(
        tickers=tickers,
        start=min_date,
        end=max_date,
        batch_size=10
        )

    return_data = daily_return_calc(raw_daily_price)
