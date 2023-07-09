import os
import re
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from src.data.sp500 import hist_sp500
from src.data.utils import (get_nearest_trading_date, get_path,
                            load_pickled_obj, sleep_time)
from src.logger import setup_logger
from src.meta_data import get_meta_data

logger = setup_logger('data', 'data.log')


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
def get_tickers(dir_path: str, obj_name: str) -> List[str]:
    """fetch pickled tickers that mentioned enough times in the news

    Args:
        dir_path (str): pickle obj dir path
        obj_name (str): pickle obj name

    Returns:
        List[str]: tickers of interest
    """
    ticker_freq = load_pickled_obj(dir_path, obj_name)
    # qualified = {k: v for k, v in ticker_freq.items() if v >= min_count}
    ticker_freq = dict(sorted(ticker_freq.items(), key=lambda item: item[1], reverse=True))
    tickers = hist_sp500['2013/01/31']
    qualified =  {k: v for k, v in ticker_freq.items() if k in tickers}

    return list(qualified.keys())


def fetch_daily_price(
    tickers: List[str],
    start: str,
    end: str,
    batch_size: int=20
    ) -> pd.DataFrame:
    """fetch daily price for target tickers

    Args:
        tickers (List[str]): target tickers
        start (str): start date
        end (str): end date_
        batch_size (int, optional): batch size for yf. Defaults to 20.

    Returns:
        pd.DataFrame: stock data
    """
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

    # logging
    ticker_cnt = len(output_df['ticker'].unique())
    min_date = output_df['date'].min()
    max_date = output_df['date'].max()
    logger.info(f'{ticker_cnt} tickers')
    logger.info(f'{len(output_df)} return data from {min_date} to {max_date}')

    return output_df


def stock_annual_return_calc(
    tickers: List[str],
    min_year: int,
    max_year: int,
    mmdd: str = '01-01'
    ) -> pd.DataFrame:
    """calculate historical annual return

    Args:
        tickers (List[str]): input tickers
        min_year (int): min year to include
        max_year (int): max year to include
        mmdd (str): month and day

    Returns:
        pd.DataFrame: df with column of ticker and annual_return
    """
    ticker_str = ' '.join(tickers)
    df_lst = []
    for t in range(min_year, max_year + 1):
        d = f'{t}-{mmdd}'
        d1, d2 = get_nearest_trading_date(d)
        df = yf.download(ticker_str, start=d1, end=d2)
        df = df.loc[:, df.columns.get_level_values(0) == 'Adj Close']
        df.columns = df.columns.droplevel(0)
        df_lst.append(df.dropna(how='all').reset_index().melt(id_vars='Date', var_name='ticker'))

    price_df = pd.concat(df_lst).dropna(how='any').reset_index(drop=True)
    price_df['shifted_value'] = price_df \
        .sort_values(['ticker', 'Date']) \
        .groupby('ticker')['value'] \
        .shift()

    price_df['annual_return'] = (price_df['value'] - price_df['shifted_value']) / price_df['shifted_value']
    avg_return = price_df.dropna(how='any').groupby('ticker').agg({'annual_return': np.mean}).reset_index()

    return avg_return


def main(
    news_dir_path: str,
    processed_data_dir_path: str,
    ticker_obj_name: str,
    nstock: int,
    ) -> pd.DataFrame:
    """generate training target

    Args:
        news_dir_path (str): news directory
        processed_data_dir_path (str): _description_
        ticker_obj_name (str): _description_
        nstock (int): number of stocks to fetch

    Returns:
        pd.DataFrame: _description_
    """
    min_date, max_date = get_news_date_range(data_dir_path=news_dir_path)

    tickers = get_tickers(
        dir_path=processed_data_dir_path,
        obj_name=ticker_obj_name,
        )[: nstock]

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
        nstock=60
        )

