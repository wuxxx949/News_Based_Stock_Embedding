import hashlib
import os
import pickle
import random
import re
import uuid
import datetime
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal


STOPWORDS = [
    'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once',
    'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for',
    'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is',
    's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until',
    'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were',
    'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above',
    'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before',
    'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves',
    'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now',
    'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
    'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my',
    'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'
    ]

def pickle_results(dir: str, name: str, obj: Any) -> None:
    """pickle an object
    """
    dbfile = open(os.path.join(dir, name), 'wb')
    pickle.dump(obj, dbfile)
    dbfile.close()

def load_pickled_obj(dir: str, name: str):
    file = open(os.path.join(dir, name), 'rb')
    data = pickle.load(file)
    # close the file
    file.close()

    return data

def get_path(dir_path: str) -> List[str]:
    """get all news article file paths

    Args:
        dir_path (str): directory contains all news

    Returns:
        List[str]: news file paths
    """
    path_lst = []
    subdirs = os.listdir(dir_path)
    for subdir in subdirs:
        # avoid hidden file
        if subdir.startswith('.'):
            continue
        # assume two layer folder structure
        news_paths = os.listdir(os.path.join(dir_path, subdir))
        for path in news_paths:
            path_lst.append(os.path.join(dir_path, subdir, path))

    return path_lst


def sample_news(dir_path: str) -> str:
    """sample one news path from a folder
    """
    all_path = get_path(dir_path=dir_path)

    return random.choice(all_path)


def get_raw_news(news_path: str) -> str:
    """fetch the raw news text
    """
    with open(news_path, 'r', encoding="utf8") as f:
        lines = f.readlines()

    return ' '.join(lines)


def get_news_path(
    reuters_dir: str,
    bloomberg_dir: str,
    seed: int = 42
    ) -> List[Tuple[str, str]]:
    """fetch path and randomize order for all news sources

    Args:
        reuters_dir (str): Reuters news directory
        bloomberg_dir (str): Bloomberg news directory
        seed (int, optional): random number seed. Defaults to 42.

    Returns:
        List[Tuple[str, str]]: list of news file path and type
    """
    rpaths = get_path(reuters_dir)
    bpaths = get_path(bloomberg_dir)

    rtuples = list(zip(rpaths, ['r'] * len(rpaths)))
    btuples = list(zip(bpaths, ['b'] * len(bpaths)))

    all_tuples = rtuples + btuples
    np.random.default_rng(seed).shuffle(all_tuples)

    return all_tuples


def longest_line(fpath: str) -> int:
    """find the max number of tokens in a line

    Args:
        fpath (str): file path

    Returns:
        int: max number of tokens
    """
    with open(fpath, 'r', encoding="utf8") as f:
        lines = f.readlines()

    return max([len(e.split(' ')) for e in lines])

def headline_processing(headline: str):
    headline = re.sub('^-- ', r'', headline)
    headline = re.sub('\n$', r'', headline)
    headline = re.sub(r' +', ' ', headline)

    return headline

def create_uuid_from_string(val: str) -> str:
    hex_string = hashlib.md5(val.encode("UTF-8")).hexdigest()
    return str(uuid.UUID(hex=hex_string))


def get_target_tickers(dir: str, name: str) -> List[str]:
    """fetch tickers of interested based on return data

    Args:
        dir (str): file directory
        name (str): name of parquet file

    Returns:
        List[str]: list of tickers
    """
    return_df = pd.read_parquet(os.path.join(dir, name))

    return list(return_df['ticker'].unique())


def sleep_time(loc: float, scale: float):
    out = 0
    while out <= 1:
        out = np.random.normal(loc, scale)

    return out


def get_nearest_trading_date(date: str) -> Tuple[str, str]:
    """find the closet non-past US trading date

    Args:
        date (str): input date

    Returns:
        Tuple[str, str]: nearest non-past trading date and next trading date
    """
    # assume US exchange
    date_dt = datetime.datetime.strptime(date, '%Y-%m-%d')
    end_date = date_dt + datetime.timedelta(days=10)
    end_date_str = datetime.datetime.strftime(end_date, '%Y-%m-%d')
    nyse = mcal.get_calendar('NYSE')
    date_range = nyse.schedule(start_date=date, end_date=end_date_str)
    td = mcal.date_range(date_range, frequency='1D')[:2]
    td_str = [str(e .to_pydatetime().date()) for e in td]

    return td_str[0], td_str[1]


def find_next_traindg_date(date: str) -> str:
    """find the trading date w.r.t. US trading calender

    Args:
        date (str): date

    Returns:
        str: next trading date
    """
    _, next_td = get_nearest_trading_date(date)

    return next_td


def shift_to_future_one_year(date: str) -> str:
    """shift 1 year to the future

    Args:
        date (str): date

    Returns:
        str: date shfited 1 year to future
    """
    date_dt = datetime.datetime.strptime(date, '%Y-%m-%d')
    date_p1y = date_dt.replace(year=date_dt.year + 1)

    return str(date_p1y.date())


def text_preprocessing(text: str):
    """preprocessing text

    Args:
        text (str): news headline
    """
    # rm non-alphabetical
    s = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    # at most one space
    s = re.sub(r' +', ' ', s)
    # rm single letter
    s = ' '.join([e for e in s.split(' ') if len(e) > 1 and e not in STOPWORDS])

    return s
