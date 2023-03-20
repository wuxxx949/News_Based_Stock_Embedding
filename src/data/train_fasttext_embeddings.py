import collections
import multiprocessing as mp
import os
from typing import List, Tuple

import fasttext
import numpy as np

from src.data.extract_headline import single_file_process
from src.data.sp500 import hist_sp500
from src.data.utils import pickle_results


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


def get_news_path(
    reuters_dir: str,
    bloomberg_dir: str,
    seed: int = 42
    ) -> List[Tuple[str, str]]:
    rpaths = get_path(reuters_dir)
    bpaths = get_path(bloomberg_dir)

    rtuples = list(zip(rpaths, ['r'] * len(rpaths)))
    btuples = list(zip(bpaths, ['b'] * len(bpaths)))

    all_tuples = rtuples + btuples
    np.random.default_rng(seed).shuffle(all_tuples)

    return all_tuples


def worker(news_type, path, tickers, rm_punctuation, q):
    """single file
    """
    matched, text = single_file_process(
        news_type=news_type,
        path=path,
        tickers=tickers,
        rm_punctuation=rm_punctuation
        )
    if len(text) > 0:
        q.put(text)

    return matched


def text_listener(q, fp):
    """listens for messages on the q, writes to file
    """
    with open(fp, 'w', encoding='utf8') as f:
        while True:
            text = q.get()
            if text == 'kill':
                break
            f.write(str(text) + '\n')
            f.flush()


def news_preprocessing(
    reuters_data_dir_path: str,
    bloomberg_data_dir_path: str,
    save_dir_path: str,
    tickers: List[str],
    rm_punctuation: bool = False,
    min_count: int = 200
    ) -> List[List[str]]:
    """apply mp to process all files
    """
    all_files = get_news_path(
        reuters_dir=reuters_data_dir_path,
        bloomberg_dir=bloomberg_data_dir_path
        )

    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(mp.cpu_count() + 2)
    write_file=os.path.join(save_dir_path, 'headlines.txt')
    pool.apply_async(text_listener, (q, write_file))

    # fire up workers
    jobs = []
    results = []
    for f in all_files:
        news_type = f[1]
        news_path = f[0]
        job = pool.apply_async(worker, (news_type, news_path, tickers, rm_punctuation, q))
        jobs.append(job)

    for job in jobs:
        results.append(job.get())

    # now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()

    all_tickers = [item for sublist in results for item in sublist]
    ticker_freq = collections.Counter(all_tickers)
    qualified = {k: v for k, v in ticker_freq.items() if v >= min_count}
    pickle_results(dir=save_dir_path, name='ticker_freq.pickle', obj=ticker_freq)
    pickle_results(dir=save_dir_path, name='qualified_tickers.pickle', obj=qualified)

    return results, qualified


def train_word_embedding(training_file: str, dim: int=64, **kargs) -> None:
    """train for word embeddings using fasttext

    Args:
        training_file (str): training text
        dim (int, optional): dimension of the embeddings. Defaults to 64.
    """
    # https://fasttext.cc/docs/en/python-module.html#train_unsupervised-parameters
    model = fasttext.train_unsupervised(training_file, dim=dim, **kargs)
    # model.get_word_vector
    dir = os.path.dirname(os.path.realpath(__file__))
    model.save_model(os.path.join(dir, 'news.bin'))


if __name__ == '__main__':
    rdir = '/home/timnaka123/Documents/financial-news-dataset/ReutersNews106521'
    bdir = '/home/timnaka123/Documents/financial-news-dataset/bloomberg'

    out = news_preprocessing(
        reuters_data_dir_path=rdir,
        bloomberg_data_dir_path=bdir,
        save_dir_path='/home/timnaka123/Documents/stock_embedding_nlp/src/data/',
        tickers=hist_sp500['2013/01/31']
    )

    train_word_embedding(
        training_file='/home/timnaka123/Documents/stock_embedding_nlp/src/data/headlines.txt',
        epoch=7
    )