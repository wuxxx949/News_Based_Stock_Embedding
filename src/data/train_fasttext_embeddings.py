
import collections
import multiprocessing as mp
import os
from multiprocessing import Queue
from typing import List

import fasttext

from src.data.extract_news import single_file_process
from src.data.utils import get_news_path, pickle_results


def worker(news_type, path, rm_punctuation, q):
    """single file
    """
    matched, text = single_file_process(
        news_type=news_type,
        path=path,
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
    rm_punctuation: bool = False
    ) -> None:
    """apply mp to process all files

    Args:
        reuters_data_dir_path (str): Reuters news dir
        bloomberg_data_dir_path (str): Bloomberg news dir
        save_dir_path (str): directory to save output
        tickers (List[str]): target tickers
        rm_punctuation (bool, optional): if remove punctuation in the headline. Defaults to False.
    """
    all_files = get_news_path(
        reuters_dir=reuters_data_dir_path,
        bloomberg_dir=bloomberg_data_dir_path
        )

    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(mp.cpu_count() + 2)
    write_file=os.path.join(save_dir_path, 'news_corpus.txt')
    pool.apply_async(text_listener, (q, write_file))

    # fire up workers
    jobs = []
    results = []
    for f in all_files:
        news_type = f[1]
        news_path = f[0]
        job = pool.apply_async(worker, (news_type, news_path, rm_punctuation, q))
        jobs.append(job)

    for job in jobs:
        results.append(job.get())

    # now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()

    all_tickers = [item for sublist in results for item in sublist]
    ticker_freq = collections.Counter(all_tickers)
    pickle_results(dir=save_dir_path, name='ticker_freq.pickle', obj=ticker_freq)


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

    news_preprocessing(
        reuters_data_dir_path=rdir,
        bloomberg_data_dir_path=bdir,
        save_dir_path='/home/timnaka123/Documents/stock_embedding_nlp/src/data/',
        )

    train_word_embedding(
        training_file='/home/timnaka123/Documents/stock_embedding_nlp/src/data/news_corpus.txt',
        epoch=7
        )