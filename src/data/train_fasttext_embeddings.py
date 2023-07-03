
import collections
import multiprocessing as mp
import os

import fasttext

from src.data.extract_news import single_file_process
from src.data.utils import get_news_path, pickle_results
from src.meta_data import get_meta_data


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


def news_preprocessing(rm_punctuation: bool = False) -> None:
    """apply mp to process all files

    Args:
        tickers (List[str]): target tickers
        rm_punctuation (bool, optional): if remove punctuation in the headline. Defaults to False.
    """
    save_dir_path = get_meta_data()['SAVE_DIR']
    all_files = get_news_path(
        reuters_dir=get_meta_data()['REUTERS_DIR'],
        bloomberg_dir=get_meta_data()['BLOOMBERG_DIR']
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


def train_word_embedding(dim: int=64, **kargs) -> None:
    """train for word embeddings using fasttext

    Args:
        dim (int, optional): dimension of the embeddings. Defaults to 64.
    """
    # https://fasttext.cc/docs/en/python-module.html#train_unsupervised-parameters
    training_file = os.path.join(get_meta_data()['SAVE_DIR'], 'news_corpus.txt')
    model = fasttext.train_unsupervised(training_file, dim=dim, **kargs)
    # model.get_word_vector
    model.save_model(os.path.join(get_meta_data()['SAVE_DIR'], 'news.bin'))


if __name__ == '__main__':
    news_preprocessing()
    train_word_embedding(epoch=7)