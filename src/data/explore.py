import collections
import multiprocessing as mp
import os
import pickle
import re
from string import punctuation
from typing import Any, List, Tuple

import fasttext

from src.data.sp500 import hist_sp500
from src.data.utils import STOPWORDS

# test path
file = 'us-aig-idUSTRE62E0GQ20100315'
folder = '/home/timnaka123/Documents/financial-news-dataset/ReutersNews106521/20100315'
path = os.path.join(folder, file)

def single_file_process(
        path: str,
        tickers: List[str],
        rm_punctuation: bool,
        rm_number: bool
        ) -> Tuple[List[str], str]:
    """find all matched ticker in a news

    Args:
        path (str): path of a news file

    Returns:
        List[str]: ticker(s) mentioned in the news
    """
    try:
        with open(path, 'r', encoding="utf8") as f:
            lines = f.readlines()
    except Exception:
        print(f'fail to load file: {path}')
        return [], '', ''

    # remove meta info and line breakers
    del lines[1:7]


    # remove reporter info
    try:
        # match last right parenthesis but no ) in between
        # [^)]* match any thing but not ')', [^(]* also works
        lines[-1] = re.sub(r'\([^)]*\)$', '', lines[-1])
    except Exception:
        print(f'empty file: {path}')
        return [], '', ''

    try:
        # remove Reuters and location
        lines[1] = re.sub(r'^.*\(Reuters\)', '', lines[1])
    except Exception:
        pass

    news_text = ' '.join(lines)
    pattern = '.N | '.join(tickers) + '.N '
    regexp = re.compile(pattern)
    matched = set(regexp.findall(news_text))

    if len(matched) > 0:
        matched = [re.sub(r'\.N| ', '', e) for e in matched]
    else:
        matched = []

    # retain N in ticker but remove '.' so we can train ticker embedding
    training_text = re.sub(r'\.N', 'N', news_text)
    training_text = re.sub(r'U\.S\.', 'us', training_text)

    # make each sentence in a newline
    pattern = r"(?<=\. )(?=[A-Z1-9])"
    training_text = re.sub(pattern, r'\n', training_text)

    pattern = '([' + ''.join(punctuation) + '])'
    if rm_punctuation:
        training_text = re.sub(pattern, ' ', training_text)
    else:
        training_text = re.sub(pattern, r' \1 ', training_text)
         # rm '-' as it always appears in the beginning
        training_text = re.sub('-', r'', training_text)


    if rm_number:
        pattern = '[0-9]'
        training_text = re.sub(pattern, ' ', training_text)

    # remove non-alphabetical and stop words for embedding training
    # training_text = re.sub(r'[^a-zA-Z ]', ' ', training_text)
    # keep at most one space
    training_text = re.sub(r' +', ' ', training_text)
    # remove stop words
    words = training_text.split(' ')
    training_text = ' '.join([w.lower() for w in words if w.lower() not in STOPWORDS])
    training_text = training_text.lstrip()

    return matched, news_text, training_text


def get_path(dir_path: str):
    """get all news article file paths
    """
    path_lst = []
    subdirs = os.listdir(dir_path)
    for subdir in subdirs:
        # avoid hidden file
        if subdir.startswith('.'):
            continue
        news_paths = os.listdir(os.path.join(dir_path, subdir))
        for path in news_paths:
            path_lst.append(os.path.join(dir_path, subdir, path))

    return path_lst


def worker(path, tickers, rm_punctuation, rm_number, q1, q2):
    """single file
    """
    matched, text, training_text = single_file_process(path, tickers, rm_punctuation, rm_number)
    q1.put(text)
    q2.put(training_text)

    return matched


def text_listener(q, fp):
    """listens for messages on the q, writes to file
    """
    with open(fp, 'w', encoding='utf8') as f:
        while True:
            text = q.get()
            if text == 'kill':
                # f1.write('killed')
                break
            f.write(str(text) + '\n')
            f.flush()

def training_listener(q, fp):
    """listens for messages on the q, writes to file
    """
    with open(fp, 'w', encoding='utf8') as f:
        while True:
            text = q.get()
            if text == 'kill':
                break
            f.write(str(text))
            f.flush()

# TODO: to utils
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

def news_preprocessing(data_dir_path: str,
                       save_dir_path: str,
                       tickers: List[str],
                       rm_punctuation: bool = False,
                       rm_number: bool = True,
                       min_count: int = 100) -> List[List[str]]:
    """apply mp to process all files
    """
    all_files = get_path(data_dir_path)

    manager = mp.Manager()
    q1 = manager.Queue()
    q2 = manager.Queue()
    pool = mp.Pool(mp.cpu_count() + 2)
    write_file=os.path.join(save_dir_path, 'news_corpus.txt')
    write_training_file=os.path.join(save_dir_path, 'news_training_corpus.txt')
    pool.apply_async(text_listener, (q1, write_file))
    pool.apply_async(training_listener, (q2, write_training_file))

    # fire up workers
    jobs = []
    results = []
    for f in all_files:
        job = pool.apply_async(worker, (f, tickers, rm_punctuation, rm_number, q1, q2))
        jobs.append(job)

    for job in jobs:
        results.append(job.get())

    # now we are done, kill the listener
    q1.put('kill')
    q2.put('kill')
    pool.close()
    pool.join()

    all_tickers = [item for sublist in results for item in sublist]
    ticker_freq = collections.Counter(all_tickers)
    qualified = {k: v for k, v in ticker_freq.items() if v > min_count}
    pickle_results(dir=save_dir_path, name='ticker_lst.pickle', obj=results)
    pickle_results(dir=save_dir_path, name='qualified_tickers.pickle', obj=qualified)

    return results, qualified


def train_word_embedding(file: str, dim: int=64, **kargs):
    # https://fasttext.cc/docs/en/python-module.html#train_unsupervised-parameters
    model = fasttext.train_unsupervised(file, dim=dim, **kargs)
    # model.get_word_vector
    dir = os.path.dirname(os.path.realpath(__file__))
    model.save_model(os.path.join(dir, 'news.bin'))

    return model


if __name__ == "__main__":

    # a, b, c = single_file_process(path, hist_sp500['2013/01/31'], False, True)


    mentioned_tickers, qualified_tickers = news_preprocessing(
        data_dir_path='/home/timnaka123/Documents/financial-news-dataset/ReutersNews106521',
        save_dir_path='/home/timnaka123/Documents/stock_embedding_nlp/src/data/',
        tickers=hist_sp500['2013/01/31']
    )

    embedding_model = train_word_embedding(
        file='/home/timnaka123/Documents/stock_embedding_nlp/src/data/news_training_corpus.txt',
        epoch=7
        )

