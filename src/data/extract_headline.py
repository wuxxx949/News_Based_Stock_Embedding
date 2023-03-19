import collections
import multiprocessing as mp
import os
import pickle
import re
from string import punctuation
from typing import Any, List, Tuple

from src.data.sp500 import hist_sp500
from src.data.utils import STOPWORDS


punc_pattern = '([' + ''.join(punctuation) + '])'

def process_punc(text: str, rm_punc: bool) -> str:
    if rm_punc:
        text = re.sub(r'U\.S\.', 'us', text)
        text = re.sub(r'U\.K\.', 'uk', text)
        text = re.sub(punc_pattern, ' ', text)
    else:
        # rm '-' as it always appears in the beginning
        text = re.sub('^-- ', r'', text)
        # rm line break in the end
        text = re.sub('\n$', r'', text)
        # add space betwween punc
        text = re.sub(punc_pattern, r' \1 ', text)
        # at most one space
        text = re.sub(r' +', ' ', text)

    return text

def reuters_single_file_process(
    path: str,
    tickers: List[str],
    rm_punctuation: bool
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
    headline = lines[0]

    news_text = ' '.join(lines)
    pattern = '.N | '.join(tickers) + '.N '
    regexp = re.compile(pattern)
    matched = set(regexp.findall(news_text))

    if len(matched) > 0:
        matched = [re.sub(r'\.N| ', '', e) for e in matched]
    else:
        matched = []

    headline = process_punc(text=headline, rm_punc=rm_punctuation)

    return matched, headline

def bloomberg_single_file_process(
    path: str,
    tickers: List[str],
    rm_punctuation: bool
    ) -> Tuple[List[str], str]:
    try:
        with open(path, 'r', encoding="utf8") as f:
            lines = f.readlines()
    except Exception:
        print(f'fail to load file: {path}')
        return [], '', ''

    headline = lines[0]
    headline = re.sub('‘|’|“|”', "'", headline)

    news_text = ' '.join(lines)
    pattern = '\(' + '\)|\('.join(tickers) + '\)'
    regexp = re.compile(pattern)
    matched = set(regexp.findall(news_text))
    if len(matched) > 0:
        matched =  [re.search(r'\((.*?)\)',s).group(1) for s in matched]
    else:
        matched = []

    headline = process_punc(text=headline, rm_punc=rm_punctuation)

    return matched, headline


if __name__ == '__main__':
    # test reuters path
    file = 'us-aig-idUSTRE62E0GQ20100315'
    folder = '/home/timnaka123/Documents/financial-news-dataset/ReutersNews106521/20100315'
    path = os.path.join(folder, file)
    tickers=hist_sp500['2013/01/31']
    print(reuters_single_file_process(path ,tickers, False))

    # test bloomberg path
    bb_folder = '/home/timnaka123/Documents/financial-news-dataset/bloomberg/2013-11-20'
    file = 'apple-says-smartphone-inventor-patent-claim-falls-shor'
    path = os.path.join(bb_folder, file)
    print(bloomberg_single_file_process(path, tickers, False))