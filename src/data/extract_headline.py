import collections
import multiprocessing as mp
import os
import pickle
import re
from string import punctuation
from typing import Any, List, Tuple

from src.data.sp500 import hist_sp500
from src.data.utils import STOPWORDS

# test path
file = 'us-aig-idUSTRE62E0GQ20100315'
folder = '/home/timnaka123/Documents/financial-news-dataset/ReutersNews106521/20100315'
path = os.path.join(folder, file)
tickers=hist_sp500['2013/01/31']


def reuters_single_file_process(
    path: str,
    tickers: List[str],
    rm_punctuation: bool,
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

    punc_pattern = '([' + ''.join(punctuation) + '])'
    if rm_punctuation:
        headline = re.sub(r'U\.S\.', 'us', headline)
        headline = re.sub(punc_pattern, ' ', headline)
    else:
        # rm '-' as it always appears in the beginning
        headline = re.sub('^-- ', r'', headline)
        # rm line break in the end
        headline = re.sub('\n$', r'', headline)
        # add space betwween punc
        headline = re.sub(punc_pattern, r' \1 ', headline)
        headline = re.sub(r' +', ' ', headline)

    return matched, headline