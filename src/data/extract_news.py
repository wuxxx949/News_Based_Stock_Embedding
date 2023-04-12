import os
import re
from string import punctuation
from typing import List, Tuple

from src.data.sp500 import hist_sp500


punc_pattern = '([' + ''.join(punctuation) + '])'

def process_punc(text: str, rm_punc: bool) -> str:
    # make each sentence in a newline
    pattern = r"(?<=\. )(?=[A-Z1-9])"
    text = re.sub(pattern, r'\n', text)

    if rm_punc:
        text = re.sub(r'U\.S\.', 'us', text)
        text = re.sub(r'U\.K\.', 'uk', text)
        text = re.sub(punc_pattern, ' ', text)
    else:
        # rm '--' as it always appears in the beginning
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
    """find all matched ticker in a Reuters news and fetch headline

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
        return [], ''

    # remove meta info and line breakers
    del lines[1:7]

    # remove reporter info
    try:
        # match last right parenthesis but no ) in between
        # [^)]* match any thing but not ')', [^(]* also works
        lines[-1] = re.sub(r'\([^)]*\)$', '', lines[-1])
    except Exception:
        print(f'empty file: {path}')
        return [], ''

    try:
        # remove Reuters and location
        lines[1] = re.sub(r'^.*\(Reuters\)', '', lines[1])
    except Exception:
        pass

    # fetch ticker mentioned, ( AAPL.N ) -> AAPL
    news_text = ' '.join(lines)
    pattern = '.N | '.join(tickers) + '.N '
    regexp = re.compile(pattern)
    matched = set(regexp.findall(news_text))

    if len(matched) > 0:
        matched = [re.sub(r'\.N| ', '', e) for e in matched]
        # normalized ticker
        # (\w+): apturing group with one or more letters
        pattern = r"(\s?\(\s(\w+)\.N\s\)\s?)"
        # backreference \2 to replace the matched pattern with the contents of the second capturing group.
        news_text = re.sub(pattern, r' \2 ', news_text)
    else:
        matched = []

    news_text = process_punc(text=news_text, rm_punc=rm_punctuation)

    return matched, news_text


def bloomberg_single_file_process(
    path: str,
    tickers: List[str],
    rm_punctuation: bool
    ) -> Tuple[List[str], str]:
    """find all matched ticker in bloomberg news and fetch headline

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
        return [], ''

    headline = lines[0]
    if len(re.sub(r'[^a-zA-Z]', '', headline)) == 0:
        headline = lines[1]

    # bloomberg uses different quotation mark
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


def single_file_process(
    news_type: str,
    path: str,
    tickers: List[str],
    rm_punctuation: bool
    ) -> Tuple[List[str], str]:
    try:
        if news_type == 'r':
            matched, headline = reuters_single_file_process(path, tickers, rm_punctuation)
        else:
            matched, headline = bloomberg_single_file_process(path, tickers, rm_punctuation)
    except Exception as e:
        print(f'{path}: {e}')
        matched, headline = [], ''

    return matched, headline


if __name__ == '__main__':
    # test reuters path
    file = 'us-aig-idUSTRE62E0GQ20100315'
    folder = '/home/timnaka123/Documents/financial-news-dataset/ReutersNews106521/20100315'
    path = os.path.join(folder, file)
    tickers=hist_sp500['2013/01/31']
    print(reuters_single_file_process(path ,tickers, False))

    # test bloomberg path
    bb_folder = '/home/timnaka123/Documents/financial-news-dataset/bloomberg/2011-01-26'
    file = 'federal-open-market-committee-s-statement-on-monetary-policy-full-tex'
    path = os.path.join(bb_folder, file)
    print(bloomberg_single_file_process(path, tickers, False))