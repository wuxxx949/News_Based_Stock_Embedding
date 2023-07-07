import re
from string import punctuation
from typing import List, Tuple

from src.data.utils import text_preprocessing


def process_punc(text: str, rm_punc: bool) -> str:
    # make each sentence in a newline
    # pattern = r"(?<=\. )(?=[A-Z1-9])"
    # text = re.sub(pattern, r'\n', text)
    punc_pattern = '([' + ''.join(punctuation) + '])'
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
        # at most one \n and remove white space after \n
        text = re.sub(r'(\n\s*)+', '\n', text)

    return text


def break_sentence(text: str) -> str:
    """Add '\n' at the end of each sentence

    Args:
        text (str): input text

    Returns:
        str: output text with line breaker
    """
    # looks for periods, question marks or exclamation marks
    # followed by a space and a capital letter
    sentences = re.split(r'(?<=[^A-Z]{1}[.?!]) +(?=[A-Z])', text)

    return"\n".join(sentences)


def reuters_single_file_process(
    path: str,
    rm_punctuation: bool
    ) -> Tuple[List[str], str]:
    """find all matched ticker in a Reuters news and fetch headline

    Args:
        path (str): path of a single Reuters news file

    Returns:
        Tuple[List[str], str]: ticker(s) mentioned in the news and
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
        lines[1] = re.sub(r'^.*\(Reuters\)\s+-\s+', '', lines[1])
    except Exception:
        pass

    # fetch ticker mentioned, ( AAPL.N ) -> AAPL, ( BRKa.N ) -> BRKa
    news_text = ' '.join(lines)
    # remove random \n
    news_text = re.sub('\n', '', news_text)

    matched = set(re.findall(r"\s[A-Z]+[a-z]?\.[N|O|OQ]\s", news_text))

    if len(matched) > 0:
        matched = [re.sub(r'\.[N|O|OQ]| ', '', e).upper() for e in matched]
        # normalize ticker
        pattern = r"(\s?\(\s([A-Z]+[a-z]?)\.[N|O|OQ]\s\)\s?)"
        # backreference \2 to replace the matched pattern with the contents of the second capturing group.
        news_text = re.sub(pattern, lambda x: ' ' + x.group(2).upper() + ' ', news_text)
    else:
        matched = []

    # news_text = break_sentence(news_text)

    # news_text = process_punc(text=news_text, rm_punc=rm_punctuation)
    news_text = text_preprocessing(text=news_text)

    return matched, news_text


def bloomberg_single_file_process(
    path: str,
    rm_punctuation: bool
    ) -> Tuple[List[str], str]:
    """find all matched ticker in bloomberg news and fetch headline

    Args:
        path (str): path of a single Bloomberg news file

    Returns:
        List[str]: ticker(s) mentioned in the news
    """
    try:
        with open(path, 'r', encoding="utf8") as f:
            lines = f.readlines()
    except Exception:
        print(f'fail to load file: {path}')
        return [], ''

    # remove empty first line if no content
    if lines[0].strip() == '--':
        del lines[0]

    # remove -- in headline
    lines[0] = re.sub('^ *-- *', '', lines[0])

    for i in range(1, len(lines)):
        lines[i] = '' if lines[i][:2] == '--' else lines[i]

    news_text = ' '.join(lines)
    # remove -- from headline
    # remove timestamp
    news_text = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z', '', news_text)
    # remove random \n
    news_text = re.sub('\n', '', news_text)
    # bloomberg uses different quotation mark
    news_text = re.sub('‘|’|“|”', "'", news_text)

    # remove editor and reporter contact info
    pattern = r"To contact the(.*?)bloomberg\.net"
    # re.DOTALL flag to make it match newlines as well.
    news_text = re.sub(pattern , '', news_text, flags=re.DOTALL)
    # news_text = break_sentence(news_text)

    # find mentioned tickers
    # pattern = '\(' + '\)|\('.join(tickers) + '\)'
    pattern = r"\(([A-Z.]+)\)"
    matched = list(set(re.findall(pattern, news_text)))

    if len(matched) > 0:
        # normalize ticker
        news_text = re.sub(r'\s?\(([A-Z.]+)\)\s?', r' \1 ', news_text)
    else:
        matched = []

    # news_text = process_punc(text=news_text, rm_punc=rm_punctuation)
    news_text = text_preprocessing(text=news_text)

    return matched, news_text


def single_file_process(
    news_type: str,
    path: str,
    rm_punctuation: bool
    ) -> Tuple[List[str], str]:
    """process a single Reuters or Bloomberg news file

    Args:
        news_type (str): 'r' for Reuters and 'b' for Bloomberg
        path (str): path of a single news file
        rm_punctuation (bool): if remove punctuation

    Returns:
        Tuple[List[str], str]: ticker(s) mentioned in the news and news headline
    """
    try:
        if news_type == 'r':
            matched, headline = reuters_single_file_process(path, rm_punctuation)
        else:
            matched, headline = bloomberg_single_file_process(path, rm_punctuation)
    except Exception as e:
        print(f'{path}: {e}')
        matched, headline = [], ''

    return matched, headline


if __name__ == '__main__':
    from src.data.utils import get_raw_news, sample_news

    # test reuters path
    file = 'us-aig-idUSTRE62E0GQ20100315'
    folder = '/home/timnaka123/Documents/financial-news-dataset/ReutersNews106521/20100315'
    path = sample_news('/home/timnaka123/Documents/financial-news-dataset/ReutersNews106521/')
    # path = '/home/timnaka123/Documents/financial-news-dataset/ReutersNews106521/20111213/us-markets-stocks-idUSTRE7AO0B420111213'
    # path = os.path.join(folder, file)
    out = reuters_single_file_process(path, False)
    print(get_raw_news(path))
    print(out[0])
    print(out[1])

    # test bloomberg path
    bb_folder = '/home/timnaka123/Documents/financial-news-dataset/bloomberg/2011-01-26'
    file = 'federal-open-market-committee-s-statement-on-monetary-policy-full-tex'
    # bb_folder = '/home/timnaka123/Documents/financial-news-dataset/bloomberg/2011-09-26'
    # file = 'apple-cuts-ipad-supply-chain-orders-jpmorgan'
    path = sample_news('/home/timnaka123/Documents/financial-news-dataset/bloomberg/')
    out = bloomberg_single_file_process(path, False)
    print(get_raw_news(path))
    print(out[0])
    print(out[1])