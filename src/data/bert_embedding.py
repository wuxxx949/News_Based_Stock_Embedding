import multiprocessing as mp
import os
import re
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn import decomposition
from transformers import BertTokenizer, TFBertModel

from src.data.explore import get_path, pickle_results
from src.data.sp500 import hist_sp500

# test path
file = 'us-aig-idUSTRE62E0GQ20100315'
folder = '/home/timnaka123/Documents/financial-news-dataset/ReutersNews106521/20100315'

folder = '/home/timnaka123/Documents/financial-news-dataset/ReutersNews106521/20130103'
file = 'us-usa-transocean-settlement-idUSBRE9020H720130103'
path = os.path.join(folder, file)
tickers = hist_sp500['2013/01/31']


def single_file_process(path: str) -> str:
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
        return None

    # remove meta info and line breakers
    del lines[1:7]

    # remove reporter info
    try:
        # match last right parenthesis but no ) in between
        # [^)]* match any thing but not ')', [^(]* also works
        lines[-1] = re.sub(r'\([^)]*\)$', '', lines[-1])
        lines[0] =  re.sub(r'-- ', '', lines[0])
    except Exception:
        print(f'empty file: {path}')
        return None

    try:
        # remove Reuters and location at beginning
        lines[1] = re.sub(r'^.*\(Reuters\) - ', '', lines[1])
    except Exception:
        pass

    news_text = ' '.join(lines)

    # remove tickers
    # pattern = f"\(( {'.N | '.join(tickers)+ '.N '})\)"
    pattern = f"\( [^)]*\.N \)"
    news_text = re.sub(pattern, '', news_text)

    # insert [SEP]
    # pattern = r"(?<=\. )(?=[A-Z1-9])"
    # news_text = re.sub(pattern, r'[SEP] ', news_text)
    # news_text = re.sub('\n', r' [SEP]', news_text)

    # keep at most one space
    news_text = re.sub(r'\s+', ' ', news_text)

    # news_text = ' '.join(news_text.split(' ')[:512])

    # # retain N in ticker but remove '.' so we can train ticker embedding
    # training_text = re.sub(r'\.N', 'N', news_text)
    # training_text = re.sub(r'U\.S\.', 'us', news_text)
    # # remove non-alphabetical and stop words for embedding training
    # training_text = re.sub(r'[^a-zA-Z ]', ' ', training_text)

    return news_text


def bert_processing(path: str) -> Tuple[str, str, str]:
    date = [e for e in path.split('/')[1:] if len(re.sub(r'[0-9]', '', e)) == 0][0]
    file_name = path.split('/')[-1]
    news_id = file_name.split('-')[-1]

    return date, news_id, single_file_process(path)


def embedding_batch_preprocessing(
        paths: List[str],
        ncores: Optional[int] = None,
        ) -> List[Tuple[str, str, str]]:
    """preprocess news text for Bert embedding

    Args:
        pathes (List[str]): pathes in the batch
        tickers (List[str]): all tickers of interest
        ncores (Optional[int], optional): number of workers for mp. Defaults to None.

    Returns:
        List[Tuple[str, str, str]]: date and processed news text
    """
    ncores = ncores if ncores is not None else mp.cpu_count()
    with mp.Pool(processes=ncores) as p:
        out = p.map(bert_processing, paths)

    return out


def bert_embedding(
        input: Tuple[str, str, str],
        tokenizer,
        model
        ) -> List[Tuple[str, np.array]]:
    """handle input from embedding_batch_preprocessing

    Args:
        inpt (Tuple[str, str, str]): date, news id, and processed text
        tickers (List[str]): _description_
        tokenizer (_type_): _description_
        model (_type_): _description_

    Returns:
        np.array: _description_
    """
    text = [e[2] for e in input if e[2] is not None]
    date = [e[0] for e in input if e[2] is not None]
    news_id = [e[1] for e in input if e[2] is not None]

    encoded_input = tokenizer(
        text,
        max_length=512,
        truncation=True,
        return_tensors='tf',
        padding=True)

    output = model(encoded_input)
    embedding = output.pooler_output

    return list(zip(date, news_id, embedding.numpy()))


def generate_embedding(
        all_pathes: List[str],
        tokenizer,
        model,
        save_dir_path: str,
        batch_size: int = 40,
        ) -> pd.DataFrame:
    results = []
    text_input = embedding_batch_preprocessing(all_pathes)

    nparts = -(-len(text_input) // batch_size)

    for i in range(nparts):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        print(f'train {start_idx} to {end_idx}')
        out = bert_embedding(
            input= text_input[start_idx: end_idx],
            tokenizer=tokenizer,
            model=model
            )
        results.extend(out)

    pickle_results(dir=save_dir_path, name='embedding_lst.pickle', obj=results)
    embedding_df = pd.DataFrame([e[2] for e in results])
    embedding_df['date'] =  pd.to_datetime([e[0] for e in results])
    embedding_df['news_id'] = [e[1] for e in results]
    embedding_df = embedding_df.set_index(['date', 'news_id'])
    embedding_df.columns = ['c' + str(e) for e in embedding_df.columns]

    embedding_save_path = os.path.join(save_dir_path, 'bert_embedding_df.parquet.gzip')
    embedding_df.to_parquet(embedding_save_path, compression='gzip')

    return embedding_df


def bert_compression(
        embedding: pd.DataFrame,
        save_dir_path: str,
        n_component: int = 256
        ) -> pd.DataFrame:
    """apply PCA to reduce dimension

    Args:
        embedding (pd.DataFrame): original bert embedding

    Returns:
        pd.DataFrame: PCA embedding
    """
    pca = decomposition.PCA(n_components=n_component)
    pca_array =  pca.fit_transform(embedding)
    pca_cols = ['c' + str(i) for i in range(n_component)]
    pca_df = pd.DataFrame(pca_array, index=embedding.index, columns=pca_cols)

    pca_save_path = os.path.join(save_dir_path, 'pca_embedding_df.parquet.gzip')
    pca_df.to_parquet(pca_save_path, compression='gzip')

    return pca_df

if __name__ == '__main__':
    all_pathes = get_path(
        dir_path='/home/timnaka123/Documents/financial-news-dataset/ReutersNews106521'
        )

    # text_input = embedding_batch_preprocessing(all_pathes, tickers)


    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    model = TFBertModel.from_pretrained("bert-large-uncased")

    embedding_df = generate_embedding(
        all_pathes = all_pathes,
        tokenizer=tokenizer,
        model=model,
        save_dir_path='/home/timnaka123/Documents/stock_embedding_nlp/src/data/'
        )

    pca_df = bert_compression(
        embedding=embedding_df,
        save_dir_path='/home/timnaka123/Documents/stock_embedding_nlp/src/data/'
        )



    # embedding_df.columns = ['c' + str(e) for e in embedding_df.columns]
    # embedding_df.to_parquet(
    #     '/home/timnaka123/Documents/stock_embedding_nlp/src/data/embedding_df.parquet.gzip',
    #     compression='gzip'
    #     )
