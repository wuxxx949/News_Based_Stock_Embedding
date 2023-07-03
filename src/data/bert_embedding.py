import multiprocessing as mp
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn import decomposition

from src.data.utils import (create_uuid_from_string, get_path,
                            headline_processing, longest_line, pickle_results)
from src.meta_data import get_meta_data


def get_news_path() -> List[str]:
    """fetch path for all news sources

    Returns:
        List[str]: list of paths
    """
    meta_data = get_meta_data()
    reuters_dir = meta_data['REUTERS_DIR']
    bloomberg_dir = meta_data['BLOOMBERG_DIR']
    return get_path(reuters_dir) + get_path(bloomberg_dir)


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
        headline = headline_processing(lines[0])
        if len(headline) == 0:
            headline = headline_processing(lines[1])
    except Exception:
        print(f'fail to load file: {path}')
        return None

    if len(headline) <= 0: # empty string
        return None

    return headline


def bert_processing(path: str) -> Tuple[str, str, str]:
    """fetch date, news

    Args:
        path (str): _description_

    Returns:
        Tuple[str, str, str]: _description_
    """
    # TODO: make it os agnostic
    date = [e for e in path.split('/')[1:] if len(re.sub(r'[0-9]|-', '', e)) == 0][0]
    date = re.sub(r'[^0-9]', '', date)
    file_name = path.split('/')[-1]
    news_id = create_uuid_from_string(file_name)

    return date, news_id, single_file_process(path)


def embedding_batch_preprocessing(
    paths: List[str],
    ncores: Optional[int] = None,
    ) -> List[Tuple[str, str, str]]:
    """preprocess news text for Bert embedding

    Args:
        pathes (List[str]): pathes in the batch
        ncores (Optional[int], optional): number of workers for mp. Defaults to None.

    Returns:
        List[Tuple[str, str, str]]: date, news_id, and processed news headlines
    """
    ncores = ncores if ncores is not None else mp.cpu_count()
    with mp.Pool(processes=ncores) as p:
        out = p.map(bert_processing, paths)

    return out


def bert_embedding(
    input: Tuple[str, str, str],
    tokenizer,
    max_len: int,
    model
    ) -> List[Tuple[str, np.array]]:
    """handle input from embedding_batch_preprocessing

    Args:
        inpt (Tuple[str, str, str]): date, news id, and processed text
        tokenizer (_type_): _description_
        max_len (int): token length
        model (_type_): _description_

    Returns:
        np.array: _description_
    """
    text = [e[2] for e in input if e[2] is not None]
    date = [e[0] for e in input if e[2] is not None]
    news_id = [e[1] for e in input if e[2] is not None]

    encoded_input = tokenizer(
        text,
        max_length=max_len,
        truncation=True,
        return_tensors='tf',
        padding=True
        )

    output = model(encoded_input)
    embedding = output.pooler_output

    return list(zip(date, news_id, embedding.numpy()))


def generate_embedding(
    all_paths: List[str],
    tokenizer,
    model,
    max_len: int,
    batch_size: int = 1000,
    ) -> Tuple[np.ndarray, List[str], List[str]]:
    """produce

    Args:
        all_pathes (List[str]): _description_
        tokenizer (_type_): _description_
        model (_type_): _description_
        batch_size (int, optional): _description_. Defaults to 40.

    Returns:
        Tuple[np.ndarray, List[str], List[str]]: news embedding, date, and news id
    """
    save_dir_path = get_meta_data()['SAVE_DIR']
    results = []
    text_input = embedding_batch_preprocessing(all_paths)

    nparts = -(-len(text_input) // batch_size)

    for i in range(nparts):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        print(f'train {start_idx} to {end_idx}')
        out = bert_embedding(
            input= text_input[start_idx: end_idx],
            tokenizer=tokenizer,
            max_len=max_len,
            model=model
            )
        results.extend(out)

    pickle_results(dir=save_dir_path, name='embedding_lst.pickle', obj=results)
    # avoid pandas due to extremely large memory footprint
    embedding_array = np.array([e[2] for e in results])
    news_date = [e[0] for e in results]
    news_id = [str(e[1]) for e in results]

    return embedding_array, news_date, news_id


def bert_compression(
    embedding_array: np.ndarray,
    news_date: List[str],
    news_id: List[str],
    n_component: int = 256
    ) -> pd.DataFrame:
    """apply PCA to reduce dimension

    Args:
        embedding (pd.DataFrame): original bert embedding

    Returns:
        pd.DataFrame: PCA embedding
    """
    save_dir_path = get_meta_data()['SAVE_DIR']
    pca = decomposition.PCA(n_components=n_component)
    pca_array =  pca.fit_transform(embedding_array)
    pca_cols = ['c' + str(i) for i in range(n_component)]
    pca_df = pd.DataFrame(pca_array, columns=pca_cols)
    pca_df['date'] = pd.to_datetime(news_date)
    pca_df['news_id'] = news_id
    pca_df = pca_df.set_index(['date', 'news_id'])

    pca_save_path = os.path.join(save_dir_path, 'pca_embedding_df.parquet.gzip')
    pca_df.to_parquet(pca_save_path, compression='gzip')

    return pca_df


if __name__ == '__main__':
    from transformers import BertTokenizer, TFBertModel

    headline_path = '/home/timnaka123/Documents/stock_embedding_nlp/src/data/headlines.txt'
    max_headline = longest_line(headline_path)

    all_paths = get_news_path()

    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    model = TFBertModel.from_pretrained("bert-large-uncased")

    embedding_array, news_date, news_id = generate_embedding(
        all_pathes = all_paths,
        tokenizer=tokenizer,
        model=model,
        max_len=max_headline,
        )

    pca_df = bert_compression(
        embedding_array=embedding_array,
        news_date=news_date,
        news_id=news_id,
        )
