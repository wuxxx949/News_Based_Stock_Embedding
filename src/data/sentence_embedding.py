import os
from typing import List, Tuple

import numpy as np
import pandas as pd

try: # for modeling with tf
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
except ModuleNotFoundError:
    print('no sentence_transformers installed, proceed with modeling using tf')

from src.data.bert_embedding import (embedding_batch_preprocessing,
                                     get_news_path)
from src.meta_data import get_meta_data


def sentence_embedding(
    input: Tuple[str, str, str],
    batch_size: int
    ) -> List[Tuple[str, str, np.array]]:
    """handle input from embedding_batch_preprocessing

    Args:
        inpt (Tuple[str, str, str]): date, news id, and processed text

    Returns:
        List[Tuple[str, str, np.array]]: date, news_id, and embedding
    """
    text = [e[2] for e in input if e[2] is not None]
    date = [e[0] for e in input if e[2] is not None]
    news_id = [e[1] for e in input if e[2] is not None]

    sentence_embeddings = model.encode(text, batch_size=batch_size)

    return list(zip(date, news_id, sentence_embeddings))


def generate_embedding(
    all_paths: List[str],
    batch_size: int = 1000,
    ) -> None:
    """produce

    Args:
        all_paths (List[str]): news file paths
        batch_size (int, optional): batch size for sentence transformer. Defaults to 1000.
    """
    save_dir_path = get_meta_data()['SAVE_DIR']
    results = []
    text_input = embedding_batch_preprocessing(all_paths)

    nparts = -(-len(text_input) // batch_size)

    for i in range(nparts):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        print(f'train {start_idx} to {end_idx}')
        out = sentence_embedding(
            input= text_input[start_idx: end_idx],
            batch_size=batch_size
            )
        results.extend(out)

    embedding_array = np.array([e[2] for e in results])
    news_date = [e[0] for e in results]
    news_id = [str(e[1]) for e in results]

    array_len = embedding_array.shape[1]
    cols = ['c' + str(i) for i in range(array_len)]
    embedding_df = pd.DataFrame(embedding_array, columns=cols)
    embedding_df['date'] = pd.to_datetime(news_date)
    embedding_df['news_id'] = news_id
    embedding_df = embedding_df.set_index(['date', 'news_id'])

    embedding_df.to_parquet(
        path=os.path.join(save_dir_path, 'sentence_embedding_df.parquet.gzip'),
        compression='gzip'
    )


if __name__ == '__main__':
    all_paths = get_news_path()

    generate_embedding(all_paths=all_paths, batch_size=2000)


