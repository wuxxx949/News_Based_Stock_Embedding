import multiprocessing as mp
from functools import reduce
from typing import List, Optional
import fasttext

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.bert_embedding import (embedding_batch_preprocessing,
                                     single_file_process)
from src.data.explore import get_path, pickle_results
from src.data.utils import STOPWORDS

bin_file='/home/timnaka123/Documents/stock_embedding_nlp/src/data/news.bin'
# cannot pickle fasttext object in MP so make it global
model = fasttext.load_model(path=str(bin_file))


def _tfidf_weighted_embedding(
    x: csr_matrix,
    news_id: List[str],
    vectorizer
    ):
    nzero = x.nonzero()
    nonzero_idx = list(zip(nzero[0], nzero[1]))
    # i = 0
    i = nonzero_idx[0][0]
    results = {}

    tmp_embedding = 0
    for c in list(nonzero_idx):
        if c[0] > i:
            results[news_id[i]] = tmp_embedding
            # i += 1
            i = c[0]
            tmp_embedding = 0

        w = vectorizer.get_feature_names_out()[c[1]]
        tfidf = x[c]
        tmp_embedding += tfidf * model.get_word_vector(w)

    # attach last row
    results[news_id[i]] = tmp_embedding

    return results


def tfidf_weighted_embedding(
    x: csr_matrix,
    trained_vecterizer,
    news_id: List[str],
    ncores: Optional[int] = None,
    batch_size: int = 500,
    ) -> List[str]:
    nrows = x.shape[0]
    npart = -(-nrows // batch_size)
    input_lst = [(x[i * batch_size: (i+1) * batch_size, :], news_id[i * batch_size: (i+1) * batch_size], trained_vecterizer)
                 for i in range(npart)]
    ncores = ncores if ncores is not None else mp.cpu_count()

    with mp.Pool(processes=ncores) as p:
        out = p.starmap(_tfidf_weighted_embedding, input_lst)

    # reduce(lambda d1, d2: dict(d1, **d2), out)
    return reduce(lambda d1, d2: dict(d1, **d2), out)


if __name__ == '__main__':
    all_paths = get_path(
        dir_path='/home/timnaka123/Documents/financial-news-dataset/ReutersNews106521'
        )[: 1000]

    input_data = embedding_batch_preprocessing(paths=all_paths)

    corpus = [e[2] for e in input_data if e[2] is not None]
    news_id = [e[1] for e in input_data if e[2] is not None]

    vectorizer = TfidfVectorizer(min_df=0.05, stop_words=STOPWORDS, dtype=np.float32)

    X = vectorizer.fit_transform(corpus)

    test = tfidf_weighted_embedding(
        x=X,
        trained_vecterizer=vectorizer,
        news_id=news_id,
        batch_size=100
        )

    from scipy.sparse import csr_matrix
    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    vectorizer.get_feature_names_out()

    x = X.toarray()
    x = np.concatenate([np.array([0] * 9).reshape((1, 9)), x, np.array([0] * 9).reshape((1, 9))])
    x[3, :] = 0
    y = csr_matrix(x)

    out = _tfidf_weighted_embedding(x=y, news_id=['a', 'b', 'c', 'd', 'e'], vectorizer=vectorizer)