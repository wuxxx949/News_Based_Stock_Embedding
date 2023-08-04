import multiprocessing as mp
import os
from functools import reduce
from typing import List, Optional

import fasttext
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.meta_data import get_meta_data

try:
    bin_file=os.path.join(get_meta_data()['SAVE_DIR'], 'news.bin')
    # cannot pickle fasttext object in MP so make it global
    model = fasttext.load_model(path=str(bin_file))
except ValueError:
    print(f'No fasttext model object {bin_file}, train embeddings first')


def _tfidf_weighted_embedding(
    x: csr_matrix,
    news_id: List[str],
    vectorizer: TfidfVectorizer
    ):
    """note that some news may have all 0 in the sparse matrix depends on min_df input
    """
    nzero = x.nonzero()
    nonzero_idx = list(zip(nzero[0], nzero[1]))
    # i = 0
    i = nonzero_idx[0][0]
    results = {}
    tmp_embedding = 0
    for idx in nonzero_idx:
        if idx[0] > i:
            results[news_id[i]] = tmp_embedding
            # reset
            i = idx[0]
            tmp_embedding = 0

        w = vectorizer.get_feature_names_out()[idx[1]]
        tfidf = x[idx]
        tmp_embedding += tfidf * model.get_word_vector(w)

    # attach the last row
    results[news_id[i]] = tmp_embedding

    return results


def tfidf_weighted_embedding(
    x: csr_matrix,
    trained_vecterizer: TfidfVectorizer,
    news_id: List[str],
    ncores: Optional[int] = None,
    batch_size: int = 500,
    debug: bool = False
    ) -> List[str]:
    nrows = x.shape[0]
    npart = -(-nrows // batch_size)
    input_lst = [(x[i * batch_size: (i+1) * batch_size, :], news_id[i * batch_size: (i+1) * batch_size], trained_vecterizer)
                 for i in range(npart)]
    ncores = ncores if ncores is not None else mp.cpu_count()

    if debug:
        out = []
        for i in input_lst:
            out.append(_tfidf_weighted_embedding(*i))
    else:
        with mp.Pool(processes=ncores) as p:
            out = p.starmap(_tfidf_weighted_embedding, input_lst)

    # reduce(lambda d1, d2: dict(d1, **d2), out)
    return reduce(lambda d1, d2: dict(d1, **d2), out)
