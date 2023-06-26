from typing import Dict, List

import cvxpy as cp
import numpy as np

from src.data.stock_data import stock_anual_return_calc
from src.data.utils import load_pickled_obj
from src.portfolio.utils import embeddings_to_corr


class PortfolioConstruction:
    def __init__(
        self,
        embedding_path: str,
        embedding_file: str,
        last_news_date: str
        ) -> None:
        """constructor

        Args:
            embedding_path (str): direcotry path for embedding pickle
            embedding_file (str): embedding pickle file name
            last_news_date (str): the last date of news used in training embeddings
        """
        embedding_dict = load_pickled_obj(
            dir=embedding_path,
            name=embedding_file
            )
        self.last_news_date = last_news_date
        self.tickers, self.embeddings = self._get_embeddings(embedding_dict)
        self.embedding_corr = embeddings_to_corr(self.embeddings)
        # historical average annual return over 10 years
        self.hist_return = self.get_stock_avg_return()


    @staticmethod
    def _get_embeddings(embedding_dict: Dict[str, np.array]) -> List[str, np.array]:
        tickers = list(embedding_dict.keys())
        embeddings = np.vstack(embedding_dict.values())

        return tickers, embeddings

    def _return_calc_input(self) -> None:
        date_split = self.last_news_date.split('-')
        self.max_year = date_split[0]
        self.mmdd = '-'.join(date_split[1:])

    def get_stock_avg_return(self) -> np.array:
        self._return_calc_input()
        out = stock_anual_return_calc(
            tickers=self.tickers,
            min_year=self.max_year - 10,
            max_year=self.max_year,
            mmdd=self.mmdd
            ) \
            .set_index('ticker') \
            .filter(self.tickers, axis=0) \
            .loc[:, 'annual_return']

        assert len(out) == len(self.tickers), 'avg return len differs from input tickers'

        return out

    def portfolio_opt(self, exp_return: float, cov_mat: np.array) -> np.array:
        wp = cp.Variable(self.n, nonneg=True)