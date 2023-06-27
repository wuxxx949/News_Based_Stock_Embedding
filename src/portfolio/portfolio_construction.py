from typing import Dict, Tuple

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
    def _get_embeddings(embedding_dict: Dict[str, np.array]) -> Tuple[str, np.array]:
        tickers = list(embedding_dict.keys())
        embeddings = np.vstack(embedding_dict.values())

        return tickers, embeddings

    def _return_calc_input(self) -> None:
        date_split = self.last_news_date.split('-')
        self.max_year = int(date_split[0])
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

        return out.to_numpy()

    def portfolio_opt(self, exp_return: float, cov_mat: np.array) -> np.array:
        constraints = []
        wp = cp.Variable(len(self.tickers), nonneg=True)

        # sum to 1 constraint
        constraints.append(sum(wp) == 1)
        # expectation constraint
        constraints.append(self.hist_return[np.newaxis,:] @ wp == exp_return)
        # 0 <= w <= 1 constraint
        for i, _ in enumerate(self.tickers):
            constraints.append(wp[i] <= 1)
            constraints.append(wp[i] >= 0)

        prob = cp.Problem(cp.Minimize(cp.quad_form(wp, cov_mat)), constraints)
        prob.solve()

        return wp.value

    def embedding_based_opt(self, exp_return:float) -> np.array:
        out = self.portfolio_opt(exp_return=exp_return, cov_mat=self.embedding_corr)

        return out

    def get_one_year_back_test_return(self) -> np.array:
        out = stock_anual_return_calc(
            tickers=self.tickers,
            min_year=self.max_year,
            max_year=self.max_year + 1,
            mmdd=self.mmdd
            ) \
            .set_index('ticker') \
            .filter(self.tickers, axis=0) \
            .loc[:, 'annual_return']

        return out.to_numpy()

    def get_backtest_return(self, exp_return: float) -> float:
        # optimal weights
        opt_w = self.portfolio_opt(exp_return=exp_return, cov_mat=self.embedding_corr)
        # future 1 year return
        backtest_return = self.get_one_year_back_test_return()

        return backtest_return @ opt_w



if __name__ == '__main__':
    portfolio_constructor = PortfolioConstruction(
        embedding_path='/home/timnaka123/Documents/stock_embedding_nlp/src/model/',
        embedding_file='embedding.pickle',
        last_news_date='2013-10-20'
        )

    weights = portfolio_constructor.embedding_based_opt(exp_return=0.2)

    r = portfolio_constructor.get_backtest_return(exp_return=0.2)