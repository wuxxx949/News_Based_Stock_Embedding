from typing import Dict, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd

from src.data.stock_data import stock_annual_return_calc
from src.model.utils import lazyproperty
from src.portfolio.utils import embeddings_to_corr


class PortfolioConstruction:
    def __init__(
        self,
        embedding_dict: Dict[str, np.array],
        last_news_date: str,
        year_lookback: int = 3
        ) -> None:
        """constructor

        Args:
            embedding_dict (Dict[str, np.array]): ticker as key and embedding as value
            last_news_date (str): the last date of news used in training embeddings
            year_lookback (int): number of years look back for historical return
        """
        self.last_news_date = last_news_date
        tickers, self.embeddings = self._get_embeddings(embedding_dict)
        self.tickers = [e.upper() for e in tickers]
        self.embedding_corr = embeddings_to_corr(self.embeddings)
        # historical average annual return over 10 years
        self.hist_return = self.get_stock_avg_return(year_lookback)


    @staticmethod
    def _get_embeddings(embedding_dict: Dict[str, np.array]) -> Tuple[str, np.array]:
        tickers = sorted(list(embedding_dict.keys()))
        embedding_dict = {i: embedding_dict[i] for i in tickers}
        embeddings = np.vstack(embedding_dict.values())

        return tickers, embeddings

    def _return_calc_input(self) -> None:
        date_split = self.last_news_date.split('-')
        self.max_year = int(date_split[0])
        self.mmdd = '-'.join(date_split[1:])

    def get_stock_avg_return(self, year_lookback: int = 3) -> np.array:
        """calculate average stock annual return

        Args:
            year_lookback (int): number of years look back

        Returns:
            np.array: average annual return
        """
        self._return_calc_input()
        out = stock_annual_return_calc(
            tickers=self.tickers,
            min_year=self.max_year - year_lookback,
            max_year=self.max_year,
            mmdd=self.mmdd
            ) \
            .set_index('ticker') \
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
        """calculate return of the tickers in the backtest period

        Returns:
            np.array: one year return with order aligned with tickers
        """
        out = stock_annual_return_calc(
            tickers=self.tickers,
            min_year=self.max_year,
            max_year=self.max_year + 1,
            mmdd=self.mmdd
            ) \
            .set_index('ticker') \
            .filter(self.tickers, axis=0) \
            .loc[:, 'annual_return']

        return out.to_numpy()

    @lazyproperty
    def backtest_return(self) -> np.array:
        return self.get_one_year_back_test_return()

    def get_backtest_results(self, exp_return: float) -> Tuple[float, pd.DataFrame]:
        """calculate backtest return rate

        Args:
            exp_return (float): expected return used in the optimization

        Returns:
            Tuple[float, pd.DataFrame]:  portfolio return in the backtest period and portfolio weights
        """
        # optimal weights
        opt_w = self.portfolio_opt(exp_return=exp_return, cov_mat=self.embedding_corr)

        # portfolio weights
        weights_df = pd.DataFrame({'ticker': self.tickers, 'weight': opt_w}) \
            .sort_values('weight', ascending=False, ignore_index=True)


        return self.backtest_return @ opt_w, weights_df
