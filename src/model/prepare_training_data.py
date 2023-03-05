import datetime as dt
import os
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.bert_embedding import embedding_batch_preprocessing
from src.data.explore import get_path
from src.data.tfidf_embedding import tfidf_weighted_embedding
from src.data.utils import STOPWORDS
from src.model.utils import extract_date, lazyproperty, to_date


class DateManager:
    """manage date related task for model and portfolio optimization
    """
    def __init__(self, news_path: str) -> None:
        """
        Args:
            news_path (str): path to news directory
        """
        self.news_path = news_path
        self.min_date, self.max_date = self._get_dates()

    def _get_dates(self) -> Tuple[datetime, datetime]:
        """get min and max dates of news article

        Returns:
            Tuple[datetime, datetime]: min and max of news dates
        """
        subdirs = os.listdir(self.news_path)
        dates = [datetime.strptime(d, '%Y%m%d') for d in subdirs if d[0].isdigit()]

        return min(dates), max(dates)

    def get_model_date(
        self,
        training_len: int,
        validation_len: int
        ) -> Tuple[datetime, datetime, datetime]:
        """configure model dates

        Args:
            training_len (int): number of years for training
            validation_len (int): number of years for validation

        Raises:
            ValueError: if training length too long to have validation data

        Returns:
            Tuple[datetime, datetime, datetime]: training start date, validation start date,
            and validation end date
        """
        validation_start = self.min_date.replace(year=self.min_date.year + training_len)
        if validation_start >= self.max_date:
            raise ValueError('training length to long, no validation data left')

        validation_end = self.min_date.replace(year=validation_start.year + validation_len)
        validation_end = min(validation_end, self.max_date)

        return self.min_date, validation_start, validation_end


class ModelDataPrep:
    def __init__(
        self,
        news_path: str,
        save_dir_path: str,
        training_start_date: datetime,
        validation_start_date: datetime,
        validation_end_date: datetime
        ) -> None:
        """
        Args:
            news_path (str): path to news directory
            save_dir_path (str): path to news directory
            training_start_date (datetime): training start date, inclusive
            validation_start_date (datetime): validation start date, inclusive
            validation_end_date (datetime): validation end date, non-inclusive
        """
        self.news_path = news_path
        self.save_dir_path = save_dir_path
        self.training_start_date = training_start_date
        self.validation_start_date = validation_start_date
        self.validation_end_date = validation_end_date
        self.target = pd.DataFrame()
        self.bert_embedding = pd.DataFrame()
        self._load_bert_embeddings()
        self._load_target()

    def _load_target(self):
        target = pd.read_parquet(
            os.path.join(self.save_dir_path, 'target_df.parquet.gzip')
            )
        self.target = target.set_index('date')


    def _load_bert_embeddings(self):
        bert_embedding = pd.read_parquet(
            os.path.join(self.save_dir_path, 'pca_embedding_df.parquet.gzip')
            )
        # use date as index for time slicing
        self.bert_embedding = bert_embedding.reset_index(level=1, drop=False)

    def _news_file_filter(
        self,
        min_date: datetime,
        max_date: datetime
        ) -> List[str]:
        """find all news article between min date and max date

        Args:
            min_date (datetime): start date
            max_date (datetime): end date (not included)

        Returns:
            List[str]: file paths of qualified files
        """
        all_paths = get_path(self.news_path)
        min_date_str = datetime.strftime(min_date, '%Y%m%d')
        max_date_str = datetime.strftime(max_date, '%Y%m%d')

        out = []
        for p in all_paths:
            date = extract_date(p)
            if date < max_date_str and date >= min_date_str:
                out.append(p)

        return out

    @staticmethod
    def _tfidf_to_df(data_dict: Dict[str, np.array]) -> pd.DataFrame:
        df = pd.DataFrame(data_dict).T \
            .reset_index(names='news_id') \
            .rename(lambda x: f'f{x}' if str(x).isdigit() else x, axis=1)

        return df

    def _get_tfidf_vector(
        self,
        tfidf_start_date: datetime,
        tfidf_end_date: datetime,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        ) -> pd.DataFrame:
        """calculate tfidf embedding for given date inputs

        Args:
            tfidf_start_date (datetime): news start date for known date range
            tfidf_end_date (datetime): news end date for known date range
            start_date (Optional[datetime], optional): start date for output. Defaults to None.
            end_date (Optional[datetime], optional): end date for output. Defaults to None.

        Returns:
            Dict[str, np.array]: key of news id and value of news tfidf embeddings
        """
        if start_date is None:
            start_date = tfidf_start_date
        if end_date is None:
            end_date = tfidf_end_date

        # training peroid data
        tfidf_news_files = self._news_file_filter(tfidf_start_date, tfidf_end_date)
        input_data = embedding_batch_preprocessing(paths=tfidf_news_files)
        tfidf_corpus = [e[2] for e in input_data if e[2] is not None]
        tfidf_news_id = [e[1] for e in input_data if e[2] is not None]

        vectorizer = TfidfVectorizer(min_df=0.05, stop_words=STOPWORDS, dtype=np.float32)
        vectorizer.fit(tfidf_corpus)

        if (start_date is None) and (end_date is None):
            corpus = tfidf_corpus
            news_id = tfidf_news_id
        else:
            news_files = self._news_file_filter(start_date, end_date)
            input_data = embedding_batch_preprocessing(paths=news_files)
            corpus = [e[2] for e in input_data if e[2] is not None]
            news_id = [e[1] for e in input_data if e[2] is not None]

        X = vectorizer.transform(corpus)

        out = tfidf_weighted_embedding(
            x=X,
            trained_vecterizer=vectorizer,
            news_id=news_id)

        return self._tfidf_to_df(out)

    @lazyproperty
    def training_tfidf_df(self) -> pd.DataFrame:
        """tfidf based on training peroid data
        """
        embedding = self._get_tfidf_vector(
            tfidf_start_date=self.training_start_date,
            tfidf_end_date=self.validation_start_date
            )

        return embedding

    @lazyproperty
    def validation_tfidf_df(self) -> pd.DataFrame:
        """tfidf embedding based on both training and validation peroid data
        """
        embedding = self._get_tfidf_vector(
            tfidf_start_date=self.training_start_date,
            tfidf_end_date=self.validation_start_date,
            start_date=self.validation_start_date,
            end_date=self.validation_end_date
            )

        return embedding

    def _get_bert_vector(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        start_date = str(start_date.date())
        end_date = str(end_date.date())

        return self.bert_embedding[start_date: end_date]

    @lazyproperty
    def training_bert_df(self) -> pd.DataFrame:
        end_date = self.validation_start_date - dt.timedelta(1)
        out = self._get_bert_vector(
            start_date=self.training_start_date,
            end_date=end_date
            )

        return out

    @lazyproperty
    def validation_bert_df(self) -> pd.DataFrame:
        end_date = self.validation_end_date - dt.timedelta(1)
        out = self._get_bert_vector(
            start_date=self.validation_start_date,
            end_date=end_date
            )

        return out

    def _get_target(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        start_date = str(start_date.date())
        end_date = str(end_date.date())

        return self.target[start_date: end_date]

    @lazyproperty
    def training_target(self) -> pd.DataFrame:
        end_date = self.validation_start_date - dt.timedelta(1)
        out = self._get_target(
            start_date=self.training_start_date,
            end_date=end_date
            )

        return out

    @lazyproperty
    def validation_target(self) -> pd.DataFrame:
        end_date = self.validation_end_date - dt.timedelta(1)
        out = self._get_target(
            start_date=self.validation_start_date,
            end_date=end_date
            )

        return out

    def prep_raw_model_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        training_df = self.training_bert_df.reset_index() \
            .merge(self.training_tfidf_df, on='news_id', how='inner') \
            .sort_values('date') \
            .set_index('date')

        validation_df = self.validation_bert_df.reset_index() \
            .merge(self.validation_tfidf_df, on='news_id', how='inner') \
            .sort_values('date') \
            .set_index('date')

        return training_df, self.training_target, validation_df, self.validation_target

    @lazyproperty
    def training_trading_date(self) -> List[datetime]:
        return list(set(to_date(self.training_target.index)))

    @lazyproperty
    def validation_trading_date(self) -> List[datetime]:
        return list(set(to_date(self.validation_target.index)))

    @lazyproperty
    def training_news_date(self) -> List[datetime]:
        return list(set(to_date(self.training_bert_df.index)))

    @lazyproperty
    def validation_news_date(self) -> List[datetime]:
        return list(set(to_date(self.validation_bert_df.index)))

    @staticmethod
    def _fetch_embeddings(
        trading_dates: Iterable[datetime],
        news_dates: Iterable[datetime],
        embedding_df: pd.DataFrame,
        days_lookback: int = 5
        ) -> Dict[str, List[np.array]]:
        """prepare embedding vectors within the range of days for given trading date

        Args:
            trading_dates (Iterable[datetime]): a set of trading dates
            news_dates (Iterable[datetime]): a set of news dates
            embedding_df (pd.DataFrame): bert and tfidf embedding dataframe
            days_lookback (int, optional): number of days to look back for news. Defaults to 5.

        Returns:
            Dict[str, np.array]: trading date str as key and embedding arrays as value
        """
        out = {}
        for td in trading_dates:
            nd = [e for e in news_dates if 0 <= (e - td).days <= (days_lookback - 1)]
            if len(nd) != 5:
                continue
            # fetch newes
            min_date = str(min(nd))
            max_date = str(max(nd))
            tmp_df = embedding_df[min_date: max_date]
            bert_embedding = tmp_df.filter(regex='^c', axis=1).groupby('date')
            tfidf_embedding = tmp_df.filter(regex='^f', axis=1).groupby('date')
            tmp_bert = []
            bert_max_news = max([len(e) for e in dict(list(bert_embedding)).values()])
            for _, d in bert_embedding:
                # need to pad to the longest number of news in the neighboring days
                d = d.reset_index(drop=True).reindex(range(bert_max_news ), fill_value=0)
                tmp_bert.append(d.to_numpy().tolist())
            tmp_tfidf = []
            for _, d in tfidf_embedding:
                d = d.reset_index(drop=True).reindex(range(bert_max_news ), fill_value=0)
                tmp_tfidf.append(d.to_numpy().tolist())

            out[str(td)] = [np.array(tmp_bert), np.array(tmp_tfidf)]

        return out



if __name__ == '__main__':
    dm = DateManager(news_path='/home/timnaka123/Documents/financial-news-dataset/ReutersNews106521')
    t_start_date, v_start_date, v_end_date = dm.get_model_date(2, 1)

    mdp = ModelDataPrep(
        news_path='/home/timnaka123/Documents/financial-news-dataset/ReutersNews106521',
        save_dir_path='/home/timnaka123/Documents/stock_embedding_nlp/src/data',
        training_start_date=t_start_date,
        validation_start_date=v_start_date,
        validation_end_date=v_end_date
    )

    train_df, train_target, valid_df, valid_target = mdp.prep_raw_model_data()

    out = {}
    for td in mdp.training_trading_date:
        nd = [e for e in mdp.training_news_date if 0 <= (e - td).days <= 4]
        if len(nd) != 5:
            continue
        # fetch newes
        min_date = str(min(nd))
        max_date = str(max(nd))
        tmp_df = train_df[min_date: max_date]
        bert_embedding = tmp_df.filter(regex='^c', axis=1).groupby('date')
        tfidf_embedding = tmp_df.filter(regex='^f', axis=1).groupby('date')
        tmp_bert = []
        bert_max_news = max([len(e) for e in dict(list(bert_embedding)).values()])
        for _, d in bert_embedding:
            # need to pad to the longest number of news in the neighboring days
            d = d.reset_index(drop=True).reindex(range(bert_max_news ), fill_value=0)
            tmp_bert.append(d.to_numpy().tolist())
        tmp_tfidf = []
        for _, d in tfidf_embedding:
            d = d.reset_index(drop=True).reindex(range(bert_max_news ), fill_value=0)
            tmp_tfidf.append(d.to_numpy().tolist())

        out[str(td)] = [np.array(tmp_bert), np.array(tmp_tfidf)]



