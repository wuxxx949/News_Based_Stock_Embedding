import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.bert_embedding import embedding_batch_preprocessing
from src.data.explore import get_path
from src.data.tfidf_embedding import tfidf_weighted_embedding
from src.data.utils import STOPWORDS


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


def extract_date(fpath: str) -> str:
    # TODO: make it OS agnostic
    # https://stackoverflow.com/questions/4579908/cross-platform-splitting-of-path-in-python
    d = fpath.split('/')
    date = [e for e in d if all([ee.isnumeric() for ee in e]) and len(e) > 0]
    if len(date) == 0:
        raise ValueError('no date in the path')

    if len(date) > 1:
        raise ValueError('multiple candidates for date in a path')

    return date[0]


class ModelDataPrep:
    def __init__(
        self,
        news_path: str,
        training_start_date: datetime,
        validation_start_date: datetime,
        validation_end_date: datetime
        ) -> None:
        """
        Args:
            news_path (str): path to news directory
            training_start_date (datetime): training start date, inclusive
            validation_start_date (datetime): validation start date, inclusive
            validation_end_date (datetime): validation end date, non-inclusive
        """
        self.news_path = news_path
        self.training_start_date = training_start_date
        self.validation_start_date = validation_start_date
        self.validation_end_date = validation_end_date


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

    def _get_tfidf_vector(
        self,
        tfidf_start_date: datetime,
        tfidf_end_date: datetime,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        ) -> Dict[str, np.array]:
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

        return out

    @property
    def training_tfidf_vector(self) -> Dict[str, np.array]:
        """tfidf based on training peroid data
        """
        embedding = self._get_tfidf_vector(
            tfidf_start_date=self.training_start_date,
            tfidf_end_date=self.validation_start_date
            )

        return embedding

    @property
    def validation_tfidf_vector(self) -> Dict[str, np.array]:
        """tfidf embedding based on both training and validation peroid data
        """
        embedding = self._get_tfidf_vector(
            tfidf_start_date=self.training_start_date,
            tfidf_end_date=self.validation_start_date,
            start_date=self.validation_start_date,
            end_date=self.validation_end_date
            )

        return embedding


if __name__ == '__main__':
    dm = DateManager(news_path='/home/timnaka123/Documents/financial-news-dataset/ReutersNews106521')
    t_start_date, v_start_date, v_end_date = dm.get_model_date(2, 1)

    mdp = ModelDataPrep(
        news_path='/home/timnaka123/Documents/financial-news-dataset/ReutersNews106521',
        training_start_date=t_start_date,
        validation_start_date=v_start_date,
        validation_end_date=v_end_date
    )

    train_tfidf_embedding = mdp.training_tfidf_vector