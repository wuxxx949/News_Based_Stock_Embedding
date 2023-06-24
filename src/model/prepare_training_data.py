import datetime as dt
import os
from datetime import datetime
from itertools import chain
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.python.data.ops.dataset_ops import PaddedBatchDataset

from src.data.bert_embedding import embedding_batch_preprocessing
from src.data.explore import get_path
from src.data.tfidf_embedding import tfidf_weighted_embedding
from src.data.utils import STOPWORDS
from src.logger import setup_logger
from src.model.utils import extract_date, lazyproperty, to_date

logger = setup_logger('data', 'data.log')


class DateManager:
    """manage date related task for model and portfolio optimization
    """
    def __init__(
        self,
        reuters_news_path: str,
        bloomberg_news_path: str
        ) -> None:
        """
        Args:
            news_path (str): path to news directory
        """
        self.reuters_news_path = reuters_news_path
        self.bloomberg_news_path = bloomberg_news_path
        self.min_date, self.max_date = self._get_dates()

    def _get_dates(self) -> Tuple[datetime, datetime]:
        r_min_date, r_max_date = self._get_reuters_dates()
        b_min_date, b_max_date = self._get_bloomberg_dates()

        # take intersection
        min_date = max(r_min_date, b_min_date)
        max_date = min(r_max_date, b_max_date)
        logger.info(f'news data range: {min_date} to {max_date}')

        return min_date, max_date

    def _get_reuters_dates(self) -> Tuple[datetime, datetime]:
        """get min and max dates of reuters news article

        Returns:
            Tuple[datetime, datetime]: min and max of news dates
        """
        subdirs = os.listdir(self.reuters_news_path)
        dates = [datetime.strptime(d, '%Y%m%d') for d in subdirs if d[0].isdigit()]

        return min(dates), max(dates)

    def _get_bloomberg_dates(self) -> Tuple[datetime, datetime]:
        """get min and max dates of news article

        Returns:
            Tuple[datetime, datetime]: min and max of news dates
        """
        subdirs = os.listdir(self.bloomberg_news_path)
        dates = [datetime.strptime(d, '%Y-%m-%d') for d in subdirs if d[0].isdigit()]

        return min(dates), max(dates)

    def get_ts_model_date(
        self,
        training_len: int,
        validation_len: int
        ) -> Tuple[datetime, datetime, datetime]:
        """configure model dates if use time series for training / validation partition

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

    def get_random_split_date(self, data_len: int) -> Tuple[datetime, datetime]:
        """configure news date range for given data length in year

        Args:
            data_len (int): number of years used for training and validation

        Returns:
            Tuple[datetime, datetime]: start and end date
        """
        max_date = self.min_date.replace(year=self.min_date.year + data_len)
        if max_date >= self.max_date:
            logger.warn('max date exceeds last day of news data')
            max_date = self.max_date

        return self.min_date, max_date


class ModelDataPrep:
    def __init__(
        self,
        reuters_news_path: str,
        bloomberg_news_path: str,
        save_dir_path: str,
        min_date: datetime,
        max_date: datetime,
        min_df: float = 0.0001
        ) -> None:
        """Constructor

        Args:
            reuters_news_path (str): path to reuters news directory
            bloomberg_news_path (str): path to bloomberg news directory
            save_dir_path (str): path to news directory
            min_date (datetime): min news date, inclusive
            max_date (datetime): max news date, inclusive
            min_dfend_date (float, optional): min_df arg for TfidfVectorizer. Defaults to 0.0001.
        """
        self.reuters_news_path = reuters_news_path
        self.bloomberg_news_path = bloomberg_news_path
        self.save_dir_path = save_dir_path
        self.min_date = min_date
        self.max_date = max_date
        self.min_df = min_df
        self.target = pd.DataFrame()
        self.bert_embedding = pd.DataFrame()
        self._load_bert_embeddings()
        self._load_target()
        # for storing processed tfidf embedding
        self.date_str = f'{str(min_date.date())}_{str(max_date.date())}'
        self.tfidf_embedding_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'tfidf'
            )
        if not os.path.exists(self.tfidf_embedding_dir):
            os.makedirs(self.tfidf_embedding_dir)

    def _load_target(self):
        target = pd.read_parquet(
            os.path.join(self.save_dir_path, 'target_df.parquet.gzip')
            )
        self.target = target.set_index('date')

    def _load_bert_embeddings(self):
        bert_embedding = pd.read_parquet(
            os.path.join(self.save_dir_path, 'sentence_embedding_df.parquet.gzip')
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
        all_paths = get_path(self.reuters_news_path) + get_path(self.bloomberg_news_path)
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
        start_date: datetime,
        end_date: datetime
        ) -> pd.DataFrame:
        """calculate tfidf embedding for given date inputs

        Args:
            tfidf_start_date (datetime): news start date for known date range
            tfidf_end_date (datetime): news end date for known date range
            start_date (Optional[datetime], optional): start date for output. Defaults to None.
            end_date

        Returns:
            Dict[str, np.array]: key of news id and value of news tfidf embeddings
        """
        # training period data
        tfidf_news_files = self._news_file_filter(
            min_date=start_date,
            max_date=end_date
            )
        input_data = embedding_batch_preprocessing(paths=tfidf_news_files)
        tfidf_corpus = [e[2] for e in input_data if e[2] is not None]
        tfidf_news_id = [e[1] for e in input_data if e[2] is not None]

        vectorizer = TfidfVectorizer(
            min_df=self.min_df,
            stop_words=STOPWORDS,
            dtype=np.float32
            )
        X = vectorizer.fit_transform(tfidf_corpus)
        tfidf_df_path = os.path.join(
            self.tfidf_embedding_dir, f'tfidf_{self.date_str}.parquet.gzip'
            )

        # cached processed tfidf df
        if os.path.exists(tfidf_df_path):
            return pd.read_parquet(tfidf_df_path)

        out = tfidf_weighted_embedding(
            x=X,
            trained_vecterizer=vectorizer,
            news_id=tfidf_news_id
            )
        results = self._tfidf_to_df(out)
        # save results to file
        results.to_parquet(tfidf_df_path, compression='gzip')

        return results

    @lazyproperty
    def tfidf_df(self) -> pd.DataFrame:
        """tfidf df for selected news dates
        """
        embedding = self._get_tfidf_vector(
            start_date=self.min_date,
            end_date=self.max_date
            )

        return embedding

    def _get_bert_vector(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        start_date = str(start_date.date())
        end_date = str(end_date.date())

        return self.bert_embedding[start_date: end_date]

    @lazyproperty
    def bert_df(self) -> pd.DataFrame:
        out = self._get_bert_vector(
            start_date=self.min_date,
            end_date=self.max_date
            )

        return out

    def _get_target(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        start_date = str(start_date.date())
        end_date = str(end_date.date())

        return self.target[start_date: end_date]

    @lazyproperty
    def target(self) -> pd.DataFrame:
        out = self._get_target(
            start_date=self.min_date,
            end_date=self.max_date
            )

        return out

    def prep_raw_model_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        embedding_df = self.bert_df.reset_index() \
            .merge(self.tfidf_df, on='news_id', how='inner') \
            .sort_values('date') \
            .set_index('date')

        return embedding_df, self.target

    @lazyproperty
    def trading_date(self) -> List[datetime]:
        return to_date(self.target.index.unique().sort_values())

    @lazyproperty
    def news_date(self):
        return to_date(self.bert_df.index.unique().sort_values())

    @staticmethod
    def _fetch_embeddings(
        trading_dates: Iterable[datetime],
        news_dates: Iterable[datetime],
        embedding_df: pd.DataFrame,
        days_lookback: int = 5
        ) -> Dict[str, List[np.array]]:
        """prepare embedding vectors within the range of days for a given trading date

        Args:
            trading_dates (Iterable[datetime]): a set of trading dates
            news_dates (Iterable[datetime]): a set of news dates
            embedding_df (pd.DataFrame): sentence embedding and tfidf embedding dataframe
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
            sentence_embedding = tmp_df.filter(regex='^c', axis=1).sort_index().groupby('date')
            tfidf_embedding = tmp_df.filter(regex='^f', axis=1).sort_index().groupby('date')
            tmp_bert = []
            sentence_max_news = max([len(e) for e in dict(list(sentence_embedding)).values()])
            for _, d in sentence_embedding:
                # need to pad to the longest number of news in the neighboring days
                d = d.reset_index(drop=True).reindex(range(sentence_max_news), fill_value=0)
                tmp_bert.append(d.to_numpy().tolist())
            tmp_tfidf = []
            for _, d in tfidf_embedding:
                d = d.reset_index(drop=True).reindex(range(sentence_max_news), fill_value=0)
                tmp_tfidf.append(d.to_numpy().tolist())

            out[str(td)] = [np.array(tmp_bert), np.array(tmp_tfidf)]

        return out

    @staticmethod
    def _create_dataset(
        label_df: pd.DataFrame,
        embedding_lookup: Dict[str, List[np.array]],
        batch_size: int = 32
        ) -> PaddedBatchDataset:
        """create training and validation dataset for tf model

        Args:
            label_df (pd.DataFrame): stock price change label
            embedding_lookup (Dict[str, List[np.array, np.array]): date as key, sentence and tfidf pair as value
            batch_size (int, optional): batch size. Defaults to 32.
            seed_value (int, optional): seed for dataset shuffling. Defaults to 42.

        Returns:
            PaddedBatchDataset: dataset
        """
        label = label_df['target'].to_numpy()
        ticker = label_df['ticker'].to_numpy()
        dates = [str(e.date()) for e in label_df.index.to_pydatetime()]
        sentence_embeddings = [embedding_lookup.get(e, [None, None])[0] for e in dates]
        tfidf_embeddings = [embedding_lookup.get(e, [None, None])[1] for e in dates]

        def generator():
            for se, te, t, l in zip(sentence_embeddings, tfidf_embeddings, ticker, label):
                if se is not None:
                    yield (se, te, t), l

        # .shuffle(buffer_size=1000, seed=seed_value)
        dataset = tf.data.Dataset \
            .from_generator(
                generator,
                output_types=((tf.float32, tf.float32, tf.string), tf.int8),
                output_shapes=(((5, None, 384), (5, None, 64), ()), ())
                ) \
            .padded_batch(
                batch_size,
                padded_shapes=(((5, None, 384), (5, None, 64), ()), ())
                )

        return dataset

    def random_split_data(self, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """randomly split dates

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: training and validation target df
        """
        _, target_df = self.prep_raw_model_data()
        # all trading date
        all_trading_dates = target_df.index.unique().to_list()
        # sample training trading dates
        np.random.seed(seed)
        train_dates = np.random.choice(
            all_trading_dates,
            size=int(0.8 * len(all_trading_dates)),
            replace=False
            )

        train_target_mask = target_df.index.isin(train_dates)
        train_target = target_df.loc[train_target_mask, :].sample(frac = 1)
        valid_target = target_df.loc[~train_target_mask, :]
        logger.info(f'training target len: {len(train_target)}')
        logger.info(f'validation target len: {len(valid_target)}')

        return train_target, valid_target

    def create_dataset(
        self,
        days_lookback: int = 5,
        batch_size: int = 32,
        seed_value: int = 42
        ) -> Tuple[PaddedBatchDataset, PaddedBatchDataset]:
        """create modeling dataset

        Args:
            split_method (str): how to split data into training and validation
            days_lookback (int, optional): _description_. Defaults to 5.
            batch_size (int, optional): _description_. Defaults to 32.
            seed (int, optional): _description_. Defaults to 42.

        Returns:
            Tuple[PaddedBatchDataset, PaddedBatchDataset]: training and validation dataset
        """
        embedding_df, _ = self.prep_raw_model_data()

        train_target_random, valid_target_random = self.random_split_data(seed=seed_value)
        embedding_lookup = self._fetch_embeddings(
            trading_dates=self.trading_date,
            news_dates=self.news_date,
            embedding_df=embedding_df,
            days_lookback=days_lookback
            )
        training_dataset = self._create_dataset(
            label_df=train_target_random,
            embedding_lookup=embedding_lookup,
            batch_size=batch_size
            )
        validation_dataset = self._create_dataset(
            label_df=valid_target_random,
            embedding_lookup=embedding_lookup,
            batch_size=batch_size
            )

        return training_dataset, validation_dataset


if __name__ == '__main__':
    dm = DateManager(
        reuters_news_path='/home/timnaka123/Documents/financial-news-dataset/ReutersNews106521',
        bloomberg_news_path='/home/timnaka123/Documents/financial-news-dataset/bloomberg'
        )
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

    label_df = train_target
    embedding_lookup = out


    from src.data.stock_data import get_tickers

    tickers = get_tickers(
        dir_path='/home/timnaka123/Documents/stock_embedding_nlp/src/data/',
        obj_name='qualified_tickers.pickle'
    )
    # use a tuple to create features and label
    f1, f2, labels = (np.random.sample((100,2)), np.random.choice(tickers, 100), np.random.sample((100,1)))
    dataset = tf.data.Dataset.from_tensor_slices(((f1, f2),labels))



