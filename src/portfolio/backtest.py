import os

import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from src.meta_data import get_meta_data
from src.model.model import extract_ticker_embedding, get_model
from src.model.prepare_training_data import DateManager, ModelDataPrep
from src.portfolio.portfolio_construction import PortfolioConstruction


class BackTest:
    def __init__(self, n_rep: int) -> None:
        """constructor

        Args:
            n_rep (int): number of random split for backtesting
        """
        self.meta_data = get_meta_data()
        self.tickers = pd.read_parquet(os.path.join(self.meta_data['SAVE_DIR'], 'target_df.parquet.gzip'))
        self.model_history = {}
        self.portfolio_performance = {}

    @staticmethod
    def _process_history():
        pass

    def run_single_training_validation(self, length: int):
        tf.keras.backend.clear_session()
        model, _, _ = get_model(tickers=self.tickers)

        dm = DateManager()
        start_date, end_date = dm.get_date_range(data_len=length)
        mdp = ModelDataPrep(min_date=start_date, max_date=end_date)
        training_ds, validation_ds = mdp.create_dataset(batch_size=64, seed_value=None)
        early_stop = EarlyStopping(monitor='val_accuracy', patience=2)
        hisotry = model.fit(
            training_ds,
            validation_data=validation_ds,
            epochs=1,
            callbacks=[early_stop]
            )