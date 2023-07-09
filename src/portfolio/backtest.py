import os
from typing import Dict, List, Tuple

import numpy as np
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
        self.dm = DateManager()


    @staticmethod
    def _process_history():
        pass

    def run_single_training_validation(
        self,
        length: int,
        initial_learning_rate: float,
        alpha: float,
        decay_steps: int
        ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """run a single training, validation with randomly splitted data

        Args:
            length (int): number of years of data to use
            initial_learning_rate (float): initial learning rate for cosine decay lr scheduler
            alpha (float): alpha for cosine decay lr scheduler
            decay_steps (int): decay steps for cosine decay lr scheduler

        Returns:
            Tuple[List[float], List[float], List[float], List[float]]: training loss,
            validation loss, training accuracy, and validation accuracy
        """
        tf.keras.backend.clear_session()
        model, _, _ = get_model(
            tickers=self.tickers,
            initial_learning_rate=initial_learning_rate,
            alpha=alpha,
            decay_steps=decay_steps
            )

        start_date, end_date = self.dm.get_date_range(data_len=length)
        mdp = ModelDataPrep(min_date=start_date, max_date=end_date)
        training_ds, validation_ds = mdp.create_dataset(batch_size=64, seed_value=None)
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            start_from_epoch=5
            )
        hisotry = model.fit(
            training_ds,
            validation_data=validation_ds,
            epochs=40,
            callbacks=[early_stop]
            )
        training_loss = hisotry.history['loss']
        val_loss = hisotry.history['val_loss']
        training_accuracy = hisotry.history['accuracy']
        val_accuracy = hisotry.history['val_accuracy']

        return training_loss, val_loss, training_accuracy, val_accuracy

    def run_full_data_training(
        self,
        length: int,
        epochs: int,
        initial_learning_rate: float,
        alpha: float,
        decay_steps: int        ) -> Dict[str, np.array]:
        """train model on full dataset and extract stock embeddings

        Args:
            length (int): number of years of data to use
            epochs (int): number of epochs
            initial_learning_rate (float): initial learning rate for cosine decay lr scheduler
            alpha (float): alpha for cosine decay lr scheduler
            decay_steps (int): decay steps for cosine decay lr scheduler

        Returns:
            Dict[str, np.array]: ticker as key and trained embeddings as value
        """
        tf.keras.backend.clear_session()
        model, ticker_layer, embedding_layer = get_model(
            tickers=self.tickers,
            initial_learning_rate=initial_learning_rate,
            alpha=alpha,
            decay_steps=decay_steps
            )

        start_date, end_date = self.dm.get_date_range(data_len=length)
        mdp = ModelDataPrep(min_date=start_date, max_date=end_date)
        training_ds = mdp.create_single_dataset(batch_size=64)
        _ = model.fit(training_ds, epochs=epochs)

        embedding_dict = extract_ticker_embedding(
            ticker_vec_layer=ticker_layer,
            encoder_embedding_layer=embedding_layer)

        return embedding_dict

