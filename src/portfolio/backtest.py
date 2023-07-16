import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from src.logger import setup_logger
from src.meta_data import get_meta_data
from src.model.model import extract_ticker_embedding, get_model
from src.model.prepare_training_data import DateManager, ModelDataPrep
from src.portfolio.portfolio_construction import PortfolioConstruction

logger = setup_logger(logger_name='bt', log_file='backtest.log')


class BackTest:
    """test model prediction performance and portfolio return
    """
    def __init__(self) -> None:
        """constructor
        """
        self.meta_data = get_meta_data()
        self.tickers = pd.read_parquet(
            os.path.join(self.meta_data['SAVE_DIR'], 'target_df.parquet.gzip')
            )['ticker'].unique()
        self.model_history = {}
        self.portfolio_performance = {}
        self.dm = DateManager()

    @staticmethod
    def _process_history(
        val_loss: List[float],
        val_accuracy: List[float]
        ) -> Tuple[float, int]:
        """find the accuracy corresponding to best val loss

        Args:
            val_loss (List[float]): validation loss from training history
            val_accuracy (List[float]): validation accuracy from training history

        Returns:
            Tuple[float, int]: validation accuracy and associated iteration
        """
        arg_min = np.array(val_loss).argmin()
        val_acc = val_accuracy[arg_min]

        return val_acc, arg_min

    def run_single_training_validation(
        self,
        length: int,
        initial_learning_rate: float,
        alpha: float,
        decay_steps: int,
        use_es: bool = False,
        patience: int = 5
        ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """run a single training, validation with randomly splitted data

        Args:
            length (int): number of years of data to use
            initial_learning_rate (float): initial learning rate for cosine decay lr scheduler
            alpha (float): alpha for cosine decay lr scheduler
            decay_steps (int): decay steps for cosine decay lr scheduler
            use_es (bool): if use early stop in training
            patience (int): patience for early stop

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
        if use_es:
            early_stop = EarlyStopping(
                monitor='val_accuracy',
                patience=patience
                )
            hisotry = model.fit(
                training_ds,
                validation_data=validation_ds,
                epochs=40,
                callbacks=[early_stop]
                )
        else:
            hisotry = model.fit(
                training_ds,
                validation_data=validation_ds,
                epochs=40
                )

        training_loss = hisotry.history['loss']
        val_loss = hisotry.history['val_loss']
        training_accuracy = hisotry.history['accuracy']
        val_accuracy = hisotry.history['val_accuracy']

        return training_loss, val_loss, training_accuracy, val_accuracy

    def run_multiple_training_validation(
        self,
        n: int,
        length: int,
        initial_learning_rate: float,
        alpha: float,
        decay_steps: int,
        patience: int
        ) -> Tuple[List[float], List[int]]:
        """run model on multiple randomly split training and validation datasets

        Args:
            n (int): number of repetitions
            length (int): number of years of data to use
            initial_learning_rate (float): initial learning rate for cosine decay lr scheduler
            alpha (float): alpha for cosine decay lr scheduler
            decay_steps (int): decay steps for cosine decay lr scheduler
            patience (int): patience for early stop

        Returns:
            Tuple[List[float], List[int]]: max validation accuracies and assoicated iterations
        """
        result_max_acc = []
        result_min_iter = []
        logger.info(f"Run multiple training validation using {length} years data {'-' * 20}")
        for i in range(n):
            out = self.run_single_training_validation(
                length=length,
                initial_learning_rate=initial_learning_rate,
                alpha=alpha,
                decay_steps=decay_steps,
                patience=patience
                )

            val_acc, arg_min = self._process_history(val_loss=out[1], val_accuracy=out[3])
            result_max_acc.append(val_acc)
            result_min_iter.append(arg_min)

            logger.info(f"training loss on iter {i}: {out[0]}")
            logger.info(f"validatoin loss on iter {i}: {out[1]}")
            logger.info(f"training accuracy on iter {i}: {out[2]}")
            logger.info(f"validatoin accuracy on iter {i}: {out[3]}")

        return result_max_acc, result_min_iter

    def run_full_data_training(
        self,
        length: int,
        epochs: int,
        initial_learning_rate: float,
        alpha: float,
        decay_steps: int
        ) -> Dict[str, np.array]:
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

    def run_training_pipeline(self) -> None:
        """train model on various length
        """
        for i in range(4, 5):
            train_out = self.run_multiple_training_validation(
                n=3,
                length=i,
                initial_learning_rate=5e-4,
                alpha=0.3,
                decay_steps=2000,
                patience=10
                )
            self.model_history[i] = train_out



if __name__=='__main__':
    bt = BackTest()
    bt.run_training_pipeline()