import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from src.data.utils import find_next_traindg_date, shift_to_future_one_year
from src.logger import setup_logger
from src.meta_data import get_meta_data
from src.model.model import extract_ticker_embedding, get_model
from src.model.prepare_training_data import DateManager, ModelDataPrep
from src.portfolio.portfolio_construction import PortfolioConstruction

logger = setup_logger(logger_name='bt', log_file='backtest.log')


class BackTest:
    """test model prediction performance and portfolio return
    """
    def __init__(self, n: int, epochs: int) -> None:
        """constructor

        Args:
            n (int): number of repetitions
            epochs (int): number of epochs
        """
        self.n = n
        self.epochs = epochs
        self.meta_data = get_meta_data()
        self.tickers = pd.read_parquet(
            os.path.join(self.meta_data['SAVE_DIR'], 'target_df.parquet.gzip')
            )['ticker'].unique()
        self.model_history = {}
        self.portfolio_performance = {}
        self.dm = DateManager()
        # expected return in portfolio construction
        self.exp_return = np.arange(0.05, 0.30, 0.01)

    @staticmethod
    def _process_history(
        val_loss: List[float],
        val_accuracy: List[float],
        use_accuracy: bool = True
        ) -> Tuple[float, int]:
        """find the accuracy corresponding to best val loss

        Args:
            val_loss (List[float]): validation loss from training history
            val_accuracy (List[float]): validation accuracy from training history
            use_accuracy (bool): if use accuracy

        Returns:
            Tuple[float, int]: validation accuracy and associated iteration
        """
        if use_accuracy:
            arg_min = np.array(val_accuracy).argmax()
        else:
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
        patience: Optional[int] = None
        ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """run a single training, validation with randomly splitted data

        Args:
            length (int): number of years of data to use
            initial_learning_rate (float): initial learning rate for cosine decay lr scheduler
            alpha (float): alpha for cosine decay lr scheduler
            decay_steps (int): decay steps for cosine decay lr scheduler
            use_es (bool, optional): if use early stop in training. Defaults to False.
            patience (Optional[int], optional): patience for early stop. Defaults to None.

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
                epochs=self.epochs,
                callbacks=[early_stop]
                )
        else:
            hisotry = model.fit(
                training_ds,
                validation_data=validation_ds,
                epochs=self.epochs
                )

        training_loss = hisotry.history['loss']
        val_loss = hisotry.history['val_loss']
        training_accuracy = hisotry.history['accuracy']
        val_accuracy = hisotry.history['val_accuracy']

        return training_loss, val_loss, training_accuracy, val_accuracy

    def run_multiple_training_validation(
        self,
        length: int,
        initial_learning_rate: float,
        alpha: float,
        decay_steps: int,
        use_es: bool = False,
        patience: Optional[int] = None
        ) -> Tuple[List[float], List[int]]:
        """run model on multiple randomly split training and validation datasets

        Args:
            length (int): number of years of data to use
            initial_learning_rate (float): initial learning rate for cosine decay lr scheduler
            alpha (float): alpha for cosine decay lr scheduler
            decay_steps (int): decay steps for cosine decay lr scheduler
            use_es (bool, optional): if use early stop in training. Defaults to False.
            patience (Optional[int], optional): patience for early stop. Defaults to None.

        Returns:
            Tuple[List[float], List[int]]: max validation accuracies and assoicated iterations
        """
        result_max_acc = []
        result_min_iter = []
        logger.info(f"Run multiple training validation using {length} years data {'-' * 20}")
        for i in range(self.n):
            out = self.run_single_training_validation(
                length=length,
                initial_learning_rate=initial_learning_rate,
                alpha=alpha,
                decay_steps=decay_steps,
                use_es=use_es,
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
        training_ds = mdp.create_single_dataset()
        history = model.fit(training_ds, epochs=epochs)

        logger.info(f"training loss using {length} years: {history.history['loss']}")
        logger.info(f"training accuracy using {length} years: {history.history['accuracy']}")

        embedding_dict = extract_ticker_embedding(
            ticker_vec_layer=ticker_layer,
            encoder_embedding_layer=embedding_layer
            )

        return embedding_dict

    def run_data_training(
        self,
        length: int,
        initial_learning_rate: float,
        alpha: float,
        decay_steps: int
        ) -> Dict[str, np.array]:
        """train model using optimal epochs using validation set

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
        model, _, _ = get_model(
            tickers=self.tickers,
            initial_learning_rate=initial_learning_rate,
            alpha=alpha,
            decay_steps=decay_steps
            )

        start_date, end_date = self.dm.get_date_range(data_len=length)
        mdp = ModelDataPrep(min_date=start_date, max_date=end_date)
        training_ds, validation_ds = mdp.create_dataset(batch_size=64, seed_value=None)
        hisotry = model.fit(
            training_ds,
            validation_data=validation_ds,
            epochs=self.epochs
            )

        val_accuracy = hisotry.history['val_accuracy']
        opt_epochs = np.array(val_accuracy).argmax() + 1

        tf.keras.backend.clear_session()
        model1, ticker_layer, embedding_layer = get_model(
            tickers=self.tickers,
            initial_learning_rate=initial_learning_rate,
            alpha=alpha,
            decay_steps=decay_steps
            )

        hisotry1 = model1.fit(
            training_ds,
            validation_data=validation_ds,
            epochs=opt_epochs
            )

        logger.info(f'opt epochs: {opt_epochs}')
        logger.info(f"training loss: {hisotry1.history['loss']}")
        logger.info(f"validatoin loss: {hisotry1.history['val_loss']}")
        logger.info(f"training accuracy: {hisotry1.history['accuracy']}")
        logger.info(f"validatoin accuracy: {hisotry1.history['val_accuracy']}")

        embedding_dict = extract_ticker_embedding(
            ticker_vec_layer=ticker_layer,
            encoder_embedding_layer=embedding_layer
            )

        return embedding_dict

    def run_training_pipeline(self, min_n: int = 3, max_n: int = 4) -> None:
        """train model on various length

        Args:
            min_n (int, optional): min length in years to run. Defaults to 3.
            max_n (int, optional): max length in years to run. Defaults to 4.
        """
        for i in range(min_n, max_n + 1):
            train_out = self.run_multiple_training_validation(
                length=i,
                initial_learning_rate=5e-4,
                alpha=0.5,
                decay_steps=2000
                )
            self.model_history[i] = train_out

    def _make_summary(self) -> None:
        if len(self.portfolio_performance) == 0:
            print('run run_backtest method first')
            raise

        for k, v in self.portfolio_performance.items():
            start_date, end_date = self.dm.get_date_range(data_len=k)
            train_start_date = str(start_date.date())
            train_end_date = str(end_date.date())
            test_start_date = find_next_traindg_date(train_end_date)
            test_end_date = shift_to_future_one_year(test_start_date)

            title = f'training period: {train_start_date}-{train_end_date} \n' + \
                f'test period: {test_start_date}-{test_end_date}'

            actual_return = [d['actual_return'] for d in v]
            expected_return = [round(d['exp_return'], 2) for d in v]
            plt.plot(expected_return, actual_return, marker='o', linestyle='-')
            plt.xlabel('Expected Return')
            plt.ylabel('Actual Return')
            plt.title(title)
            plt.savefig(f'training_length_{k}.png')

    def run_backtest(self, min_n: int = 3, max_n: int = 4) -> None:
        """run backtest for selected length of data

        Args:
            min_n (int, optional): min length in years to run. Defaults to 3.
            max_n (int, optional): max length in years to run. Defaults to 4.
        """
        # determine epochs
        # self.run_training_pipeline(min_n=min_n, max_n=max_n)

        for i in range(min_n, max_n + 1):
            # opt_epoch = int(np.array(self.model_history[i][1]).mean()) + 10
            # ogger.info(f'use {opt_epoch} epochs for {i} training years')
            # opt_epoch = 50
            embeddings = bt.run_data_training(
                length=i,
                # epochs=opt_epoch,
                initial_learning_rate=5e-4,
                alpha=0.5,
                decay_steps=2000
                )

            _, end_date = self.dm.get_date_range(data_len=i)

            pc = PortfolioConstruction(
                embedding_dict=embeddings,
                last_news_date=str(end_date.date())
                )
            tmp_return_lst = []
            for r in self.exp_return:
                tmp_return, tmp_portfolio = pc.get_backtest_results(exp_return=r)
                tmp_return_dict = {}
                tmp_return_dict['exp_return'] = r
                tmp_return_dict['actual_return'] = tmp_return
                tmp_return_dict['weights'] = tmp_portfolio
                tmp_return_lst.append(tmp_return_dict)

            self.portfolio_performance[i] = tmp_return_lst

            # make summary plot and save to folder
            self._make_summary()


if __name__=='__main__':
    bt = BackTest(n=3, epochs=40)

    bt.run_backtest(min_n=3, max_n=3)
    idx = 15
    print(bt.portfolio_performance[3][idx]['exp_return'])
    print(bt.portfolio_performance[3][idx]['actual_return'])
    print(bt.portfolio_performance[3][idx]['weights'])

    # bt.run_training_pipeline()
    # test = bt.run_full_data_training(
    #     length=3,
    #     epochs=10,
    #     initial_learning_rate=5e-4,
    #     alpha=0.5,
    #     decay_steps=2000
    #     )
