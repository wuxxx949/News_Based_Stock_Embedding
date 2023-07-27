"""make interface for backtest
"""
import click

from src.portfolio.backtest import BackTest

@click.option('--n', default=5, help='number of reps')
@click.option('--epochs', default=40, help='epochs for DL model')
@click.option('--min_n', default=2, help='min number of training years')
@click.option('--max_n', default=7, help='max number of training years')
def run_backtest(
    n: int,
    epochs: int,
    min_n: int,
    max_n: int,
    year_lookback
    ) -> None:
    """run backtest pipeline interface

    Args:
        n (int): number of repetitions
        epochs (int): number of epochs in dl training
        min_n (int): _description_
        max_n (int): _description_
        year_lookback (_type_): _description_
    """
    bt = BackTest(n=n, epochs=epochs)
    bt.run_backtest(
        min_n=min_n,
        max_n=max_n,
        year_lookback=year_lookback
        )
