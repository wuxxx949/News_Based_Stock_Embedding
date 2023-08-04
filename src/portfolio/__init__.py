"""make interface for backtest
"""
import click

from src.portfolio.backtest import BackTest

@click.command(help='portfolio backtest')
@click.option('--n', type=int, default=5, help='number of reps')
@click.option('--epochs', type=int, default=40, help='epochs for DL model')
@click.option('--min_n', type=int, default=2, help='min number of training years')
@click.option('--max_n', type=int, default=7, help='max number of training years')
@click.option('--year_lookback', type=int, default=3, help='number of year for avg return')
def run_backtest(
    n: int,
    epochs: int,
    min_n: int,
    max_n: int,
    year_lookback: int
    ) -> None:
    """run backtest pipeline interface

    Args:
        n (int): number of repetitions
        epochs (int): number of epochs in dl training
        min_n (int): min length in years to run
        max_n (int): max length in years to run
        year_lookback (int):  number of years look back for avg return
    """
    if year_lookback < 1:
        raise ValueError('year_lookback must be greater than 0')
    if min_n < 1 or min_n > 7:
        raise ValueError('min_n must be between 1 and 7')
    if max_n < 1 or max_n > 7:
        raise ValueError('max_n must be between 1 and 7')
    if min_n > max_n:
        raise ValueError('min_n cannot be greater than max_n')

    bt = BackTest(n=n, epochs=epochs)
    bt.run_backtest(
        min_n=min_n,
        max_n=max_n,
        year_lookback=year_lookback
        )
