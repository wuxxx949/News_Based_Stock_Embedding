import click

from src.data import run_sentence_embedding, run_word_embedding
from src.portfolio import run_backtest

@click.group()
def cli():
    pass

cli.add_command(run_word_embedding)
cli.add_command(run_sentence_embedding)
cli.add_command(run_backtest)
