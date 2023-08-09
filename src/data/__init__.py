import click

from src.data.bert_embedding import get_news_path
from src.data.sentence_embedding import generate_embedding
from src.data.train_fasttext_embeddings import (news_preprocessing,
                                                train_word_embedding)


@click.command(help='faxttext embedding training')
@click.option('--epoch', type=int, default=10, help='number of epochs')
def run_word_embedding(epoch: int) -> None:
    """train W2V embedding using fasttext

    Args:
        epoch (int): number of epochs
    """
    print("============== Starting: faxttext embedding training ============")
    news_preprocessing()
    train_word_embedding(epoch=epoch)
    print("============== Ending: faxttext embedding training ==============")


@click.command(help='make sentence embedding')
@click.option('--batch_size', type=int, default=1000, help='batch size')
def run_sentence_embedding(batch_size: int) -> None:
    """run sentence embedding using sentence_transformers

    Args:
        batch_size (int): number of news article headlines
    """
    print("============== Starting: make sentence embedding ================")
    generate_embedding(all_paths=get_news_path(), batch_size=batch_size)
    print("============== Ending: make sentence embedding ==================")

