import os
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import tensorflow as tf
from keras.engine.functional import Functional
from keras.layers.core.embedding import Embedding
from keras.layers.preprocessing.text_vectorization import TextVectorization
from tensorflow.keras.callbacks import EarlyStopping

from src.logger import setup_logger

logger = setup_logger(logger_name='model', log_file='model.log')


# attention
class NewsAttention(tf.keras.layers.Layer):
    """News attention layer described in Eq (3)
    """
    def __init__(self, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)

    def call(self, embedding_inputs):
        attention_layer = tf.keras.layers.Attention()
        ticker_embedding = embedding_inputs[0]
        context_embedding = embedding_inputs[1]
        tfidf_embedding = embedding_inputs[2]

        market_embedding = []
        batch_size = tf.shape(ticker_embedding)[0]

        for i in range(5):
            tmp_key = tfidf_embedding[:, i, :, :]
            tmp_value = context_embedding[:, i, :, :]
            market_embedding.append(attention_layer([ticker_embedding, tmp_value, tmp_key]))

        market_embedding = tf.concat(market_embedding, axis=1)
        ndays = tf.shape(market_embedding)[1]
        embedding_size = tf.shape(market_embedding)[2]

        return tf.reshape(market_embedding, [batch_size, ndays, embedding_size])


def get_model(
    tickers: List[str],
    learning_rate: float = 0.0005
    ) -> Tuple[Functional, TextVectorization, Embedding]:
    """model architecture

    Args:
        tickers (List[str]): unique tickers
        learning_rate (float, optional): learning rate for Nadam. Defaults to 0.0005.

    Returns:
        Functional: model object
    """
    # map ticker string to int, +2 for reserved int
    ticker_vec_layer = tf.keras.layers.TextVectorization(
        len(tickers) + 2, output_sequence_length=1
        )
    ticker_vec_layer.adapt(tickers)

    # sentence embedding as context vector, which is value of attention
    context_embedding = tf.keras.layers.Input(shape=[5, None, 384])
    # tfidf embedding as word vector, which is key of attention
    word_embedding = tf.keras.layers.Input(shape=[5, None, 64])

    # ticker to embedding
    ticker_inputs = tf.keras.layers.Input(shape=[1], dtype=tf.string)
    ticker_input_ids = ticker_vec_layer(ticker_inputs)
    encoder_embedding_layer = tf.keras.layers.Embedding(len(tickers) + 2, 64)
    ticker_embeddings = encoder_embedding_layer(ticker_input_ids)

    # market layer from attention
    market_layer = NewsAttention()
    market_embeddings = market_layer([ticker_embeddings, context_embedding, word_embedding])

    # time series layer with optional temporal attention
    ts_layer = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(64, return_sequences=True) # return sequences as h_tau
        )
    # h is defined in Equation (4)
    htau = ts_layer(market_embeddings)
    # use embedding vector as v
    v_layer = tf.keras.layers.Embedding(htau.shape[1], htau.shape[2])
    v_vec = v_layer(tf.constant(range(htau.shape[1]), dtype=tf.int8))
    # v_{tau-t} * h_{tau}^O terms
    vh = [tf.matmul(htau[:, i, :], tf.transpose(tf.expand_dims(v_vec[i], axis=0))) for i in range(5)]
    tensors_stacked = tf.stack(vh, axis=1) # shape=(None, 5, 1)
    beta_tau = tf.nn.softmax(tensors_stacked, axis=1) # shape=(None, 5, 1)
    ho= tf.reduce_sum(htau * beta_tau, axis=1)  # shape: (None, 256)

    # MLP for prediction
    output_layer = tf.keras.layers.Dense(1, activation="sigmoid")
    Y_proba = output_layer(ho)

    model = tf.keras.Model(
        inputs=[context_embedding, word_embedding, ticker_inputs],
        outputs=[Y_proba]
        )

    model.compile(
        loss='binary_crossentropy',
        optimizer= tf.keras.optimizers.Nadam(learning_rate=learning_rate),
        metrics=['accuracy']
        )

    return model, ticker_vec_layer, encoder_embedding_layer


def extract_ticker_embedding(
    ticker_vec_layer: TextVectorization,
    encoder_embedding_layer: Embedding
    ) -> Dict[str, np.array]:
    """fetch trained ticker embeddings

    Args:
        ticker_vec_layer (TextVectorization): an adapted ticker int lookup layer
        encoder_embedding_layer (Embedding): a trained ticker embeddings lookup layer

    Returns:
        Dict[str, np.array]: _description_
    """
    tickers = ticker_vec_layer.get_vocabulary()[2:]
    out = {}
    for t in tickers:
        out[t] = encoder_embedding_layer(ticker_vec([t])[0])[0].numpy()

    return out


if __name__ == '__main__':
    from src.data.utils import pickle_results
    from src.model.prepare_training_data import DateManager, ModelDataPrep
    from src.meta_data import get_meta_data
    tickers = pd.read_parquet(
        os.path.join(get_meta_data()['SAVE_DIR'], 'target_df.parquet.gzip')
        )['ticker'].unique()

    model, ticker_vec, ticker_embedding = get_model(tickers=tickers)

    dm = DateManager()

    start_date, end_date = dm.get_date_range(data_len=7)

    mdp = ModelDataPrep(
        min_date=start_date,
        max_date=end_date,
        min_df=0.0001
        )
    training_ds, validation_ds = mdp.create_dataset(seed_value=28, batch_size=64)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5)

    hisotry = model.fit(
        training_ds,
        validation_data=validation_ds,
        epochs=20,
        callbacks=[early_stop]
        )

    # log model history
    logger.info(f"training loss: {hisotry.history['loss']}")
    logger.info(f"validatoin loss: {hisotry.history['val_loss']}")
    logger.info(f"training accuracy: {hisotry.history['accuracy']}")
    logger.info(f"validatoin accuracy: {hisotry.history['val_accuracy']}")

    # extract embeddings
    # TODO: make a function
    # out = {}y