import os

import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.optimizers.schedules import CosineDecay

from src.data.stock_data import get_tickers
from src.logger import setup_logger
from src.model.prepare_training_data import ModelDataPrep

logger = setup_logger(logger_name='model', log_file='model.log')


tickers = pd.read_parquet(
    os.path.join('/home/timnaka123/Documents/stock_embedding_nlp/src/data/', 'target_df.parquet.gzip')
    )['ticker'].unique()

ticker_vec_layer = tf.keras.layers.TextVectorization(
    len(tickers) + 2, output_sequence_length=1
    )
ticker_vec_layer.adapt(tickers)

# attention
attention_layer = tf.keras.layers.Attention()
class NewsAttention(tf.keras.layers.Layer):
    def __init__(self, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)

    def call(self, embedding_inputs):
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

# Adam optimizer with cosine annealing
# initial_learning_rate = 0.01
# lr_schedule = CosineDecay(initial_learning_rate, decay_steps=2000, alpha=0.001)
# optimizer = Adam(learning_rate=lr_schedule)

early_stop = EarlyStopping(monitor='val_accuracy', patience=2)

model.compile(
    loss='binary_crossentropy',
    optimizer= tf.keras.optimizers.Nadam(learning_rate=0.0005),
    # optimizer=optimizer,
    metrics=['accuracy']
    )


if __name__ == '__main__':
    from src.data.utils import pickle_results
    from src.model.prepare_training_data import DateManager


    dm = DateManager(
        reuters_news_path='/home/timnaka123/Documents/financial-news-dataset/ReutersNews106521',
        bloomberg_news_path='/home/timnaka123/Documents/financial-news-dataset/bloomberg'
        )

    start_date, end_date = dm.get_random_split_date(data_len=7)

    mdp = ModelDataPrep(
        reuters_news_path='/home/timnaka123/Documents/financial-news-dataset/ReutersNews106521',
        bloomberg_news_path='/home/timnaka123/Documents/financial-news-dataset/bloomberg',
        save_dir_path='/home/timnaka123/Documents/stock_embedding_nlp/src/data',
        min_date=start_date,
        max_date=end_date
    )

    training_ds, validation_ds = mdp.create_dataset(seed_value=41, batch_size=64)
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
    out = {}
    for t in tickers:
        out[t] = encoder_embedding_layer(ticker_vec_layer([t])[0])[0].numpy()

    pickle_results(
        dir='/home/timnaka123/Documents/stock_embedding_nlp/src/model/',
        name='embedding.pickle',
        obj=out
        )

