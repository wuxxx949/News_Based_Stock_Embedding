import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.data.stock_data import get_tickers
from src.model.prepare_training_data import ModelDataPrep

tickers = get_tickers(
    dir_path='/home/timnaka123/Documents/stock_embedding_nlp/src/data/',
    obj_name='qualified_tickers.pickle'
)

ticker_vec_layer = tf.keras.layers.TextVectorization(
    len(tickers), output_sequence_length=1
    )
ticker_vec_layer.adapt(tickers)

# attention
attention_layer = tf.keras.layers.Attention()
class NewsAttention1(tf.keras.layers.Layer):
    def __init__(self, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)

    def call(self, raw_inputs):
        ticker_input = raw_inputs[0]
        context_input = raw_inputs[1]
        word_input = raw_inputs[2]

        market_embedding = []
        batch_size = tf.shape(ticker_input)[0]

        for i in range(5):
            tmp_key = word_input[:, i, :, :]
            tmp_value = context_input[:, i, :, :]
            market_embedding.append(attention_layer([ticker_input, tmp_value, tmp_key]))

        market_embedding = tf.concat(market_embedding, axis=1)
        ndays = tf.shape(market_embedding)[1]
        embedding_size = tf.shape(market_embedding)[2]
        # embedding_size = tf.shape(market_embedding)[2]
        # tf.reshape(market_embedding, [batch_size, ndays, embedding_size])

        return tf.reshape(market_embedding, [batch_size, ndays, embedding_size])

# bert embedding as context vector, which is value of attention
context_embedding = tf.keras.layers.Input(shape=[5, None, 256])
# tfidf embedding as word vector, which is key of attention
word_embedding = tf.keras.layers.Input(shape=[5, None, 64])


# # ticker to embedding
ticker_inputs = tf.keras.layers.Input(shape=[1], dtype=tf.string)
ticker_input_ids = ticker_vec_layer(ticker_inputs)
encoder_embedding_layer = tf.keras.layers.Embedding(len(tickers), 64)
ticker_embeddings = encoder_embedding_layer(ticker_input_ids)

# market layer from attention
market_layer = NewsAttention1()
market_embeddings = market_layer([ticker_embeddings, context_embedding, word_embedding])

# time series layer
ts_layer = tf.keras.layers.Bidirectional(
    tf.keras.layers.GRU(256, return_state=False)
    )
# h is defined in Equation (4)
h = ts_layer(market_embeddings)

# MLP for prediction
output_layer = tf.keras.layers.Dense(1, activation="sigmoid")
Y_proba = output_layer(h)
model = tf.keras.Model(inputs=[context_embedding, word_embedding, ticker_inputs], outputs=[Y_proba])
model.compile(loss="binary_crossentropy", optimizer="nadam",
              metrics=["accuracy"])


if __name__ == '__main__':
    from src.model.prepare_training_data import DateManager

    dm = DateManager(news_path='/home/timnaka123/Documents/financial-news-dataset/ReutersNews106521')
    t_start_date, v_start_date, v_end_date = dm.get_model_date(training_len=3, validation_len=1)

    mdp = ModelDataPrep(
        news_path='/home/timnaka123/Documents/financial-news-dataset/ReutersNews106521',
        save_dir_path='/home/timnaka123/Documents/stock_embedding_nlp/src/data',
        training_start_date=t_start_date,
        validation_start_date=v_start_date,
        validation_end_date=v_end_date
    )

    training_ds, validation_ds = mdp.create_dataset()

    hisotry = model.fit(training_ds, validation_data=validation_ds, epochs=10)
