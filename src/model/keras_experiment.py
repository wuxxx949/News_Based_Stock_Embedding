from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import tensorflow as tf
from src.data.stock_data import get_tickers

tickers = get_tickers(
    dir_path='/home/timnaka123/Documents/stock_embedding_nlp/src/data/',
    obj_name='qualified_tickers.pickle'
)

n = 10000
ticker_col = np.random.choice(tickers, n)
target = np.random.choice([0, 1], n, p=[0.4, 0.6])
dates = np.random.choice(pd.date_range('2018-01-01', '2020-12-31'), n)
max_date  = pd.date_range('2018-01-01', '2020-12-31')[-1]
df = pd.DataFrame({
    'ticker': ticker_col,
    'date': dates,
    'target': target
    })
# def simulate_data(n)

news_embedding = pd.DataFrame(
    np.random.normal(size=(10000, 128)),
    columns=['c' + str(i) for i in range(128) ]
    )

max_news_cnt = 8
def prepare_news_data(news_embedding, n):
    out = []
    # select random number of news
    for _ in range(n):
        tmp_out = []
        for _ in range(5):
            news_cnt = np.random.choice([4,5,6,7,8], 1)
            padding_cnt = max_news_cnt - news_cnt
            padding = np.zeros(shape=(int(padding_cnt), 128)).astype(np.float32)
            tmp_embedding = news_embedding.sample(news_cnt).to_numpy().astype(np.float32)
            tmp_out.append(tf.constant(np.concatenate([tmp_embedding, padding])))

        out.append(tmp_out)

    return np.array(out)

test = prepare_news_data(news_embedding, n=n)


ticker_vec_layer = tf.keras.layers.TextVectorization(
    len(tickers), output_sequence_length=1
    )

ticker_vec_layer.adapt(tickers)

ticker_inputs = tf.keras.layers.Input(shape=[1], dtype=tf.string)
market_inputs = tf.keras.layers.Input(shape=[5, 8, 128])

ticker_input_ids = ticker_vec_layer(ticker_inputs)
encoder_embedding_layer = tf.keras.layers.Embedding(len(tickers), 128)
ticker_embeddings = encoder_embedding_layer(ticker_input_ids)

attention_layer = tf.keras.layers.Attention()


# tf.config.run_functions_eagerly(True)
# https://faroit.com/keras-docs/2.0.1/layers/writing-your-own-keras-layers/
class NewsAttention(tf.keras.layers.Layer):
    def __init__(self, news_embedding, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.news_embedding = news_embedding

    def call(self, raw_inputs):
        inputs = raw_inputs[0]
        market = raw_inputs[1]

        market_embedding = []
        batch_size = tf.shape(inputs)[0]
        # for _ in range(4):
        #     tmp_embedding = self.news_embedding.sample(50)
        #     market_embedding.append(attention_layer([inputs, tmp_embedding]))
        for i in range(5):
            tmp_key = market[:, i, :, :]
            market_embedding.append(attention_layer([inputs, tmp_key]))

        market_embedding = tf.concat(market_embedding, axis=1)
        ndays = tf.shape(market_embedding)[1]
        embedding_size = tf.shape(market_embedding)[2]

        return tf.reshape(market_embedding, [batch_size, ndays, embedding_size])

market_layer = NewsAttention(news_embedding=news_embedding)
merket_embedding = market_layer([ticker_embeddings, market_inputs])


ts_layer = tf.keras.layers.Bidirectional(
    tf.keras.layers.GRU(256, return_state=False)
    )

h = ts_layer(merket_embedding)
output_layer = tf.keras.layers.Dense(1, activation="sigmoid")
Y_proba = output_layer(h)

model = tf.keras.Model(inputs=[ticker_inputs, market_inputs], outputs=[Y_proba])
model.compile(loss="binary_crossentropy", optimizer="nadam",
              metrics=["accuracy"])

X_train = tf.constant(df['ticker'])
Y_train = df['target'].to_numpy()
model.fit([X_train, test], Y_train, epochs=2)

# ------------------------------------------------------------------------------
query = tf.constant(np.random.normal(size=(10, 1, 128)))
key_value = np.random.normal(size=(3, 128)).astype(np.float32)
padding = np.zeros(shape=(2, 128)).astype(np.float32)
value_input = np.concatenate([key_value, padding])
key = tf.constant(value_input)

key_mask = tf.constant([True, True, True, False, False])
query_mask = tf.constant(np.array([True] * 10)[:, np.newaxis])

attention_layer([query, key], mask=[query_mask, key_mask])
attention_layer([query, key])
attention_layer([query, tf.constant(key_value)])

key = tf.constant(np.random.normal(size=(10, 5, 3, 128)))
market_embedding = []
for i in range(5):
    tmp_key = key[:, i, :, :]
    market_embedding.append(attention_layer([query, tmp_key]))

market_embedding = tf.concat(market_embedding, axis=1)



ticker_embeddings = tf.constant(np.random.normal(size=(10, 1, 128)))

market_embedding = []
for _ in range(4):
    tmp_embedding = news_embedding.sample(50)
    market_embedding.append(attention_layer([ticker_embeddings, tmp_embedding]))

tf.reshape(tf.concat(market_embedding, axis=1), (10, 4, 128))