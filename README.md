## Introduction
This repo is implemented according to the methodology described in paper [_Stock Embeddings Acquired from News Articles and Price History, and an Application to Portfolio Optimization_](https://aclanthology.org/2020.acl-main.307/).

The goal is to construct embeddings for a set of selected US stocks, which represents a given ticker by a numeric vector. The concept of embedding is popular in Natural Language Processing (NLP) and is the fundamental building block for any Large.  Language Model (LLM).

Once a set of vector representations is learnt for selected stocks, we can use the learnt embeddings for any downstream task. Follow the paper, a quartratic programming portfolio optimization is implemented using the stock covariance matrix based on the embeddings.

## Data
There are two types of data used in this repo.
- News data from Reuters and Bloomberg from 2006 to 2013. The data from original [repo](https://github.com/philipperemy/financial-news-dataset) is no longer available, but you can email the author for the dataset for your own research.
    > However, if you have a request about it, send it to me at premy.enseirb@gmail.com and put the words "bloomberg dataset" in your email body.

    Note that the Reuters data shared above only has title, so I only used the Bloomberg data.
- For complete Reuters data, you can find [here](https://github.com/HanssonMagnus/financial-news-dataset).
- Stock price data is fetched based on the popular Python package `yfinance`.

## Methodology
### News Preprocessing
The paper used a dual-vector representation of news article texts, namely, the TFIDF-weighted word embedding and BERT encoder for news title. In this repo, I used `Fasttext` to train word embeddings based on entire news article collections f or TFIDF-weighted embedding and used `sentence_transformers` package instead to encode the article title.

The target variable the daily stock price movement. The daily log return > 0.0068 is labeled as 1 and daily log return < -0.0059 is labeled as 0.


### Model
The stock price movement prediction is considered a very difficult problem, if we only include news article strictly before the price date. However, if we include the news for the same price date, this problem is more reasonable as the goal is to learn the embedding vector rather than making a good stock price movement preditive model. Therefore, the news of same price date and 4 days prior (total 5 days of news) are included to 'predict' the stock movement.

Motivated by the attention mechanism, for day $t$, the input data is a pair of 'key' and 'value' news embeddings $(n_i^K, n_i^V)$. The paper denoted the collection of the TFIDF-weighted news embeddings by $N_t^K=\lbrace n_i^K\rbrace_t$ and sentence embeddings by $N_t^V=\lbrace n_i^V\rbrace_t$. The model first calculate the attention score of stock $i$ and news $j$ as $\text{score}_{i,j} = n_i^K \cdot s_j$, where $s_j$ is the stock embedding for stock $j$.

The attention to the sentence embedding is defined as
$$\alpha_i^j=\frac{exp({score}_{i,j})}{\sum_{l}exp({score}_{l,j})}.$$

$$\frac{exp({score}_{i,j})}{\sum_{l}exp({score}_{l,j})}.$$


Finally, the market vector for stock $j$ on day $t$ is
$$m_t^j = \sum_{n_i^V\in N_t^V} \alpha_i^j n_i^V$$


To predict the stock movement, we need the most recent 5 days market vectror collection $M^j_{[t-4, t]}=[m^j_{t-4}, m^j_{t-3}, \dots, m^j_{t}]$. The model used a Bi-GRU layer and a MLP layer to estimate the positive movement probability $\hat y_t^j$, i.e., $h_t^O = \text{GRU}(M^j_{[t-4, t]})$, and $\hat y_t^j = \sigma(\text{MLP}(h^O-t))$.

The optional temporal re-weighting method is also implemented.

![Drag Racing](image/classifier.png)

## Backtest

## Portfolio Optimization

## Run Pipeline