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
The paper uses a dual-vector representation of news article texts, namely, the TFIDF-weighted word embedding and BERT encoder for news title. In this repo, I used `Fasttext` to train word embeddings based on entire news article collections and used `sentence_transformers` package with model `all-MiniLM-L6-v2` to encode the article title.

### Model
The stock price movement prediction is considered a very difficult problem, if we only include news article strictly before the price date. However, if we include the news for the same price date, this problem is more reasonable as the goal is to learn the embedding vector rather than making a good stock price movement preditive model. Therefore, the news of same price date and 4 days prior (total 5 days of news) are included to 'predict' the stock movement.

The daily log return > 0.0068 is labeled as 1 and daily log return < -0.0059 is labeled as 0.

![Drag Racing](image/classifier.png)