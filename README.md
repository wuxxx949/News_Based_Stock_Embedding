## Introduction
This repo is implemented according to the methodology described in paper [_Stock Embeddings Acquired from News Articles and Price History, and an Application to Portfolio Optimization_](https://aclanthology.org/2020.acl-main.307/).

The goal is to construct embeddings for a set of selected US stocks, which represents a given ticker by a numeric vector. The concept of embedding is popular in Natural Language Processing (NLP) and is the fundamental building block for any Large.  Language Model (LLM).

Once a set of vector representations is learnt for selected stocks, we can use the learnt embeddings for any downstream task. Follow the paper, a quartratic programming portfolio optimization is implemented using the stock covariance matrix based on the embeddings.

## Data
There are two types of data used in this repo.
- News data from Reuters and Bloomberg from 2006 to 2013. The data from original [repo](https://github.com/philipperemy/financial-news-dataset) is no longer available, but you can email the author for the dataset for your own research.
    > However, if you have a request about it, send it to me at premy.enseirb@gmail.com and put the words "bloomberg dataset" in your email body.

- Stock price data is fetched based on the popular Python package `yfinance`.