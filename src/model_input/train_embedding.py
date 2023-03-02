import os

import fasttext


def train_word_embedding(file: str, dim: int=64, **kargs):
    # https://fasttext.cc/docs/en/python-module.html#train_unsupervised-parameters
    model = fasttext.train_unsupervised(file, dim=dim, **kargs)
    # model.get_word_vector
    dir = os.path.dirname(os.path.realpath(__file__))
    model.save_model(os.path.join(dir, 'news.bin'))

    return model

# from transformers import BertTokenizer, TFBertModel
# tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
# model = TFBertModel.from_pretrained("bert-large-uncased")
# text = "you like orange"
# text1 = "I like pizza and going to beach"
# text2 = 'This is something totally different'
# text3 = 'Attention is all you need'
# text4 = 'what if I add more junks to the batch'
# encoded_input = tokenizer((text, text1, text2, text3, text4), return_tensors='tf', padding=True)
# output = model(encoded_input)
# # sentence embedding
# embedding = output.pooler_output
# out1 = embedding.numpy()[1]


# text2 = "you like orange. [SEP] I like pizza"
# text21 = "you like orange. I like pizza"
# text3 = 'I like pizza and going to beach'
# encoded_input = tokenizer(text3, return_tensors='tf')
# output = model(encoded_input)
# embedding2 = output.pooler_output
# out2 = embedding2.numpy()[0]

if __name__ == '__main__':
    embedding_model = train_word_embedding(
        file='/home/timnaka123/Documents/stock_embedding_nlp/src/data/news_training_corpus.txt',
        epoch=7
        )
