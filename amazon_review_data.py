import torch
from torchtext import data
import numpy as np
from sklearn.metrics import accuracy_score
import spacy
import pandas as pd
from torchtext.vocab import Vectors

class AmazonReviewData(object):
    def __init__(self, config):
        self.config = config
        self.vocab = []
        self.word_embeddings = {}
        self.train_iterator = None
        self.test_iterator = None
        self.val_iterator = None

    
    def load_all_data(self, embedding_file, train_df, test_df, NLP):
        '''
        Load data into 
        train, validation, and test iterators
        '''        
        
        # Spacy allows us to convert text into single words
        tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]
        
        # Creating Field for data
        REVIEW_TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config["max_sen_len"])
        RATING = data.Field(sequential=False, use_vocab=False)
        PRODUCT_CATEGORY = data.Field(sequential=False, use_vocab=False)
        LENGTH = data.Field(sequential=False, use_vocab=False)
        EXCLAMATIONS = data.Field(sequential=False, use_vocab=False)
        TITLE_SENTIMENT = data.Field(sequential=False, use_vocab=False)
        TEXT_SENTIMENT = data.Field(sequential=False, use_vocab=False)
        
        LABEL = data.Field(sequential=False, use_vocab=False)
        
        datafields = [("REVIEW_TEXT", REVIEW_TEXT),
                      ("RATING", RATING), 
                      ("PRODUCT_CATEGORY", PRODUCT_CATEGORY), 
                      ("LENGTH", LENGTH), 
                      ("EXCLAMATIONS", EXCLAMATIONS),
                      ("TITLE_SENTIMENT", TITLE_SENTIMENT),
                      ("TEXT_SENTIMENT", TEXT_SENTIMENT),
                      ("LABEL",LABEL)]

        # Load data from pd.DataFrame into torchtext.data.Dataset
        # train_df = self.get_pandas_df(train_file)
        train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        train_data = data.Dataset(train_examples, datafields)
        
        # test_df = self.get_pandas_df(test_file)
        test_examples = [data.Example.fromlist(i, datafields) for i in test_df.values.tolist()]
        test_data = data.Dataset(test_examples, datafields)
        
        # If validation file exists, load it. Otherwise get validation data from training data

        train_data, val_data = train_data.split(split_ratio=0.8)
        
        REVIEW_TEXT.build_vocab(train_data, vectors=Vectors(embedding_file,  cache="./"))
        self.word_embeddings = REVIEW_TEXT.vocab.vectors
        self.vocab = REVIEW_TEXT.vocab
        
        self.train_iterator = data.BucketIterator(
            (train_data),
            batch_size=self.config["batch_size"],
            sort_key=lambda x: len(x.REVIEW_TEXT),
            repeat=False,
            shuffle=True)
        
        self.val_iterator, self.test_iterator = data.BucketIterator.splits(
            (val_data, test_data),
            batch_size=self.config["batch_size"],
            sort_key=lambda x: len(x.REVIEW_TEXT),
            repeat=False,
            shuffle=False)

    def load_test_data(self, embedding_file, test_df, NLP):
        '''
        Load data into 
        test iterators
        '''        
        
        # Spacy allows us to convert text into single words
        tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]
        
        # Creating Field for data
        REVIEW_TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config["max_sen_len"])
        RATING = data.Field(sequential=False, use_vocab=False)
        PRODUCT_CATEGORY = data.Field(sequential=False, use_vocab=False)
        LENGTH = data.Field(sequential=False, use_vocab=False)
        EXCLAMATIONS = data.Field(sequential=False, use_vocab=False)
        TITLE_SENTIMENT = data.Field(sequential=False, use_vocab=False)
        TEXT_SENTIMENT = data.Field(sequential=False, use_vocab=False)
        
        LABEL = data.Field(sequential=False, use_vocab=False)
        
        datafields = [("REVIEW_TEXT",REVIEW_TEXT),
                      ("RATING", RATING), 
                      ("PRODUCT_CATEGORY", PRODUCT_CATEGORY), 
                      ("LENGTH", LENGTH), 
                      ("EXCLAMATIONS", EXCLAMATIONS),
                      ("TITLE_SENTIMENT", TITLE_SENTIMENT),
                      ("TEXT_SENTIMENT", TEXT_SENTIMENT),
                      ("LABEL",LABEL)]
        
        test_examples = [data.Example.fromlist(i, datafields) for i in test_df.values.tolist()]
        test_data = data.Dataset(test_examples, datafields)
                
        REVIEW_TEXT.build_vocab(test_data, vectors=Vectors(embedding_file,  cache="./"))
        self.word_embeddings = REVIEW_TEXT.vocab.vectors
        self.vocab = REVIEW_TEXT.vocab
        
        self.test_iterator = data.BucketIterator(
            (test_data),
            batch_size=self.config["batch_size"],
            sort_key=lambda x: len(x.REVIEW_TEXT),
            repeat=False,
            shuffle=False)

