import torch.optim as optim
from torch import nn
import torch
import spacy
import pandas as pd
import re
import numpy as np

# a list of indexes for the train and test set 
from test_set_indexes import train_set_indexes, test_set_indexes
from sklearn.metrics import accuracy_score
from amazon_review_data import AmazonReviewData
from fake_review_detector import *

from textblob import TextBlob



def get_numeric_label(label):
    '''
    Labels look like th following in data:
    __label1__

    __label2__

    We need to grab the integer from the string

    __label1__ -> 1
    __label2__ -> 2
    
    __label1__ is REAL REVIEW
    __label2__ is FAKE REVIEW
    '''
    return int(label[-3])

def remove_html_tags(text):
    return re.sub('<[^<]+?>', '', text)

# ASSIGN EACH PRODUCT CATEGORY A UNIQUE INTEGER. THIS 
# WILL BE FED INTO THE NEURAL NETWORK AS A FEATURE. 
PRODUCT_CATEGORY_TO_INTEGER = {'Kitchen': 0, 
                                'Home': 1, 
                                'Grocery': 2, 
                                'Sports': 3, 
                                'Jewelry': 4, 
                                'Home Entertainment': 5, 
                                'Video DVD': 6, 
                                'Books': 7, 
                                'Shoes': 8, 
                                'PC': 9, 
                                'Furniture': 10, 
                                'Video Games': 11, 
                                'Camera': 12, 
                                'Watches': 13, 
                                'Electronics': 14, 
                                'Office Products': 15, 
                                'Health & Personal Care': 16, 
                                'Pet Products': 17, 
                                'Baby': 18, 
                                'Outdoors': 19, 
                                'Toys': 20, 
                                'Musical Instruments': 21, 
                                'Wireless': 22, 
                                'Luggage': 23, 
                                'Apparel': 24, 
                                'Lawn and Garden': 25, 
                                'Automotive': 26, 
                                'Tools': 27, 
                                'Beauty': 28, 
                                'Home Improvement': 29}
def parse_label( label):
    '''
    Get the actual labels from label string
    Input:
        label (string) : labels of the form '__label__2'
    Returns:
        label (int) : integer value corresponding to label string
    '''
    return int(label.strip()[-1])

def evaluate_model(model, iterator, prev_all_preds=None):
    '''
    Evaluate model using data 
    '''
    all_preds = []
    all_y = []
    for idx,batch in enumerate(iterator):
        text = batch.REVIEW_TEXT
        product_cat = batch.PRODUCT_CATEGORY
        rating = batch.RATING
        length = batch.LENGTH
        exclamations = batch.EXCLAMATIONS
        title_sent = batch.TITLE_SENTIMENT
        text_sent = batch.TEXT_SENTIMENT
        
        
        x = (text, product_cat, rating, length, exclamations, title_sent, text_sent)
        y_pred = model(x)
        
        predicted = torch.max(y_pred.data, 1)[1] + 1
        all_preds.extend(predicted.cpu().numpy())
        all_y.extend(batch.LABEL.cpu().numpy())
    
    all_preds = np.array(all_preds).flatten()
    acc = accuracy_score(all_y, all_preds)
    print("accuracy", acc)

    return acc

def get_pandas_df(filename):
    '''
    Load the data into Pandas.DataFrame object
    This will be used to convert data to torchtext object
    '''
    with open(filename, 'r') as datafile:     
        data = [line.strip().split(',', maxsplit=1) for line in datafile]
        data_text = list(map(lambda x: x[1], data))
        data_label = list(map(lambda x: parse_label(x[0]), data))
    full_df = pd.DataFrame({"REVIEW_TEXT":data_text, "LABEL":data_label})
    return full_df
    
if __name__=='__main__':
    
    config = {
        "embed_size" : 50,
        "hidden_size" : 20, 
        "output_size" : 2, # 2 LABELS
        "epochs" : 25000,
        "lr" : 0.001, 
        "batch_size" : 64,
        "max_sen_len" : 90, 
    }


    # Check if spacy model is installed and if not, install it. 
    if(not spacy.util.is_package("en_core_web_md")):
        print("Spacy model not detected. Will install")
        spacy.cli.download("en_core_web_md")
    
    
    # Load Spacy NLP library to help with tokenization (converting text into words)
    NLP = spacy.load('en_core_web_md')    
    
    # PARSE RAW DATA
    data_df = pd.read_csv("amazon_reviews.txt", delimiter = "\t")
    data_df["LABEL"] =  data_df["LABEL"].apply(get_numeric_label)
    data_df["PRODUCT_CATEGORY"] =  data_df["PRODUCT_CATEGORY"].apply(lambda x: PRODUCT_CATEGORY_TO_INTEGER[x])
    data_df["REVIEW_TEXT"] = data_df["REVIEW_TEXT"].apply(remove_html_tags)
    data_df["REVIEW_TITLE"] = data_df["REVIEW_TITLE"].apply(remove_html_tags)
    
    # COMBINE REVIEW TITLE WITH TEXT
    data_df["ALL_TEXT"] = data_df["REVIEW_TITLE"] + ". " + data_df["REVIEW_TEXT"]
    

    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    
    '''
    Create training set and test set from raw data and engineer features
    like sentiment. 
    '''
    
    test_df["ALL_TEXT"] = data_df["ALL_TEXT"][test_set_indexes]
    test_df["RATING"] = data_df["RATING"][test_set_indexes]
    test_df["PRODUCT_CATEGORY"] = data_df["PRODUCT_CATEGORY"][test_set_indexes]
    # Length of review in words
    test_df["LENGTHS"] = test_df["ALL_TEXT"].apply(lambda x: len(NLP.tokenizer(x)))
    # Number of Exclamations
    test_df["EXCLAMATIONS"] = test_df["ALL_TEXT"].apply(lambda x: x.count("!"))
    # Title Sentiment
    test_df["TITLE_SENT"] = data_df["REVIEW_TITLE"][test_set_indexes].apply(lambda x : TextBlob(x).sentiment.polarity)
    # Text Sentiment
    test_df["TEXT_SENT"] = data_df["REVIEW_TEXT"][test_set_indexes].apply(lambda x : TextBlob(x).sentiment.polarity)
    test_df["LABEL"] = data_df["LABEL"][test_set_indexes]

    train_df["ALL_TEXT"] =  data_df["REVIEW_TEXT"][train_set_indexes]
    train_df["RATING"]=  data_df["RATING"][train_set_indexes]
    train_df["PRODUCT_CATEGORY"] =  data_df["PRODUCT_CATEGORY"][train_set_indexes]
    # Length of review in words
    train_df["LENGTHS"] = train_df["ALL_TEXT"].apply(lambda x: len(NLP.tokenizer(x)))
    # Number of Exclamations
    train_df["EXCLAMATIONS"] = train_df["ALL_TEXT"].apply(lambda x: x.count("!"))
    # Title Sentiment
    train_df["TITLE_SENT"] = data_df["REVIEW_TITLE"][train_set_indexes].apply(lambda x : TextBlob(x).sentiment.polarity)
    # Text Sentiment
    train_df["TEXT_SENT"] = data_df["REVIEW_TEXT"][train_set_indexes].apply(lambda x : TextBlob(x).sentiment.polarity)    
    train_df["LABEL"] =  data_df["LABEL"][train_set_indexes]

    print("train df label count ", train_df["LABEL"].value_counts())
    print("test df label count ", test_df["LABEL"].value_counts())
    
    print(train_df.head())
    print(test_df.head())
    
    data = AmazonReviewData(config)
    data.load_all_data("./glove.6B.50d.txt", train_df, test_df, NLP)

    model = FakeReviewDetector(config, len(data.vocab), data.word_embeddings)

    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    NLLLoss = nn.NLLLoss()
    train_losses = []
    val_accuracies = []

    for epoch in range(config["epochs"]):
        print ("Epoch: {}".format(epoch))        
        train_iterator =  data.train_iterator
        val_iterator = data.val_iterator

        train_loss = []
        val_accuracy = []
        losses = []

        for i, batch in enumerate(train_iterator):
            optimizer.zero_grad()

            text = batch.REVIEW_TEXT
            product_cat = batch.PRODUCT_CATEGORY
            rating = batch.RATING
            
            length = batch.LENGTH
            exclamations = batch.EXCLAMATIONS
            title_sent = batch.TITLE_SENTIMENT
            text_sent = batch.TEXT_SENTIMENT

            x = (text, product_cat, rating, length, exclamations, title_sent, text_sent)
            y = (batch.LABEL - 1).type(torch.LongTensor)
            
            y_pred = model(x)
            loss = NLLLoss(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            optimizer.step()
    
            if i % 100 == 0:
                print("Iter: {}".format(i+1))
                avg_train_loss = np.mean(losses)
                train_loss.append(avg_train_loss)
                print("\tAverage training loss: {:.5f}".format(avg_train_loss))
                losses = []
                
                # Evalute Accuracy on validation set
                val_accuracy = evaluate_model(model, val_iterator)

                print("\tVal Accuracy: {:.4f}".format(val_accuracy))
                model.train()

        if(epoch % 20 == 0):
            torch.save(model.state_dict(), "lstm-epoch-"+str(epoch))
        
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

