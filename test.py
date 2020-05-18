import torch.optim as optim
from torch import nn
import torch
import spacy
import pandas as pd
import seaborn as sn
import re
import numpy as np
from matplotlib import pyplot as plt

# a list of indexes for the train and test set 
from test_set_indexes import train_set_indexes, test_set_indexes
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from amazon_review_data import *
from fake_review_detector import *


from textblob import TextBlob

####################################
####################################
# PUT PATH OF MODEL TO TEST HERE:

MODEL_TO_TEST_PATH = "./trained-lstm-epoch-22000"

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

      
    return all_y, all_preds

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
    Create test set and test set from raw data and engineer features
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
    
    data = AmazonReviewData(config)
    data.load_test_data("./glove.6B.50d.txt", test_df, NLP)
 
    
    model = FakeReviewDetector(config, 31417, data.word_embeddings)
    pretrained_dict = torch.load(MODEL_TO_TEST_PATH)        
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    del pretrained_dict["embeddings.weight"]
    # print(pretrained_dict)
    
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)

    y_true, y_preds= evaluate_model(model, data.test_iterator)
    # currently y_true contains 1s for Real, and 2s for Fake
    # lets change it so 1 means Real and 0 means Fake.
    # y_true = [1 if i == 1 else 0 for i in y_true]
    

    

    accuracy = accuracy_score(y_true, y_preds)
    
    # first label is negative, second is positive.
    cf = confusion_matrix(y_true, y_preds, labels=[2,1])
    print("POSITIVES ARE REAL REVIEWS")
    print("NEGATIVES ARE FAKE REVIEWS")
    print("")
    print("Accuracy", accuracy)
    print("")
    print("CONFUSION MATRIX")
    print(cf)
    print("")
    
    tn, fp, fn, tp = cf.ravel()
    
    print("TRUE NEGATIVES", tn)
    print("FALSE POSITIVES", fp)
    print("FALSE NEGATIVES", fn)
    print("TRUE POSITIVES", tp)
    
    print("RECALL", (tp/(tp+fn)))
    print("PRECISION", (tp/ (tp+fp)))
    print("")
    print("")
    
    print("Classification Report: ")
    print(classification_report(y_true, y_preds, target_names=["REAL REVIEW", "FAKE REVIEW"]))
    
    
    df_cm = pd.DataFrame(cf, index = ["ACTUAL FAKE", "ACTUAL REAL"  ],
                  columns = ["PREDICTED FAKE", "PREDICTED REAL" ])
    
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, cmap="Blues",fmt='g',  annot=True, annot_kws={"size": 16})
    sn.set(font_scale=1.4) 
    
    plt.show()
    
