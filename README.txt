Please have python3 and pip installed on computer before running commands below. 

Install dependencies like so:
pip install -r requirements.txt

Download the 50d GloVe Model from here: 
https://nlp.stanford.edu/projects/glove/

############################################################
To train a model, edit configuration in train.py. Then run:

python train.py

############################################################
To test a model, such as the trained model, trained-lstm-epoch-22000, 
put model path of model to test in variable found at the top of test.py script called MODEL_TO_TEST_PATH, 
and then run:

python test.py
