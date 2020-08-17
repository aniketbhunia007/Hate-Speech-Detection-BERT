from __future__ import absolute_import, division, print_function

import os
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

import xgboost

import pandas as pd
import numpy as np
import pickle
import sys
import nltk

from nltk.stem.porter import *
from nltk.corpus import stopwords
from textblob import TextBlob
from textblob import Word

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from sklearn.feature_extraction.text import TfidfVectorizer

import gensim
import logging
from gensim.models import Word2Vec

import tweepy
import csv
import pandas as pd

from scrape import scrape
from scrape_username import get_all_tweets

import csv
import os
import sys
import logging


import torch
import pickle
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.nn import CrossEntropyLoss, MSELoss

from tqdm import tqdm_notebook, trange
import os
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from multiprocessing import Pool, cpu_count
# from tools import *
# from convert_examples_to_features import *

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



logger = logging.getLogger()
csv.field_size_limit(2147483647) # Increase CSV reader's field limit incase we have long text.


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class BinaryClassificationProcessor(DataProcessor):
    """Processor for binary classification dataset."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_example_to_feature(example_row):
    # return example_row
    example, label_map, max_seq_length, tokenizer, output_mode = example_row

    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if output_mode == "classification":
        label_id = label_map[example.label]
    elif output_mode == "regression":
        label_id = float(example.label)
    else:
        raise KeyError(output_mode)

    return InputFeatures(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         label_id=label_id)


import sys
# sys.path.append('/content/drive/My Drive/Hate_Speech_Detection_git/Code_Notebooks')
sys.path.append('/content')



# The input data dir. Should contain the .tsv files (or other data files) for the task.
DATA_DIR = ""

# Bert pre-trained model selected in the list: bert-base-uncased, 
# bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased,
# bert-base-multilingual-cased, bert-base-chinese.
BERT_MODEL = 'hate_speech.tar.gz'

# The name of the task to train.I'm going to name this 'yelp'.
TASK_NAME = 'hate_speech'

# The output directory where the fine-tuned model and checkpoints will be written.
OUTPUT_DIR = f'outputs/{TASK_NAME}/'

# The directory where the evaluation reports will be written to.
REPORTS_DIR = f'reports/{TASK_NAME}_evaluation_reports/'

# This is where BERT will look for pre-trained models to load parameters from.
CACHE_DIR = 'cache/'

# The maximum total input sequence length after WordPiece tokenization.
# Sequences longer than this will be truncated, and sequences shorter than this will be padded.
MAX_SEQ_LENGTH = 128

TRAIN_BATCH_SIZE = 24
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3
RANDOM_SEED = 42
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_PROPORTION = 0.1
OUTPUT_MODE = 'classification'

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"

if os.path.exists(REPORTS_DIR) and os.listdir(REPORTS_DIR):
        REPORTS_DIR += f'/report_{len(os.listdir(REPORTS_DIR))}'
        os.makedirs(REPORTS_DIR)
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)
    REPORTS_DIR += f'/report_{len(os.listdir(REPORTS_DIR))}'
    os.makedirs(REPORTS_DIR)

def get_eval_report(task_name, labels, preds):
    mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    return {
        "task": task_name,
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }

def compute_metrics(task_name, labels, preds):
    assert len(preds) == len(labels)
    return get_eval_report(task_name, labels, preds)



def clean_remove_b(data):  
  data.rename(columns={'text':'tweet'},inplace=True)
  data['tweet'] = data['tweet'].astype(str)

  data['tweet'] = data['tweet'].apply(lambda x:x[2:] if x[0:2]=="b'" or 'b"' else x)

def preprocess(data):
  data['tweet'] = data['tweet'].apply(lambda x:' '.join(i for i in [a for a in x.split() if a.find('@')==-1]))
  data['tweet'] = data['tweet'].apply(lambda x:' '.join(i for i in [a for a in x.split() if a.find('http')==-1]))
  
  ## we are removing hashtags now, but while doing transfer learning, to learn the embeddings we didnt remove these, 
  ## just to include such words in our vocabulary
  
  data['tweet'] = data['tweet'].apply(lambda x:' '.join(i for i in [a for a in x.split() if a.find('#')==-1]))
  data['tweet'] = data['tweet'].apply(lambda x:''.join([i for i in x if not i.isdigit()]))
  data['tweet'] = data['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))
  data['tweet'] = data['tweet'].str.replace('[^\w\s]','')

# for BERT we wont remove the stopwords, because it was trained on sentences containing stopwords  
#   stop = stopwords.words('english')
#   data['tweet'] = data['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

#   remove_word = ['rt','mkr','im']
#   data['tweet'] = data['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in remove_word))

# Although we used lemmatization while training so lets keep it as it iss
def preprocess_2(data):
  data['tweet'] = data['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
  data['tweet'].head()


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(OUTPUT_DIR + 'vocab.txt', do_lower_case=False)

    processor = BinaryClassificationProcessor()
    # Load pre-trained model (weights)

    if request.method == 'POST':
        if len(request.form['text_input'])>1:
            message = request.form['text_input']
            data = pd.DataFrame([str(message)],columns=['tweet'])

            print(data)
            
            preprocess(data)
            preprocess_2(data)
            
            print(data)
            print(data['tweet'][0])

            eval_examples = [InputExample(guid=0, text_a=data['tweet'][0], text_b=None, label='1')]

            label_list = processor.get_labels() # [0, 1] for binary classification
            num_labels = len(label_list)
            eval_examples_len = len(eval_examples)

            label_map = {label: i for i, label in enumerate(label_list)}
            eval_examples_for_processing = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in eval_examples]


            process_count = cpu_count() - 1
            if __name__ ==  '__main__':
                print(f'Preparing to convert {eval_examples_len} examples..')
                print(f'Spawning {process_count} processes..')
                with Pool(process_count) as p:
                    eval_features = list(tqdm_notebook(p.imap(convert_example_to_feature, eval_examples_for_processing), total=eval_examples_len))


            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

            # In [25]:
            if OUTPUT_MODE == "classification":
                all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
            elif OUTPUT_MODE == "regression":
                all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

            # In [26]:
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=EVAL_BATCH_SIZE)

            model = BertForSequenceClassification.from_pretrained(CACHE_DIR + BERT_MODEL, cache_dir=CACHE_DIR, num_labels=len(label_list))
            model.to(device)
            
            model.eval()
            eval_loss = 0
            nb_eval_steps = 0
            preds = []

            for input_ids, input_mask, segment_ids, label_ids in tqdm_notebook(eval_dataloader, desc="Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask, labels=None)

                # create eval loss and other metric required by the task
                if OUTPUT_MODE == "classification":
                    loss_fct = CrossEntropyLoss()
                    tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                elif OUTPUT_MODE == "regression":
                    loss_fct = MSELoss()
                    tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                else:
                    preds[0] = np.append(
                        preds[0], logits.detach().cpu().numpy(), axis=0)

            eval_loss = eval_loss / nb_eval_steps
            preds = preds[0]

            from scipy.special import softmax

            pred_probs = softmax(preds, axis=1)

            my_prediction = np.argmax(pred_probs)
            print(pred_probs)
            print(my_prediction,'predicted label')
            # if my_prediction == 1:
            #     probab = pred_probs[0][1]
            #     print(probab)
            # else:
            #     probab = pred_probs[0][0]
            #     print(probab)
            probab = pred_probs[0][1]
            print(probab,'hate probability')
            
            return render_template('result.html',prediction = my_prediction, probability = probab*100.00)




        elif len(request.form['hashtag'])>1:

            hashtag = request.form['hashtag']
            no_of_tweets = int(request.form['num_tweets'])
            date = str(request.form['date'])
            print(hashtag,no_of_tweets,date)
            
            scrape(hashtag,date,no_of_tweets)

            print('file saved!')
            data = pd.read_csv('scraped.csv',names=['timestamp','text'])
            # data = pd.DataFrame([str(message)],columns=['tweet'])

            clean_remove_b(data)

            data.drop_duplicates(subset='tweet',inplace=True)
            data.reset_index(drop=True, inplace=True)

            data_original = data.copy()

            preprocess(data)
            preprocess_2(data)

            bert_test_input = pd.DataFrame({
                'id':range(len(data)),
                'label':np.ones(data.shape[0], dtype=int),
                'alpha':['a']*data.shape[0],
                'text': data['tweet']
            })

            bert_test_input.to_csv('dev.tsv', sep='\t', index=False, header=False)

            eval_examples = processor.get_dev_examples(DATA_DIR)
            label_list = processor.get_labels() # [0, 1] for binary classification
            num_labels = len(label_list)
            eval_examples_len = len(eval_examples)

            label_map = {label: i for i, label in enumerate(label_list)}
            eval_examples_for_processing = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in eval_examples]

            process_count = cpu_count() - 1
            if __name__ ==  '__main__':
                print(f'Preparing to convert {eval_examples_len} examples..')
                print(f'Spawning {process_count} processes..')
                with Pool(process_count) as p:
                    eval_features = list(tqdm_notebook(p.imap(convert_example_to_feature, eval_examples_for_processing), total=eval_examples_len))
            
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

            # In [25]:
            if OUTPUT_MODE == "classification":
                all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
            elif OUTPUT_MODE == "regression":
                all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

            # In [26]:
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=EVAL_BATCH_SIZE)

            model = BertForSequenceClassification.from_pretrained(CACHE_DIR + BERT_MODEL, cache_dir=CACHE_DIR, num_labels=len(label_list))
            model.to(device)


            model.eval()
            eval_loss = 0
            nb_eval_steps = 0
            preds = []

            for input_ids, input_mask, segment_ids, label_ids in tqdm_notebook(eval_dataloader, desc="Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask, labels=None)

                # create eval loss and other metric required by the task
                if OUTPUT_MODE == "classification":
                    loss_fct = CrossEntropyLoss()
                    tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                elif OUTPUT_MODE == "regression":
                    loss_fct = MSELoss()
                    tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                else:
                    preds[0] = np.append(
                        preds[0], logits.detach().cpu().numpy(), axis=0)

            eval_loss = eval_loss / nb_eval_steps
            preds = preds[0]

            from scipy.special import softmax
            pred_probs = softmax(preds, axis=1)
            
            print(pred_probs)

            if OUTPUT_MODE == "classification":
                preds = np.argmax(pred_probs, axis=1)
            elif OUTPUT_MODE == "regression":
                preds = np.squeeze(pred_probs)

            # result = compute_metrics(TASK_NAME, all_label_ids.numpy(), preds)

            # result['eval_loss'] = eval_loss

            # output_eval_file = os.path.join(REPORTS_DIR, "eval_results.txt")
            # with open(output_eval_file, "w") as writer:
            #     logger.info("***** Eval results *****")
            #     for key in (result.keys()):
            #         logger.info("  %s = %s", key, str(result[key]))
            #         writer.write("%s = %s\n" % (key, str(result[key])))

            my_prediction = preds
            print(my_prediction)

            pd.set_option('display.max_colwidth',1000)
            data_original['prediction'] = my_prediction
            data_original['probability'] = pred_probs[::,-1] 
            data_original.sort_values(by='probability',ascending=False,inplace=True)
            data_original.reset_index(drop=True,inplace=True)
            # data_original.style.set_properties(subset=['tweet'], **{'width': '100%'})
            print(data_original)

            return render_template('result_dataframe.html',tables=[data_original.to_html(classes='hashtags')],titles = ['Analysis_on_hashtag'], hashtag = hashtag)

        elif len(request.form['username'])>1:

            username = request.form['username']
            no_of_tweets = int(request.form['no_of_tweets'])
            print(username,no_of_tweets)
            
            get_all_tweets(username,no_of_tweets)

            print('file saved!')
            data = pd.read_csv('username_tweets.csv',names=['id','created_at','text'])
            # data = pd.DataFrame([str(message)],columns=['tweet'])

            clean_remove_b(data)

            data.drop_duplicates(subset='tweet',inplace=True)
            data.reset_index(drop=True, inplace=True)

            data_original = data.copy()

            preprocess(data)
            preprocess_2(data)
                

            bert_test_input = pd.DataFrame({
                'id':range(len(data)),
                'label':np.ones(data.shape[0], dtype=int),
                'alpha':['a']*data.shape[0],
                'text': data['tweet']
            })

            bert_test_input.to_csv('dev.tsv', sep='\t', index=False, header=False)

            eval_examples = processor.get_dev_examples(DATA_DIR)
            label_list = processor.get_labels() # [0, 1] for binary classification
            num_labels = len(label_list)
            eval_examples_len = len(eval_examples)

            label_map = {label: i for i, label in enumerate(label_list)}
            eval_examples_for_processing = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in eval_examples]

            process_count = cpu_count() - 1
            if __name__ ==  '__main__':
                print(f'Preparing to convert {eval_examples_len} examples..')
                print(f'Spawning {process_count} processes..')
                with Pool(process_count) as p:
                    eval_features = list(tqdm_notebook(p.imap(convert_example_to_feature, eval_examples_for_processing), total=eval_examples_len))
            
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

            # In [25]:
            if OUTPUT_MODE == "classification":
                all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
            elif OUTPUT_MODE == "regression":
                all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

            # In [26]:
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=EVAL_BATCH_SIZE)

            model = BertForSequenceClassification.from_pretrained(CACHE_DIR + BERT_MODEL, cache_dir=CACHE_DIR, num_labels=len(label_list))
            model.to(device)


            model.eval()
            eval_loss = 0
            nb_eval_steps = 0
            preds = []

            for input_ids, input_mask, segment_ids, label_ids in tqdm_notebook(eval_dataloader, desc="Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask, labels=None)

                # create eval loss and other metric required by the task
                if OUTPUT_MODE == "classification":
                    loss_fct = CrossEntropyLoss()
                    tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                elif OUTPUT_MODE == "regression":
                    loss_fct = MSELoss()
                    tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                else:
                    preds[0] = np.append(
                        preds[0], logits.detach().cpu().numpy(), axis=0)

            eval_loss = eval_loss / nb_eval_steps
            preds = preds[0]

            from scipy.special import softmax
            pred_probs = softmax(preds, axis=1)
            
            print(pred_probs)

            if OUTPUT_MODE == "classification":
                preds = np.argmax(pred_probs, axis=1)
            elif OUTPUT_MODE == "regression":
                preds = np.squeeze(pred_probs)

            # result = compute_metrics(TASK_NAME, all_label_ids.numpy(), preds)

            # result['eval_loss'] = eval_loss

            # output_eval_file = os.path.join(REPORTS_DIR, "eval_results.txt")
            # with open(output_eval_file, "w") as writer:
            #     logger.info("***** Eval results *****")
            #     for key in (result.keys()):
            #         logger.info("  %s = %s", key, str(result[key]))
            #         writer.write("%s = %s\n" % (key, str(result[key])))

            my_prediction = preds
            print(my_prediction)



            pd.set_option('display.max_colwidth',1000)
            data_original['prediction'] = my_prediction
            data_original['probability'] = pred_probs[::,-1]
            print(data_original)
            data_original.sort_values(by='probability',ascending=False,inplace=True)
            data_original.reset_index(drop=True,inplace=True)
            print(data_original)
            print(data_original['tweet'][0])

            return render_template('result_dataframe.html',tables=[data_original.to_html(classes='hashtags')],titles = ['Analysis_on_hashtag'], hashtag = username)



if __name__ == '__main__':
	app.run(debug=True, port=3000)