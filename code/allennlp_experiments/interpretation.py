from allennlp.data.token_indexers import TokenIndexer, PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, Tokenizer, PretrainedTransformerTokenizer
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import BertPooler
from allennlp.interpret.saliency_interpreters import SmoothGradient, SimpleGradient, IntegratedGradient
from allennlp.nn import util
from allennlp.predictors import TextClassifierPredictor
from allennlp.training.metrics import CategoricalAccuracy

from os.path import join as pathjoin
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import seaborn as sns
from termcolor import colored


label_description = {
    'A1': 'argum',
    'A3': 'emotive',
    'A4': 'fictive',
    'A5': 'flippant',
    'A6': 'informal',
    'A7': 'instruct',
    'A8': 'reporting',
    'A9': 'legal',
    'A11': 'personal',
    'A12': 'commercial',
    'A13': 'propaganda',
    'A14': 'research',
    'A15': 'specialist',
    'A16': 'info',
    'A17': 'eval',
    'A19': 'poetic',
    'A20': 'appeal',
    'A22': 'stuff'
}


def predict_classes(sentence_list, predictor, vocab):
    id_to_label = vocab.get_index_to_token_vocabulary('labels')
    return [id_to_label[np.argmax(predictor.predict(sentence)['probs'])] for sentence in sentence_list]


def calc_classifier_metrics(predicted_classes, true_classes):
    for label, description in label_description.items():
        true_binary = true_classes == label
        if np.sum(true_binary) == 0:
            continue
        predicted_binary = predicted_classes == label
        print(
            f"label ({description})", 
            f"f1_score {f1_score(predicted_binary, true_binary)}", 
            f"precision {precision_score(predicted_binary, true_binary)}", 
            f"recall {recall_score(predicted_binary, true_binary)}", 
        )
    print(f"accuracy", accuracy_score(predicted_classes, true_classes))
    

def plot_confusion_matrix(y_model, y_true):
    plt.figure(figsize=(15, 15)) 
    labels_list = np.unique(list(y_model) + list(y_true))
    cm = confusion_matrix(y_model, y_true, labels=labels_list)
    sums = np.sum(cm, axis=1)
    normed_cm = (cm.T / sums).T
    sns.heatmap(normed_cm)
    labels_descr = [label_description[label] for label in labels_list]
    plt.xticks(0.5 + np.arange(len(labels_list)), labels=labels_descr, fontsize=12)
    plt.yticks(0.5 + np.arange(len(labels_list)), labels=labels_descr, fontsize=12)
    
    
def run_interpreter(sentence, tokens, k, interpreter):
    vec = np.array(interpreter.saliency_interpret_from_json({"sentence": sentence})['instance_1']['grad_input_1'])
    important_indices = set(vec.argsort()[-k:])
    
    print(type(interpreter), "TEXT:")
    for token_id, token in enumerate(tokens):
        if token_id in important_indices:
            print(colored(token , "red"), end=' ')
        else:
            print(token, end=' ')
    print("\n")
    

def interpret_sentence(sentence, tokenizer, k, interpreters=[], true_label=None, label=None):
    if true_label is not None:
        print("TRUE LABEL:", true_label)
    if label is not None:
        print("LABEL:", label)
    
    tokens = tokenizer.tokenize(sentence)[1:511]
    k = min(k, len(tokens))
    for interpreter in interpreters:
        run_interpreter(sentence, tokens, k, interpreter)
        
        
def get_trigger_words(sentence, tokenizer, k, interpretor):
    result = []
    
    tokens = tokenizer.tokenize(sentence)
    k = min(k, len(tokens))
    vec = np.array(interpretor.saliency_interpret_from_json({"sentence": sentence})['instance_1']['grad_input_1'])
    important_indices = vec.argsort()[-k:]
    
    for token_id in important_indices:
        result.append(tokens[token_id])
    return result


def get_most_frequent_trigger_words(sentence_list, tokenizer, k, interpretor):
    result = {}
    
    for sentence in sentence_list:
        trigger_words = get_trigger_words(sentence, tokenizer, k, interpretor)
        for word in trigger_words:
            if str(word) not in result:
                result[str(word)] = 0
            result[str(word)] += 1
                
    result = list(result.items())
    result = sorted(result, key=lambda x: -x[1])
    return result


def get_dataset_by_confusion_pair(sentences, confusion_true, confusion_predicted,
                                  true_classes, predicted_classes):
    result = []
    
    for sentence, true_class, predicted_class in zip(sentences, true_classes, predicted_classes):
        if true_class == confusion_true and predicted_class == confusion_predicted:
            result.append(sentence)
    return result