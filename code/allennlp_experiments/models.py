from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.token_embedders import Embedding, PretrainedTransformerEmbedder, ElmoTokenEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import BertPooler, LstmSeq2VecEncoder, CnnEncoder, BagOfEmbeddingsEncoder, ClsPooler
from allennlp.nn import util
from allennlp.predictors import TextClassifierPredictor
from allennlp.training.metrics import CategoricalAccuracy

import numpy as np
import pandas as pd
import torch
from typing import Dict, Iterable, List, Tuple


class SimpleClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder 
        num_labels = vocab.get_vocab_size("labels")
        self.encoder = encoder
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()
        

    def forward(self,
                text: Dict[str, torch.Tensor],
                label: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        #print(embedded_text.shape)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        if label is not None:
            loss = torch.nn.functional.cross_entropy(logits, label)
            self.accuracy(logits, label)
            return {'loss': loss, 'probs': probs}
        else:
            return {'probs': probs}
    
    def get_metrics(self, reset: bool = True) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


class AdversarialClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 simple_classifier: SimpleClassifier,
                 alpha: float = 0.05,
                 target: str = 'label',
                 freeze_topic: bool = False):
        super().__init__(vocab)
        self.simple_classifier = simple_classifier
        num_topics = vocab.get_vocab_size("topic_labels")
        self.topic_classifier = torch.nn.Linear(
            self.simple_classifier.encoder.get_output_dim(), 
            num_topics
        )
        self.alpha = alpha
        self.index_to_label = vocab.get_index_to_token_vocabulary('labels')
        self.label_to_dif_index = simple_classifier.vocab.get_token_to_index_vocabulary('labels')
        self.target = target
        self.freeze_topic = freeze_topic
        self.topic_accuracy = CategoricalAccuracy()
        

    def forward(self,
                text: Dict[str, torch.Tensor],
                label: torch.Tensor=None,
                topic: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        #print("label", label, flush=True)
        #print("topic", topic, flush=True)
        if self.freeze_topic:
            self.topic_classifier.requires_grad = False
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.simple_classifier.embedder(text)
        #print(embedded_text.shape)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.simple_classifier.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.simple_classifier.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        topic_logits = self.topic_classifier(encoded_text)
        
        if label is not None:
            label_device = label.get_device()
            mod_label = [self.label_to_dif_index[self.index_to_label[cur_label]] for cur_label in label.tolist()]
            mod_label = torch.tensor(mod_label, device=label_device, dtype=torch.long)
            self.simple_classifier.accuracy(logits, mod_label)
            
        if label is not None and topic is not None:
            if self.target == 'label':
                loss = torch.nn.functional.cross_entropy(logits, mod_label)
                loss -= self.alpha * torch.nn.functional.cross_entropy(topic_logits, topic)
            else:
                loss = torch.nn.functional.cross_entropy(topic_logits, topic)
            self.simple_classifier.accuracy(logits, mod_label)    
            self.topic_accuracy(topic_logits, topic)
            if self.target == 'label':
                return {'loss': loss, 'probs': probs}
            else:
                return {'loss': loss, 'probs': torch.nn.functional.softmax(topic_logits, dim=-1)}
        else:
            return {'probs': probs}
    
    def get_metrics(self, reset: bool = True) -> Dict[str, float]:
        return {"accuracy": self.simple_classifier.accuracy.get_metric(reset), 
                "topic accuracy": self.topic_accuracy.get_metric(reset)}
    

def build_transformer_model(vocab: Vocabulary, transformer_model: str) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedding = PretrainedTransformerEmbedder(model_name=transformer_model)
    embedder = BasicTextFieldEmbedder(token_embedders={'bert_tokens': embedding})
    encoder = BertPooler(transformer_model)
    return SimpleClassifier(vocab, embedder, encoder)

def build_adversarial_transformer_model(vocab: Vocabulary, transformer_model: str) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedding = PretrainedTransformerEmbedder(model_name=transformer_model)
    embedder = BasicTextFieldEmbedder(token_embedders={'bert_tokens': embedding})
    encoder = BertPooler(transformer_model)
    return SimpleClassifier(vocab, embedder, encoder)

def build_pool_transformer_model(vocab: Vocabulary, transformer_model: str) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedding = PretrainedTransformerEmbedder(model_name=transformer_model)
    embedder = BasicTextFieldEmbedder(token_embedders={'bert_tokens': embedding})
    encoder = BagOfEmbeddingsEncoder(embedding_dim=embedder.get_output_dim(), averaged=True)
    #encoder = ClsPooler(embedding_dim=embedder.get_output_dim())
    return SimpleClassifier(vocab, embedder, encoder)

def build_elmo_model(vocab: Vocabulary) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedding = ElmoTokenEmbedder()
    embedder = BasicTextFieldEmbedder(token_embedders={'bert_tokens': embedding})
    encoder = BagOfEmbeddingsEncoder(embedding_dim=embedder.get_output_dim(), averaged=True)
    
    return SimpleClassifier(vocab, embedder, encoder)

def build_simple_lstm_model(vocab: Vocabulary,
                            emb_size: int = 256,
                            hidden_size: int = 256,
                            num_layers: int = 2,
                            bidirectional: bool = True) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedder = BasicTextFieldEmbedder(
        {"bert_tokens": Embedding(embedding_dim=emb_size, num_embeddings=vocab_size)}
    )
    encoder = LstmSeq2VecEncoder(
        input_size=emb_size, hidden_size=hidden_size, 
        num_layers=num_layers, bidirectional=bidirectional
    )
    return SimpleClassifier(vocab, embedder, encoder)

def build_simple_cnn_model(vocab: Vocabulary,
                           emb_size: int = 256,
                           output_dim: int = 256,
                           num_filters: int = 16,
                           ngram_filter_sizes: Tuple[int, ...] = (2, 3, 4, 5, 6)) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedder = BasicTextFieldEmbedder(
        {"bert_tokens": Embedding(embedding_dim=emb_size, num_embeddings=vocab_size)}
    )
    encoder = CnnEncoder(
        embedding_dim=emb_size, ngram_filter_sizes=ngram_filter_sizes, output_dim=output_dim, 
        num_filters=num_filters,
    )
    return SimpleClassifier(vocab, embedder, encoder)