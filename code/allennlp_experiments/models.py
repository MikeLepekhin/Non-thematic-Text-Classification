from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.token_embedders import Embedding, PretrainedTransformerEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import BertPooler, LstmSeq2VecEncoder, CnnEncoder
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
                 simple_classifier: SimpleClassifier,
                 lambd: float=1.0):
        super().__init__(simple_classifier.vocab)
        self.simple_classifier = simple_classifier
        self.domain_classifier = torch.nn.Linear(encoder.get_output_dim(), 2)
        

    def forward(self,
                text: Dict[str, torch.Tensor],
                label: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.simple_classifier.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.simple_classifier.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.simple_classifier.classifier(encoded_text)
        # Shape: (batch_size, 2)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        domain_logits = self.simple_classifier.classifier(encoded_text)
        domain_probs = torch.nn.functional.softmax(domain_logits, dim=-1)
        
        if label is not None:
            domain_loss = torch.nn.functional.cross_entropy(domain_logits, domain)
            loss = torch.nn.functional.cross_entropy(logits, label) - lambd * domain_loss
            self.accuracy(logits, label)
            return {'loss': loss, 'probs': probs}
        else:
            return {'probs': probs}
    
    def get_metrics(self, reset: bool = True) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}

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