from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import BertPooler
from allennlp.nn import util
from allennlp.predictors import TextClassifierPredictor
from allennlp.training.metrics import CategoricalAccuracy

import numpy as np
import pandas as pd
import torch
from typing import Dict, Iterable, List, Tuple


class SimpleTransformerClassifier(Model):
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


def build_transformer_model(vocab: Vocabulary, transformer_model: str) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedding = PretrainedTransformerEmbedder(model_name=transformer_model)
    embedder = BasicTextFieldEmbedder(token_embedders={'bert_tokens': embedding})
    encoder = BertPooler(transformer_model)
    return SimpleTransformerClassifier(vocab, embedder, encoder)