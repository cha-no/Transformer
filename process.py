from typing import (
    Tuple, List, Any
)

import numpy as np

import warnings
warnings.filterwarnings(action='ignore')

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# nlp data preprocessing
def process_data(enc_input : List[List[str]], dec_input : List[List[str]], dec_target : List[List[str]]) -> Tuple[Tuple[np.array, np.array, np.array], Tuple[Any, Any]]:
    tokenizer_encoder = Tokenizer()
    tokenizer_encoder.fit_on_texts(enc_input)
    enc_input = tokenizer_encoder.texts_to_sequences(enc_input)

    tokenizer_decoder = Tokenizer()
    tokenizer_decoder.fit_on_texts(dec_input)
    tokenizer_decoder.fit_on_texts(dec_target)
    dec_input = tokenizer_decoder.texts_to_sequences(dec_input)
    dec_target = tokenizer_decoder.texts_to_sequences(dec_target)

    encoder_input = pad_sequences(enc_input, padding="post")
    decoder_input = pad_sequences(dec_input, padding="post")
    decoder_target = pad_sequences(dec_target, padding="post")

    return (encoder_input, decoder_input, decoder_target), (tokenizer_encoder, tokenizer_decoder)

# seq len
def get_seq_len(sequence : np.array) -> int:
    return max([len(sentence) for sentence in sequence])

# vocab size
def get_vocab_size(tokenizer) -> int:
    return len(tokenizer.word_index) + 1