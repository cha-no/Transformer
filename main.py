# 필요한 모듈 import
from typing import (
    Tuple, List, Any
)

import argparse

import os
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action='ignore')

# 구글 colab환경에서 tensorflow gpu 2.3.0 설치
os.system('pip install tensorflow-gpu==2.3.0')

import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

from tqdm import tqdm

from config import Config
from data_loader import download_data
from process import process_data, get_seq_len, get_vocab_size
from model import (
    PositionalEncoding,
    ScaledDotProductAttention,
    MultiHeadAttention,
    FeedForwardNetwork,
    EncoderLayer,
    DecoderLayer,
    Encoder,
    Decoder,
    Transformer,
    WarmUpScheduler
)

SEED = 2021
tf.random.set_seed(SEED)

# Config Parsing
def get_config() -> Any:
    parser = argparse.ArgumentParser(description="Transformer")
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--num_layers", default=6, type=int)
    parser.add_argument("--features", default=512, type=int)
    parser.add_argument("--num_heads", default=8, type=int)
    parser.add_argument("--fffeatures", default=2048, type=int)

    args = parser.parse_args()

    config = Config(
        EPOCHS = args.epochs,
        BATCH_SIZE = args.batch_size,
        LEARNING_RATE = args.lr,
        NUM_LAYERS = args.num_layers,
        FEATURES = args.features,
        NUM_HEADS = args.num_heads,
        FFFEATURES = args.fffeatures
    )

    return config

# train test set분리
def train_test_split(encoder_input : np.array, decoder_input : np.array, decoder_target : np.array,
    TEST_RATE : float = 0.2) -> Tuple[Tuple[Any, Any], Tuple[Any, Any], Tuple[Any, Any]]:

    shuffle_indices = np.arange(len(encoder_input))
    np.random.shuffle(shuffle_indices)

    encoder_input = encoder_input[shuffle_indices]
    decoder_input = decoder_input[shuffle_indices]
    decoder_target = decoder_target[shuffle_indices]

    val_num = int(len(encoder_input) * TEST_RATE)

    encoder_input_train, encoder_input_test = encoder_input[val_num:], encoder_input[:val_num]
    decoder_input_train, decoder_input_test = decoder_input[val_num:], decoder_input[:val_num]
    decoder_target_train, decoder_target_test = decoder_target[val_num:], decoder_target[:val_num]

    return (encoder_input_train, encoder_input_test), (decoder_input_train, decoder_input_test), (decoder_target_train, decoder_target_test)

# Padding Mask
def paddingMask(x : tf.Variable) -> tf.Variable:
    padding_mask = tf.cast(tf.equal(x, 0), tf.float32)
    return padding_mask[:, tf.newaxis, tf.newaxis, :]

# Nopeak Mask
def nopeakMask(x : tf.Variable) -> tf.Variable:
    batch, seq_len = tf.shape(x)[0], tf.shape(x)[1]
    nopeak_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    nopeak_mask = tf.tile(tf.expand_dims(nopeak_mask, 0), [batch, 1, 1])

    return nopeak_mask[:, tf.newaxis, :, :]

# Look Ahead Mask
def lookAheadMask(x : tf.Variable) -> tf.Variable:
    return tf.maximum(nopeakMask(x), paddingMask(x))

# Defining Model
def make_model(num_layers : int, features : int, num_heads : int, fffeatures : int, input_vocab_size : int,
    target_vocab_size : int, pe_input : int, pe_target : int, inp_seq_len : int, tar_seq_len : int):
    transformer = Transformer(
        num_layers = num_layers, features = features, num_heads = num_heads,
        fffeatures = fffeatures, input_vocab_size = input_vocab_size, target_vocab_size = target_vocab_size,
        pe_input = pe_input, pe_target = pe_target, rate = 0.1, name = 'transformer'
    )

    enc_inputs = Input(shape = (inp_seq_len, ), name = 'enc_inputs')
    dec_inputs = Input(shape = (tar_seq_len, ), name = 'dec_inputs')

    enc_padding_mask = Lambda(paddingMask, name = 'encoder_padding_mask')(enc_inputs)
    look_ahead_mask = Lambda(lookAheadMask, name = 'look_ahead_mask')(dec_inputs)
    dec_padding_mask = Lambda(paddingMask, name = 'decoder_padding_mask')(enc_inputs)

    outputs, _ = transformer(enc_inputs, dec_inputs,
                          enc_padding_mask = enc_padding_mask,
                          look_ahead_mask = look_ahead_mask,
                          dec_padding_mask = dec_padding_mask)
    
    return Model(inputs = [enc_inputs, dec_inputs], outputs = outputs)

# Train Model
def train_model(model, encoder_input : np.array, decoder_input : np.array, decoder_target : np.array, 
                EPOCHS : int, BATCH_SIZE : int) -> None:
    # Training Phase
    lr_schedule = WarmUpScheduler(features = config.FEATURES)       # warm_up lr scheduler
    model.compile(optimizer = Adam(lr_schedule, beta_1 = 0.9, beta_2 = 0.98, epsilon = 1e-9), loss = 'sparse_categorical_crossentropy', metrics = ['acc'])

    model.fit(x = [encoder_input, decoder_input], y = decoder_target, validation_split = 0.2,
                batch_size = BATCH_SIZE, epochs = EPOCHS)
    
    print("\nTraining Done !\n")

# translation prediction
def prediction(index : int, index2word : Any, model, encoder_inp : np.array) -> tf.Variable:
    test_input = tf.expand_dims(encoder_inp[index], axis = 0)
    test_output = tf.expand_dims([index2word.word_index['<sos>']], axis = 0)
    
    for i in range(tar_seq_len):
        preds = model(inputs = [test_input, test_output], training = False)

        ind = tf.cast(tf.argmax(preds[:, -1:, :], axis=-1), tf.int32)
        
        if ind == index2word.word_index['<eos>']:
            break
        
        test_output = tf.concat([test_output, ind], axis = -1)
    
    return tf.squeeze(test_output, axis = 0)

# 인코더의 인덱스 -> 단어
def encoder_index2word(index : int, index2word : Any, encoder_inp : np.array) -> str:
    encoder_input_sent = ''

    for ind in encoder_inp[index]:
        if not ind:
            break
        encoder_input_sent += index2word.index_word[ind] + ' '
    
    return encoder_input_sent

# 디코더의 인덱스 -> 단어
def decoder_index2word(index : int, index2word : Any, decoder_inp : np.array) -> str:
    decoder_input_sent = ''

    for ind in decoder_inp[index]:
        if not ind:
            break
        if index2word.index_word[ind] == '<sos>':
            continue
        decoder_input_sent += index2word.index_word[ind] + ' '
    
    return decoder_input_sent
  
# 디코더의 예측 인덱스 -> 단어
def decode_sentence(model, index : int, index2word : Any, encoder_inp : np.array) -> str:
    decoder_pred_sent = prediction(index, index2word = index2word, model = model, encoder_inp = encoder_inp)

    decode_sent = ''

    for ind in decoder_pred_sent.numpy():
        if not ind:
            break
        if index2word.index_word[ind] == '<sos>':
            continue
        decode_sent += index2word.index_word[ind] + ' '
    
    return decode_sent

if __name__ == "__main__":
    print(f"\nTensorflow version:[{tf.__version__}].\n")

    print(f"\nThis code use:[{tf.test.gpu_device_name() if tf.test.gpu_device_name() else 'cpu'}].\n")
    
    config = get_config()
    
    (enc_input, dec_input, dec_target) = download_data()

    (encoder_input, decoder_input, decoder_target), (tokenizer_encoder, tokenizer_decoder) = process_data(enc_input, dec_input, dec_target)
    
    inp_seq_len = get_seq_len(encoder_input)
    tar_seq_len = get_seq_len(decoder_input)

    input_vocab_size = get_vocab_size(tokenizer_encoder)
    target_vocab_size = get_vocab_size(tokenizer_decoder)

    print(f"\ninput 단어의 갯수 : {input_vocab_size}, output 단어의 갯수 : {target_vocab_size}\n")

    (encoder_input_train, encoder_input_test), (decoder_input_train, decoder_input_test), (decoder_target_train, decoder_target_test) = train_test_split(
        encoder_input, decoder_input, decoder_target
    )

    print("\nPreparing dataset done!\n")

    model = make_model(
    num_layers = config.NUM_LAYERS, features = config.FEATURES, num_heads = config.NUM_HEADS, fffeatures = config.FFFEATURES, 
    input_vocab_size = input_vocab_size, target_vocab_size = target_vocab_size, 
    pe_input = 10000, pe_target = 6000,
    inp_seq_len = inp_seq_len, tar_seq_len = tar_seq_len
    )

    model.summary()
    
    train_model(
        model, encoder_input_train, decoder_input_train, decoder_target_train, config.EPOCHS, config.BATCH_SIZE
    )
    
    print(f"\nTest_accuracy :{model.evaluate(x = [encoder_input_test, decoder_input_test], y = decoder_target_test)[1] : .4f}\n")

    for index in np.random.choice(len(encoder_input_test), 10):
        print("source :", encoder_index2word(index, tokenizer_encoder, encoder_input_test))
        print("target :", decoder_index2word(index, tokenizer_decoder, decoder_input_test))
        print("pred :", decode_sentence(model, index, tokenizer_decoder, encoder_input_test))
        print()
