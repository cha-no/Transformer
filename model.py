from typing import Tuple

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Layer, LayerNormalization, Dropout
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras import backend as K

class PositionalEncoding(object):
    
    def __init__(self, position : int, features : int) -> None:
        super(PositionalEncoding, self).__init__()
        angle_rads = self.get_angles(np.arange(position).reshape(-1, 1), np.arange(features).reshape(1, -1), features)
        
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        self.pos_encoding = np.expand_dims(angle_rads, 0)

    def __call__(self, x : np.array) -> np.array:
        # x : (seq_len, features)
        # pos_encoding: (position, features)
        return x + self.pos_encoding[:, :x.shape[1], :]

    def get_angles(self, pos : np.array, i : np.array, features : int) -> np.array:
        return pos / np.power(10000, (2 * (i // 2)) / features)

class ScaledDotProductAttention(Layer):

    def call(self, query : tf.Variable, key : tf.Variable, value : tf.Variable, mask = None) -> Tuple[tf.Variable, tf.Variable]:
        dk = tf.cast(key.shape[-1], tf.float32)   # 가중치 조정을 위한 scale값
        scores = tf.matmul(query, key, transpose_b = True) / tf.math.sqrt(dk)   # query와 key dot product

        if mask is not None:
            scores += (mask * -1e9)  
        
        attention = tf.nn.softmax(scores, axis = -1)  # 어텐션 분포
        out = tf.matmul(attention, value)             # 어텐션 값
        return out, attention

class MultiHeadAttention(Layer):

    def __init__(self, features : int, num_heads : int, bias : bool = True, name = "multi_head_attention") -> None:
        super(MultiHeadAttention, self).__init__(name = name)
        assert features % num_heads == 0, f'"features"(features) should be divisible by "head_num"(num_heads)'
        
        self.features = features            # feature의 차원
        self.num_heads = num_heads          # head의 갯수
        self.bias = bias
        self.depth = features // num_heads  # head 하나의 차원

        # query, key, value 가중치 행렬 생성
        self.wq = Dense(features, use_bias = self.bias)
        self.wk = Dense(features, use_bias = self.bias)
        self.wv = Dense(features, use_bias = self.bias)

        self.fc = Dense(features, use_bias = self.bias)

    def split_heads(self, x : tf.Variable, batch_size : int) -> tf.Variable:
        # num_heads의 갯수로 split
        # batch_size, num_heads, seq_len, depth
        split_heads = tf.reshape(x, shape = (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(split_heads, perm = [0, 2, 1, 3])

    def call(self, q : tf.Variable, k : tf.Variable, v : tf.Variable, mask = None) -> Tuple[tf.Variable, tf.Variable]:
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        # print(q.shape, k.shape, v.shape)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        # print(q.shape, k.shape, v.shape)
        
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = ScaledDotProductAttention()(q, k, v, mask)
        # print(scaled_attention.shape, attention_weights.shape)
        
        scaled_attention = tf.transpose(scaled_attention, perm = [0,2,1,3])
        # 어텐션 값을 concat
        concat_attention = tf.reshape(scaled_attention , shape = (batch_size, -1, self.features))

        out = self.fc(concat_attention)

        return out, attention_weights

class FeedForwardNetwork(Layer):

    def __init__(self, features : int, fffeatures : int, name = 'feedforward_layer') -> None:
        super(FeedForwardNetwork, self).__init__(name = name)

        self.feed = Dense(fffeatures, activation = 'relu')
        self.fffeed = Dense(features)

    def call(self, x : tf.Variable) -> tf.Variable:
        x = self.feed(x)
        x = self.fffeed(x)
        return x

class EncoderLayer(Layer):

    def __init__(self, features : int, num_heads : int, fffeatures : int, rate : float = 0.1, name = "encoder_layer") -> None:
        super(EncoderLayer, self).__init__(name = name)

        self.mha = MultiHeadAttention(features, num_heads)
        self.ffn = FeedForwardNetwork(features, fffeatures)

        self.layernorm1 = LayerNormalization(epsilon = 1e-6)
        self.layernorm2 = LayerNormalization(epsilon = 1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x : tf.Variable, mask) -> tf.Variable:
        
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)   # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)               # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output) # (batch_size, input_seq_len, d_model)

        return out2

class DecoderLayer(Layer):
    
    def __init__(self, features : int, num_heads : int, fffeatures : int, rate : float = 0.1, name = "decoder_layer") -> None:
        super(DecoderLayer, self).__init__(name = name)

        # masked multi head attention
        self.mha1 = MultiHeadAttention(features, num_heads)
        # encoder - decoder multi head attention
        self.mha2 = MultiHeadAttention(features, num_heads)
        self.ffn = FeedForwardNetwork(features, fffeatures)

        self.layernorm1 = LayerNormalization(epsilon = 1e-6)
        self.layernorm2 = LayerNormalization(epsilon = 1e-6)
        self.layernorm3 = LayerNormalization(epsilon = 1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)

    def call(self, x : tf.Variable, enc_output : tf.Variable, look_ahead_mask, padding_mask) -> Tuple[tf.Variable, tf.Variable, tf.Variable]:
        
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        # print(enc_output.shape)

        # masked multi head attention
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)                    # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)
        
        # encoder - decoder multi head attention
        attn2, attn_weights_block2 = self.mha2(out1, enc_output, enc_output, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)                                                # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)                                                         # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)                                           # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2

class Encoder(Layer):

    def __init__(self, num_layers : int, features : int, num_heads : int, fffeatures : int, input_vocab_size : int, maximum_position_encoding : int, rate : float = 0.1, name = 'encoder') -> None:
        super(Encoder, self).__init__(name = name)

        self.num_layers = num_layers
        self.features = features
        self.embedding = Embedding(input_vocab_size, features)
        self.pos_encoding = PositionalEncoding(maximum_position_encoding, features)

        self.enc_layers = [EncoderLayer(self.features, num_heads, fffeatures, rate, f'encoder_layer_{i + 1}') for i in range(self.num_layers)]

        self.dropout = Dropout(rate)

    def call(self, x : tf.Variable, mask) -> tf.Variable:
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        # print(x.shape)
        x *= tf.math.sqrt(tf.cast(self.features, tf.float32))
        # print(x.shape)
        x = self.pos_encoding(x)
        # print(x.shape, self.pos_encoding.shape)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)

        return x

class Decoder(Layer):

    def __init__(self, num_layers : int, features : int, num_heads : int, fffeatures : int, target_vocab_size : int, maximum_position_encoding : int, rate : float = 0.1, name = 'decoder') -> None:
        super(Decoder, self).__init__(name = name)

        self.num_layers = num_layers
        self.features = features
        self.embedding = Embedding(target_vocab_size, features)
        self.pos_encoding = PositionalEncoding(maximum_position_encoding, features)

        self.dec_layers = [DecoderLayer(self.features, num_heads, fffeatures, rate, f'decoder_layer_{i + 1}') for i in range(self.num_layers)]

        self.dropout = Dropout(rate)

    def call(self, x : tf.Variable, enc_output : tf.Variable, look_ahead_mask, padding_mask) -> Tuple[tf.Variable, dict]:
        seq_len = x.shape[1]
        attention_weights = {}

        x = self.embedding(x)
        # print(x.shape)
        x *= tf.math.sqrt(tf.cast(self.features, tf.float32))
        # print(x.shape)
        x = self.pos_encoding(x)
        # print(x.shape, self.pos_encoding.shape)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        return x, attention_weights

class Transformer(Layer):

    def __init__(self, num_layers : int, features : int, num_heads : int, fffeatures : int, input_vocab_size : int, 
                target_vocab_size : int, pe_input : int, pe_target : int, rate : float = 0.1, name = "transformer") -> None:
        super(Transformer, self).__init__(name = name)

        self.encoder = Encoder(num_layers, features, num_heads, fffeatures, 
                            input_vocab_size, pe_input, rate, 'encoder')

        self.decoder = Decoder(num_layers, features, num_heads, fffeatures, 
                            target_vocab_size, pe_target, rate, 'decoder')

        self.final_layer = Dense(target_vocab_size, activation = 'softmax')

    def call(self, inp : tf.Variable, tar : tf.Variable, enc_padding_mask, 
            look_ahead_mask, dec_padding_mask) -> Tuple[tf.Variable, dict]:

        # enc_output.shape == (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(inp, enc_padding_mask)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

# warm_up lr scheduler
class WarmUpScheduler(LearningRateSchedule):

    def __init__(self, features : int, warmup_steps : int = 4000) -> None:
        super(WarmUpScheduler, self).__init__()
        self.features = features
        self.features = tf.cast(self.features, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step : int) -> tf.Variable:
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.features) * tf.math.minimum(arg1, arg2)