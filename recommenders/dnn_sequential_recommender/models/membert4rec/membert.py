import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import math

class TEmbedding(keras.layers.Layer):
  def __init__(self, num_embeddings, hidden_dim, seq_len, dropout_rate):
    super(TEmbedding, self).__init__()
    
    self.num_embeddings = num_embeddings
    self.hidden_dim=hidden_dim
    self.seq_len = seq_len
    self.dropout_rate = dropout_rate

    self.embedding = keras.layers.Embedding(num_embeddings, hidden_dim)
    self.pos_embeds= keras.layers.Embedding(seq_len, hidden_dim)
    self.layernorm = keras.layers.LayerNormalization(epsilon=1e-12)
    self.dropout   = keras.layers.Dropout(dropout_rate)

  def __call__(self, input):
    seq_len = tf.shape(input)[-1]
    
    embed = self.embedding(input)
    pos   = tf.expand_dims(tf.range(seq_len, dtype=tf.int32), 0)
    pos   = self.pos_embeds(pos)

    embed = embed + pos
    embed = self.layernorm(embed)
    embed = self.dropout(embed)

    return embed

class TAttention(keras.layers.Layer):
  def __init__(self, hidden_dim, num_heads, dropout_rate):
    super(TAttention, self).__init__()
    self.hidden_dim=hidden_dim
    self.num_heads =num_heads
    
    self.head_dim = hidden_dim // num_heads

    self.q = keras.layers.Dense(self.hidden_dim)
    self.k = keras.layers.Dense(self.hidden_dim)
    self.v = keras.layers.Dense(self.hidden_dim)

    self.lin = keras.layers.Dense(self.hidden_dim)

    self.dropout = keras.layers.Dropout(dropout_rate)

    self.layernorm = keras.layers.LayerNormalization(epsilon=1e-12)
    self.dropout_final = keras.layers.Dropout(dropout_rate)

  def split_heads(self, x):
    bs, seq_len = tf.shape(x)[0], tf.shape(x)[1]
    new_shape = (bs, seq_len, self.num_heads, self.head_dim)
    x = tf.reshape(x, new_shape)
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def join_heads(self, x):
    bs, seq_len, hidden = tf.shape(x)[0], tf.shape(x)[2], tf.shape(x)[3] * tf.shape(x)[1]
    x = tf.transpose(x, perm=[0, 2, 1, 3])
    x = tf.reshape(x, shape=[bs, seq_len, hidden])

    return x

  def __call__(self, x):
    q = self.q(x)
    k = self.k(x)
    v = self.v(x)

    q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
    q = tf.math.scalar_mul(1. / math.sqrt(self.hidden_dim), q)

    qk = tf.einsum('bhqd,bhkd->bhqk', q, k)
    qk = tf.nn.softmax(qk, axis=-1)

    qk = self.dropout(qk) #Like in TF implementation; could be done before Softmax by random -inf addition

    out = tf.einsum('bhqk,bhkd->bhqd', qk, v)
    out = self.join_heads(out)
    
    out = self.lin(out)
    out = self.dropout_final(out)

    out = self.layernorm(out + x)

    return out

class TFFN(keras.layers.Layer):
  def __init__(self, hidden_dim, dropout_rate):
    super(TFFN, self).__init__()
    self.expand = keras.layers.Dense(4 * hidden_dim)
    self.act    = keras.activations.gelu
    self.contract=keras.layers.Dense(hidden_dim)

    self.dropout = keras.layers.Dropout(dropout_rate)
    self.layernorm = keras.layers.LayerNormalization(epsilon=1e-12)

  def __call__(self, q):
    x = self.expand(q)
    x = self.act(x)
    x = self.contract(x)
    x = self.dropout(x)

    x = self.layernorm(q + x)

    return x

class TBlock(keras.layers.Layer):
  def __init__(self, hidden_dim, num_heads, dropout_rate):
    super(TBlock, self).__init__()
    self.att = TAttention(hidden_dim, num_heads, dropout_rate)
    self.ffn = TFFN(hidden_dim, dropout_rate)

  def __call__(self, q):
    q = self.att(q)
    q = self.ffn(q)

    return q

class TOutput(keras.layers.Layer):
  def __init__(self, hidden_dim, vocab_size, linked_layer=None):
    super(TOutput, self).__init__()
    self.transform = keras.layers.Dense(hidden_dim)
    self.act = keras.activations.gelu
    
    self.final = None if linked_layer is not None else keras.layers.Dense(vocab_size, use_bias=True)
    self.bias = self.add_weight(shape=(vocab_size,), dtype=tf.float32, initializer="zeros", trainable=True) if linked_layer is not None else None

    self.linked_layer = linked_layer

  def __call__(self, x):
    x = self.act(self.transform(x))
    if self.linked_layer is not None:
      w = self.linked_layer.weights[0] #VOCAB_SIZE x HIDDEN
      w = tf.expand_dims(tf.expand_dims(tf.transpose(w, perm=[1, 0]), 0), 0) #1 x 1 x HIDDEN x VOCAB_SIZE
      b = tf.expand_dims(tf.expand_dims(self.bias, 0), 0) #1 x 1 x VOCAB_SIZE

      x = tf.einsum('...d,...dv->...v', x, w)
      x = x + b
    else:
      x = self.final(x)
    return x
    
class BERTLoss(keras.layers.Layer):
  def __init__(self):
    super().__init__()
    self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
  def __call__(self, true, pred):
    unmasked = self.loss(y_true=tf.nn.relu(true), y_pred=pred)
    mask = tf.cast(true != -100, dtype=unmasked.dtype)
    masked   = unmasked * mask
    reduced  = tf.reduce_sum(masked) / tf.reduce_sum(mask)

    return reduced

class TModel(keras.Model):
  def __init__(self, vocab_size, seq_len, hidden_dim, num_heads, dropout_rate, num_blocks, shared_embs=False):
    super(TModel, self).__init__()

    self.embedding = TEmbedding(vocab_size, hidden_dim, seq_len, dropout_rate)
    self.blocks = [ TBlock(hidden_dim, num_heads, dropout_rate) for _ in range(num_blocks) ]
    self.out = TOutput(hidden_dim, vocab_size, self.embedding if shared_embs else None)

  def __call__(self, x):
    x = self.embedding(x)
    for block in self.blocks:
      x = block(x)
    return self.out(x)