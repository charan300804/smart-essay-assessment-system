import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Dropout, Layer
from tensorflow.keras import backend as K
from . import config

class AttentionLayer(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # x shape: (batch_size, time_steps, features)
        # e = tanh(dot(x, W) + b)
        e = K.tanh(K.dot(x, self.W) + self.b)
        # alpha = softmax(e)
        a = K.softmax(e, axis=1)
        # output = sum(alpha * x)
        output = x * a
        return K.sum(output, axis=1)

def create_model(embedding_matrix=None):
    """
    Creates and compiles the Bi-LSTM model with Attention.
    """
    input_layer = Input(shape=(config.MAX_SEQUENCE_LENGTH,), dtype='int32')

    if embedding_matrix is not None:
        embedding_layer = Embedding(
            config.VOCAB_SIZE,
            config.EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=config.MAX_SEQUENCE_LENGTH,
            trainable=False
        )(input_layer)
    else:
        embedding_layer = Embedding(
            config.VOCAB_SIZE,
            config.EMBEDDING_DIM,
            input_length=config.MAX_SEQUENCE_LENGTH,
            trainable=True
        )(input_layer)

    lstm_layer = Bidirectional(LSTM(config.LSTM_UNITS, return_sequences=True, dropout=config.DROPOUT_RATE, recurrent_dropout=config.DROPOUT_RATE))(embedding_layer)
    
    # Attention Mechanism
    attention_layer = AttentionLayer(name='attention')(lstm_layer)
    
    dropout = Dropout(config.DROPOUT_RATE)(attention_layer)
    dense_1 = Dense(64, activation='relu')(dropout)
    output_layer = Dense(1, activation='linear')(dense_1) # Regression output

    model = Model(inputs=input_layer, outputs=output_layer)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])
    
    return model
