import tensorflow as tf

class SpeechModel_cnn_lstm():
    
    def __init__(self, inputs, outputs, save_nodes):
        self.inputs = inputs
        self.outputs = outputs
        self.save_nodes = save_nodes
        
    def initialize(self):
        self.targets = [tf.sparse_placeholder(tf.int32)]
        
    def visualize(self):
        pass
    
def LSTM_layer(X, hidden_dim, state_np_init, n_time, return_sequence=False, inp_layer=True,
              scope='lstm_layer0'):
    """
    Creates one unidirectional LSTM layer.

    args:
        X: input placeholder of shape [batch_size, n_time*features_dim]
        hidden_dim: int, dimension of hidden states of LSTM
        state_np_init: initial state of shape [batch_size, hidden_dim]
        n_time: int, number of time steps
        return_sequence: bool, whether or not to return entire sequence of hidden states

    returns:
        outputs: either last hidden state or entire sequence
    """

    # split operation only support the shape[axis] with integer multiple of 16
    if inp_layer:
        X_in = tf.split(X, n_time, 1)
    else:
        X_in = X

    # define LSTM cell
    lstm_cell = tf.contrib.rnn.LSTMCell(hidden_dim)

    # create initial state
    cell_state = tf.convert_to_tensor(state_np_init, dtype=tf.float32)
    hidden_state = tf.convert_to_tensor(state_np_init, dtype=tf.float32)
    state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)

    with tf.variable_scope(scope):
        outputs, states = tf.nn.static_rnn(lstm_cell, X_in, initial_state=state, dtype=tf.float32)

    if return_sequence:
        return outputs, states
    else:
        return outputs[-1]
    
def get_lstm(config):
    inputs = tf.placeholder(tf.float32, shape=(None, config['t_step']*config['signal_dim']))
    
    state_init = tf.placeholder(tf.float32, shape=(None, config['hidden_dim']))

    seq_length = tf.placeholder(name='label_length', shape=(None,), dtype='int32')

    encoding, states = LSTM_layer(inputs, config['hidden_dim'], state_init, 
                                  config['t_step'], return_sequence=True)

    out = tf.concat(encoding, axis=1, name='out')
    logits = tf.reshape(encoding, shape=(config['t_step'], tf.shape(out)[0], config['hidden_dim']))
    logits = logits[:, :, :config['num_classes']]
    
    model = SpeechModel_cnn_lstm([inputs, state_init, seq_length], [logits], [out])
    
    return model


def get_lstm_t_dense(config):
    inputs = tf.placeholder(tf.float32, shape=(None, config['t_step']*config['signal_dim']))
    
    state_init = tf.placeholder(tf.float32, shape=(None, config['hidden_dim']))

    encoding, states = LSTM_layer(inputs, config['hidden_dim'], state_init, 
                                  config['t_step'], return_sequence=True)

    out = []
    for i in range(len(encoding)):
        
        try:
            with tf.variable_scope("t_distributed", reuse=True):
                out.append(tf.layers.dense(encoding[i], config['num_classes_internal']))
        except:
            with tf.variable_scope("t_distributed", reuse=None):
                out.append(tf.layers.dense(encoding[i], config['num_classes_internal']))
    
    out = tf.concat(out, axis=1, name='out')

    logits = tf.reshape(out, shape=(config['t_step'], tf.shape(out)[0], config['num_classes_internal']))
    logits = logits[:, :, :config['num_classes']]
    
    model = SpeechModel_cnn_lstm([inputs, state_init, seq_length], [logits], [out])
    
    return model


def get_cnn_lstm_t_dense(config):
    inputs = tf.placeholder(tf.float32, shape=(None, config['t_step']*config['signal_dim']))
    
    state_init = tf.placeholder(tf.float32, shape=(None, config['hidden_dim']))
    
    x = tf.split(inputs, config['t_step'], 1)
    for i in range(len(x)):
        
        try:
            with tf.variable_scope("t_distributed_cnn", reuse=True):
                x[i] = tf.layers.dense(x[i], config['hidden_dim'], activation=tf.nn.relu)
        except:
            with tf.variable_scope("t_distributed_cnn", reuse=None):
                x[i] = tf.layers.dense(x[i], config['hidden_dim'], activation=tf.nn.relu)
                
        try:
            with tf.variable_scope("t_distributed_cnn_dense", reuse=True):
                x[i] = tf.layers.dense(x[i], config['hidden_dim'], activation=tf.nn.relu)
        except:
            with tf.variable_scope("t_distributed_cnn_dense", reuse=None):
                x[i] = tf.layers.dense(x[i], config['hidden_dim'], activation=tf.nn.relu)

    encoding, states = LSTM_layer(x, config['hidden_dim'], state_init, 
                                  config['t_step'], return_sequence=True, inp_layer=False)

    out = []
    for i in range(len(encoding)):

        try:
            with tf.variable_scope("t_distributed_res", reuse=True):
                encoding[i] = tf.layers.dense(encoding[i], config['hidden_dim'], activation=tf.nn.relu)
        except:
            with tf.variable_scope("t_distributed_res", reuse=None):
                encoding[i] = tf.layers.dense(encoding[i], config['hidden_dim'], activation=tf.nn.relu)
        
        try:
            with tf.variable_scope("t_distributed_pred", reuse=True):
                out.append(tf.layers.dense(encoding[i], config['num_classes_internal']))
        except:
            with tf.variable_scope("t_distributed_pred", reuse=None):
                out.append(tf.layers.dense(encoding[i], config['num_classes_internal']))
    
    out = tf.concat(out, axis=1, name='out')

    logits = tf.reshape(out, shape=(tf.shape(out)[0], config['t_step'], config['num_classes_internal']))
    logits = logits[:, :, :config['num_classes']]
    logits = tf.nn.softmax(logits, dim=2)
    
    logits = tf.check_numerics(logits, 'after logits')
    
    model = SpeechModel_cnn_lstm([inputs, state_init], [logits], [out])
    
    return model