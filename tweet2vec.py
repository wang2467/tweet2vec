from lasagne.layers import InputLayer, Gate, GRULayer, DenseLayer, SliceLayer, EmbeddingLayer, ElemwiseSumLayer, get_output
from collections import OrderedDict
import theano
import theano.tensor as T
import lasagne
import numpy as np

EMBEDDING_DIM = 150
HIDDEN_DIM = 500
N_BATCH = 64
MAX_LENGTH = 200
TWEET_DIM = 500
REGULAIZATION = 0.0001
LEARNING_RATE = 0.01
MOMENTUM = 0.9

class Tweet2Vec:
	def __init__(self, n_char, n_class):
		self.n_char = n_char
		self.n_class = n_class

		E = np.random.uniform(-np.sqrt(1 / EMBEDDING_DIM), np.sqrt(1 / EMBEDDING_DIM), (n_char, EMBEDDING_DIM))
		U = np.random.uniform(-np.sqrt(1 / HIDDEN_DIM), np.sqrt(1 / HIDDEN_DIM), (6, HIDDEN_DIM, HIDDEN_DIM))
		W = np.random.uniform(-np.sqrt(1 / HIDDEN_DIM), np.sqrt(1 / HIDDEN_DIM), (6, EMBEDDING_DIM, HIDDEN_DIM))
		W_d = np.random.uniform(-np.sqrt(1 / TWEET_DIM), np.sqrt(1 / TWEET_DIM), (2, HIDDEN_DIM, TWEET_DIM))
		W_s = np.random.uniform(-np.sqrt(1 / n_class), np.sqrt(1 / n_class), (TWEET_DIM, n_class))
		b = np.zeros((6, HIDDEN_DIM))
		b_s = np.zeros(n_class)
		h = np.zeros((2, 1, HIDDEN_DIM))

		self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))
		self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
		self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
		self.W_d = theano.shared(name='W_d', value=W_d.astype(theano.config.floatX))
		self.W_s = theano.shared(name='W_s', value=W_s.astype(theano.config.floatX))
		self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
		self.b_s = theano.shared(name='b_s', value=b_s.astype(theano.config.floatX))
		self.h = theano.shared(name='h', value=h.astype(theano.config.floatX))

		self.__build__()

	def __build__(self):
		tweets = T.itensor3('tweets')
		masks = T.imatrix('masks')
		targets = T.ivector('targets')

		# Tweets input layer
		l_in = InputLayer(shape=(N_BATCH, MAX_LENGTH, 1), input_var=tweets, name='l_in')

		# Masks input layer
		l_mask = InputLayer(shape=(N_BATCH, MAX_LENGTH), input_var=masks, name='l_mask')

		# Character embedding layer
		l_char_embedding = EmbeddingLayer(incoming=l_in, input_size=self.n_char, output_size=EMBEDDING_DIM, W=self.E)

		# Forward GRU
		f_reset = Gate(W_in=self.W[0], W_hid=self.U[0], W_cell=None, b=self.b[0], nonlinearity=lasagne.nonlinearities.sigmoid)
		f_update = Gate(W_in=self.W[1], W_hid=self.U[1], W_cell=None, b=self.b[1], nonlinearity=lasagne.nonlinearities.sigmoid)
		f_hidden = Gate(W_in=self.W[2], W_hid=self.U[2], W_cell=None, b=self.b[2], nonlinearity=lasagne.nonlinearities.tanh)

		l_fgru = GRULayer(incoming=l_char_embedding, num_units=HIDDEN_DIM, resetgate=f_reset, updategate=f_update, hidden_update=f_hidden, hid_init=self.h[0], backwards=False, learn_init=True, gradient_steps=-1, grad_clipping=5, unroll_scan=False, precompute_input=True, mask_input=l_mask)

		# Backward GRU
		b_reset = Gate(W_in=self.W[3], W_hid=self.U[3], W_cell=None, b=self.b[3], nonlinearity=lasagne.nonlinearities.sigmoid)
		b_update = Gate(W_in=self.W[4], W_hid=self.U[4], W_cell=None, b=self.b[4], nonlinearity=lasagne.nonlinearities.sigmoid)
		b_hidden = Gate(W_in=self.W[5], W_hid=self.U[5], W_cell=None, b=self.b[5], nonlinearity=lasagne.nonlinearities.tanh)

		l_bgru = GRULayer(incoming=l_char_embedding, num_units=HIDDEN_DIM, resetgate=b_reset, updategate=b_update, hidden_update=b_hidden, hid_init=self.h[1], backwards=True, learn_init=True, gradient_steps=-1, grad_clipping=5, unroll_scan=False, precompute_input=True, mask_input=l_mask)

		# Slice final states
		l_f = SliceLayer(l_fgru, -1, 1)
		l_b = SliceLayer(l_bgru, 0, 1)

		# Dense layer: no bias
		l_fdense = DenseLayer(incoming=l_f, num_units=TWEET_DIM, W=self.W_d[0], b=None, nonlinearity=None)
		l_bdense = DenseLayer(incoming=l_b, num_units=TWEET_DIM, W=self.W_d[1], b=None, nonlinearity=None)

		# Merge layer: element-wise summation
		l_merge = ElemwiseSumLayer([l_fdense, l_bdense], coeffs=1)

		# Softmax output
		l_res = DenseLayer(incoming=l_merge, num_units=self.n_class, W=self.W_s, b=self.b_s, nonlinearity=lasagne.nonlinearities.softmax)

		# Loss
		prediction = get_output(l_res)
		loss = lasagne.objectives.categorical_crossentropy(prediction, targets)

		# Add regularization
		cost = T.mean(loss) + REGULAIZATION * lasagne.regularization.regularize_network_params(l_res, lasagne.regularization.l2)

		# SGD + Nesterov Momentum
		updates = lasagne.updates.nesterov_momentum(cost, lasagne.layers.get_all_params(l_res), LEARNING_RATE, momentum=MOMENTUM)

		self.train = theano.function(inputs=[tweets, masks, targets], outputs=cost, updates=updates, allow_input_downcast=True)
		self.predict = theano.function(inputs=[tweets, masks], outputs=prediction)

	def getParams(self):
		params = OrderedDict()

		params['W'] = self.W.get_value()
		params['E'] = self.E.get_value()
		params['U'] = self.U.get_value()
		params['W_d'] = self.W_d.get_value()
		params['W_s'] = self.W_s.get_value()
		params['b'] = self.b.get_value()
		params['b_s'] = self.b_s.get_value()
		params['h'] = self.h.get_value()

		return params

if __name__ == "__main__":
	a = Tweet2Vec(200, 200)




