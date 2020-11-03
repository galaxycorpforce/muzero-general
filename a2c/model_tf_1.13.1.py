import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras.preprocessing.sequence as sequence
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Concatenate
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

# This function selects the probability distribution over actions
from baselines.common.distributions import make_pdtype
from tensorflow.keras import backend as K
from baselines.a2c.utils import fc

MM_EMBEDDINGS_DIM = 50
MM_MAX_WORD_SIZE = 20
MM_MAX_SENTENCE_SIZE = 200
MM_FEATURES_SIZE = 20000
MM_MAX_VOCAB_SIZE = 5000

MM_MAX_SENTENCE_SIZE = 200
SMALL_VOCAB_SIZE = 50  # limit used for elements, status, weather, terrain, etc
EXTRA_SMALL_VOCAB_SIZE = 10  # limit used for elements, status, weather, terrain, etc
SMALL_EMBEDDINGS_DIM = 15
EXTRA_SMALL_EMBEDDINGS_DIM = 3

GUMBALL_FIELD_REMAINDER = 5

# Fully connected layer
def fc_layer(inputs, units, activation_fn=tf.nn.relu, gain=1.0):
	return tf.layers.dense(inputs,
							units=units,
							activation=activation_fn,
							kernel_initializer=tf.orthogonal_initializer(gain))


# LSTM Layer
#def lstm_layer(vocab_size=MM_MAX_VOCAB_SIZE, word_len_limit=MM_MAX_WORD_SIZE, input_length=MM_MAX_SENTENCE_SIZE):
#	return LSTM(Embedding(vocab_size, word_len_limit, input_length=input_length, mask_zero=True), dropout=0.2, recurrent_dropout=0.2, return_sequences=True)
def lstm_layer(em_input, vocab_size=MM_MAX_VOCAB_SIZE, word_len_limit=MM_MAX_WORD_SIZE, input_length=MM_MAX_SENTENCE_SIZE):
	print('em shape', em_input.shape)
	embedding = Embedding(vocab_size, word_len_limit, input_length=input_length )(em_input)
	print('emb shape', embedding.shape)
	return LSTM(units=100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(embedding)

# future
#x_train = sequence.pad_sequences(x_train, maxlen=maxlen)   # pre_padding with 0

"""
This object creates the PPO Network architecture
"""

class PPOPolicy(object):
	def __init__(self, sess, ob_space, action_space, nbatch, nsteps, reuse=False):
		# This will use to initialize our kernels
		gain = np.sqrt(2)

		self.tokenizer = Tokenizer(num_words=5000)
		# Based on the action space, will select what probability distribution type
		# we will use to distribute action in our stochastic policy (in our case DiagGaussianPdType
		# aka Diagonal Gaussian, 3D normal distribution)
		self.pdtype = make_pdtype(action_space)

		song_text_shape = ( None, 200 )


		category_embedding_shape = ( None, 1)
		embeddings = []



		girl_1_inputs_ = tf.placeholder(tf.float32, category_embedding_shape, name="girl_1_inputs_")
		girl_1_inputs_keras = tf.keras.layers.Input(tensor=girl_1_inputs_)
		embedding_size = EXTRA_SMALL_EMBEDDINGS_DIM
		embedding = Embedding(EXTRA_SMALL_VOCAB_SIZE, embedding_size, input_length=1 )(girl_1_inputs_keras)
		embeddings.append(Reshape(target_shape=(embedding_size,))(embedding))

		girl_2_inputs_ = tf.placeholder(tf.float32, category_embedding_shape, name="girl_2_inputs_")
		girl_2_inputs_keras = tf.keras.layers.Input(tensor=girl_2_inputs_)
		embedding_size = EXTRA_SMALL_EMBEDDINGS_DIM
		embedding = Embedding(EXTRA_SMALL_VOCAB_SIZE, embedding_size, input_length=1 )(girl_2_inputs_keras)
		embeddings.append(Reshape(target_shape=(embedding_size,))(embedding))

		girl_3_inputs_ = tf.placeholder(tf.float32, category_embedding_shape, name="girl_3_inputs_")
		girl_3_inputs_keras = tf.keras.layers.Input(tensor=girl_3_inputs_)
		embedding_size = EXTRA_SMALL_EMBEDDINGS_DIM
		embedding = Embedding(EXTRA_SMALL_VOCAB_SIZE, embedding_size, input_length=1 )(girl_3_inputs_keras)
		embeddings.append(Reshape(target_shape=(embedding_size,))(embedding))

		girl_4_inputs_ = tf.placeholder(tf.float32, category_embedding_shape, name="girl_4_inputs_")
		girl_4_inputs_keras = tf.keras.layers.Input(tensor=girl_4_inputs_)
		embedding_size = EXTRA_SMALL_EMBEDDINGS_DIM
		embedding = Embedding(EXTRA_SMALL_VOCAB_SIZE, embedding_size, input_length=1 )(girl_4_inputs_keras)
		embeddings.append(Reshape(target_shape=(embedding_size,))(embedding))

		current_girl_inputs_ = tf.placeholder(tf.float32, category_embedding_shape, name="current_girl_inputs_")
		current_girl_inputs_keras = tf.keras.layers.Input(tensor=current_girl_inputs_)
		embedding_size = EXTRA_SMALL_EMBEDDINGS_DIM
		embedding = Embedding(EXTRA_SMALL_VOCAB_SIZE, embedding_size, input_length=1 )(current_girl_inputs_keras)
		embeddings.append(Reshape(target_shape=(embedding_size,))(embedding))

		# Create the input placeholder
		non_category_data_input_ = tf.placeholder(tf.float32, (None, GUMBALL_FIELD_REMAINDER), name="non_category_data_input")
		combined_inputs_ = tf.placeholder(tf.float32, (None, ob_space.shape[1] + MM_EMBEDDINGS_DIM*2 ), name="combined_input")
		text_inputs_ = tf.placeholder(tf.float32, song_text_shape, name="text_input")

		available_moves = tf.placeholder(tf.float32, [None, action_space.n], name="availableActions")

		"""
		Build the model
		Embedding
		LSTM
		3 FC for spatial dependiencies
		1 common FC
		1 FC for policy (actor)
		1 FC for value (critic)
		"""
		with tf.variable_scope('model', reuse=reuse):
			# text reading LSTM
#			lt_layer = lstm_layer()
			text_inputs_keras = tf.keras.layers.Input(tensor=text_inputs_)

			text_out = lstm_layer(text_inputs_keras)

			shape = text_out.get_shape().as_list() [1:]       # a list: [None, 9, 2]
			dim = np.prod(shape)            # dim = prod(9,2) = 18
			print('text_flatten before reshape',text_out.shape)
			text_flatten = tf.reshape(text_out, [1, -1])           # -1 means "all"

			print('embeds', len(embeddings))
			merged = Concatenate(axis=-1)(embeddings)

			# This returns a tensor
			non_category_data_input_keras = tf.keras.layers.Input(tensor=non_category_data_input_)
			categorical_dense = tf.keras.layers.Dense(512, activation='relu')(merged)
			categorical_dense = Reshape(target_shape=(512,))(categorical_dense)
			non_categorical_dense = tf.keras.layers.Dense(512, activation='relu')(non_category_data_input_keras)

			combined_fields = Concatenate(axis=-1)([non_categorical_dense, categorical_dense])
			#reshape to add dimension?
			comb_shape = combined_fields.get_shape()
			combined_fields = K.expand_dims(combined_fields, 2)
			print('combined_fields expanded dim', combined_fields.get_shape())

			conv1 = Conv1D(100, 10, activation='relu', batch_input_shape=(None, combined_fields.get_shape()[1]))(combined_fields)
#			conv1 = Conv1D(100, 10, activation='relu', batch_input_shape=(None, ob_space.shape[1]))(field_inputs_)
			conv1 = Conv1D(100, 10, activation='relu')(conv1)
			conv1 = MaxPooling1D(3)(conv1)
			conv1 = Conv1D(160, 10, activation='relu')(conv1)
			conv1 = Conv1D(160, 10, activation='relu')(conv1)
			conv1 = GlobalAveragePooling1D()(conv1)
			conv1 = Dropout(0.5)(conv1)
			print('conv1 before reshape',conv1.get_shape())
			print('text_out before flatten',text_out.get_shape())

			text_out = Flatten()(text_out)
			print('text_out ater flatten',text_out.get_shape())
			text_dense = tf.keras.layers.Dense(512, activation='relu')(text_out)
			field_dense = tf.keras.layers.Dense(512, activation='relu')(conv1)
			print('text_dense after dense',text_dense.get_shape())

#			scaled_image = tf.keras.layers.Lambda(function=lambda tensors: tensors[0] * tensors[1])([image, scale])
#			fc_common_dense = Lambda(lambda x:K.concatenate([x[0], x[1]], axis=1))([text_dense, field_dense])
#			fc_common_dense = tf.keras.layers.Concatenate(axis=-1)(list([text_dense, field_dense]))
			fc_common_dense = tf.keras.layers.Concatenate(axis=-1)(list([text_dense, field_dense]))
			fc_common_dense = tf.keras.layers.Dense(512, activation='relu')(fc_common_dense)

			#available_moves takes form [0, 0, -inf, 0, -inf...], 0 if action is available, -inf if not.
			fc_act = tf.keras.layers.Dense(256, activation='relu')(fc_common_dense)
#			self.pi = tf.keras.layers.Dense(action_space.n, activation='relu')(fc_act)
			self.pi = fc(fc_act,'pi', action_space.n, init_scale = 0.01)

			# Calculate the v(s)
			h3 = tf.keras.layers.Dense(256, activation='relu')(fc_common_dense)
			fc_vf = tf.keras.layers.Dense(1, activation=None)(h3)[:,0]

#			vf = fc_layer(fc_3, 1, activation_fn=None)[:,0]
#			vf = fc_layer(fc_common_dense, 1, activation_fn=None)[:,0]

		self.initial_state = None

		"""
		# Take an action in the action distribution (remember we are in a situation
		# of stochastic policy so we don't always take the action with the highest probability
		# for instance if we have 2 actions 0.7 and 0.3 we have 30% channce to take the second)
		a0 = self.pd.sample()
		# Calculate the neg log of our probability
		neglogp0 = self.pd.neglogp(a0)
		"""

		# perform calculations using available moves lists
		availPi = tf.add(self.pi, available_moves)

		def sample():
			u = tf.random_uniform(tf.shape(availPi))
			return tf.argmax(availPi - tf.log(-tf.log(u)), axis=-1)

		a0 = sample()
		el0in = tf.exp(availPi - tf.reduce_max(availPi, axis=-1, keep_dims=True))
		z0in = tf.reduce_sum(el0in, axis=-1, keep_dims = True)
		p0in = el0in / z0in
		onehot = tf.one_hot(a0, availPi.get_shape().as_list()[-1])
		neglogp0 = -tf.log(tf.reduce_sum(tf.multiply(p0in, onehot), axis=-1))


		# Function use to take a step returns action to take and V(s)
		def step(state_in, valid_moves, ob_texts,*_args, **_kwargs):
			# return a0, vf, neglogp0
			# padd text
#			print('ob_text', ob_texts)
			for ob_text in ob_texts:
#				print('ob_text', ob_text)
				self.tokenizer.fit_on_texts([ob_text])

			ob_text_input = []
			for ob_text in ob_texts:
#				print('ob_text', ob_text)
				token = self.tokenizer.texts_to_sequences([ob_text])
				token = sequence.pad_sequences(token, maxlen=MM_MAX_SENTENCE_SIZE)   # pre_padding with 0
				ob_text_input.append(token)
#				print('token', token)
#				print('token shape', token.shape)
			orig_ob_text_input = np.array(ob_text_input)
			shape = orig_ob_text_input.shape
#			print('ob_text_input shape', shape)
			ob_text_input = orig_ob_text_input.reshape(shape[0], shape[2])

			# Reshape for conv1
#			state_in = np.expand_dims(state_in, axis=2)
			input_dict = dict({text_inputs_:ob_text_input, available_moves:valid_moves})
			input_dict.update(split_categories_from_state(state_in))

			return sess.run([a0,fc_vf, neglogp0], input_dict)

		# Function that calculates only the V(s)
		def value(state_in, valid_moves, ob_texts, *_args, **_kwargs):
			for ob_text in ob_texts:
#				print('ob_text', ob_text)
				self.tokenizer.fit_on_texts([ob_text])

			ob_text_input = []
			for ob_text in ob_texts:
#				print('ob_text', ob_text)
				token = self.tokenizer.texts_to_sequences([ob_text])
				token = sequence.pad_sequences(token, maxlen=MM_MAX_SENTENCE_SIZE)   # pre_padding with 0
				ob_text_input.append(token)
#				print('token', token)
#				print('token shape', token.shape)
			ob_text_input = np.array(ob_text_input)
			shape = ob_text_input.shape
#			print('ob_text_input shape', shape)
			ob_text_input = ob_text_input.reshape(shape[0], shape[2])

			# Reshape for conv1
#			state_in = np.expand_dims(state_in, axis=2)
			input_dict = dict({text_inputs_:ob_text_input, available_moves:valid_moves})
			input_dict.update(split_categories_from_state(state_in))

			return sess.run(fc_vf, input_dict)
#			return sess.run(vf, {field_inputs_:state_in, text_inputs_:ob_text_input, available_moves:valid_moves})

		def select_action(state_in, valid_moves, ob_texts, *_args, **_kwargs):
			for ob_text in ob_texts:
#				print('ob_text', ob_text)
				self.tokenizer.fit_on_texts([ob_text])

			ob_text_input = []
			for ob_text in ob_texts:
#				print('ob_text', ob_text)
				token = self.tokenizer.texts_to_sequences([ob_text])
				token = sequence.pad_sequences(token, maxlen=MM_MAX_SENTENCE_SIZE)   # pre_padding with 0
				ob_text_input.append(token)
#				print('token', token)
#				print('token shape', token.shape)
			ob_text_input = np.array(ob_text_input)
			shape = ob_text_input.shape
#			print('ob_text_input shape', shape)
			ob_text_input = ob_text_input.reshape(shape[0], shape[2])

			# Reshape for conv1
#			state_in = np.expand_dims(state_in, axis=2)
			input_dict = dict({text_inputs_:ob_text_input, available_moves:valid_moves})
			input_dict.update(split_categories_from_state(state_in))

			return sess.run(fc_vf, input_dict)
#			return sess.run(vf, {field_inputs_:state_in, text_inputs_:ob_text_input, available_moves:valid_moves})

		def split_categories_from_state(obs_datas):
			input_mappings = {}
			# Initialize buckets
			current_girl = np.empty([0,1], dtype=np.float32)
			girl_1 = np.empty([0,1], dtype=np.float32)
			girl_2 = np.empty([0,1], dtype=np.float32)
			girl_3 = np.empty([0,1], dtype=np.float32)
			girl_4 = np.empty([0,1], dtype=np.float32)
			non_category_data = np.empty([0,GUMBALL_FIELD_REMAINDER], dtype=np.float32)

			input_mappings[current_girl_inputs_] = current_girl
			input_mappings[girl_1_inputs_] = girl_1
			input_mappings[girl_2_inputs_] = girl_2
			input_mappings[girl_3_inputs_] = girl_3
			input_mappings[girl_4_inputs_] = girl_4
			input_mappings[non_category_data_input_] = non_category_data

			# Everything above only happens once
			for obs_data in obs_datas:

				input_mappings[current_girl_inputs_] = np.append(input_mappings[current_girl_inputs_], np.array([[obs_data[0]]]), axis=0)
				input_mappings[girl_1_inputs_] = np.append(input_mappings[girl_1_inputs_], np.array([[obs_data[1]]]), axis=0)
				input_mappings[girl_2_inputs_] = np.append(input_mappings[girl_2_inputs_], np.array([[obs_data[2]]]), axis=0)
				input_mappings[girl_3_inputs_] = np.append(input_mappings[girl_3_inputs_], np.array([[obs_data[3]]]), axis=0)
				input_mappings[girl_4_inputs_] = np.append(input_mappings[girl_4_inputs_], np.array([[obs_data[4]]]), axis=0)

				# rest of data is numeric observation
				rest_details_index = 5
				input_mappings[non_category_data_input_] = np.append(input_mappings[non_category_data_input_], np.array([obs_data[rest_details_index:]]), axis=0)

			return input_mappings


		self.availPi = availPi
		self.split_categories_from_state = split_categories_from_state
		self.text_inputs_ = text_inputs_
		self.available_moves = available_moves
		self.vf = fc_vf
#		self.fc_vf = fc_vf
		self.step = step
		self.value = value
		self.select_action = select_action
		print('this did finish')
