# -*- encoding: utf-8 -*-
''' 

"Semantic Relatedness Based Re-ranker for Text Spotting"
   FDCLSTM model [https://arxiv.org/pdf/1909.07950.pdf]
'''

import os
import re
import logging
import csv
import json
import time
import codecs
import numpy as np
import pandas as pd
import collections

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from gensim.models import KeyedVectors
from sklearn.preprocessing import StandardScaler

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Embedding, Dropout, Reshape, Flatten, LSTM, Bidirectional, ConvLSTM2D, Permute
from keras.layers.merge import concatenate
#from keras.layers.merge import merge
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras.utils import plot_model
#from keras.utils.vis_utils import plot_model

from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.layers import Activation, Dense
from keras.layers import Bidirectional
from keras.layers import Layer
from keras import backend as K
from keras.layers import *
from keras.activations import softmax

from keras.layers import Permute, concatenate, Masking, dot
from keras.models import Model
from keras.models import Model
from keras.models import load_model
from keras.callbacks import TensorBoard
#from model.layers import Masking2D, Softmax2D, MaskedConv1D, MaskedGlobalAveragePooling1D

from keras.engine import Layer, InputSpec
import tensorflow as tf

#####################################
import keras.backend as K
from keras.engine.topology import Layer
from keras import initializers,regularizers,activations,constraints
from keras.engine import InputSpec
from keras.utils import conv_utils
######################
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Input, Dropout
from keras.layers import Permute, concatenate, Masking, dot
from keras.models import Model
from keras import optimizers
from keras.models import load_model


###################
import keras
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


class Swish(Layer):
    def __init__(self, beta, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.beta = K.cast_to_floatx(beta)

	def call(self, inputs):
            return K.sigmoid(self.beta * inputs) * inputs

        def get_config(self):
            config = {'beta': float(self.beta)}
	    base_config = super(Swish, self).get_config()
	    return dict(list(base_config.items()) + list(config.items()))

	def compute_output_shape(self, input_shape):
            return input_shape


class MaskedConv1D(Layer):

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 padding=None,
                 **kwargs):
        super(MaskedConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 1, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 1, 'strides')
        self.padding = conv_utils.normalize_padding('same')
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 1, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=3)

        self.supports_masking = True

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=3,
                                    axes={channel_axis: input_dim})
        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, inputs,  mask=None):
        outputs = K.conv1d(
            inputs,
            self.kernel,
            strides=self.strides[0],
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate[0])
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if mask is not None:
            outputs *= K.cast(K.expand_dims(mask), K.floatx())

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(MaskedConv1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class RemoveMask(Layer):
    """Remove's a mask in the sense that it does not pass along the info to
    following layers
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(RemoveMask, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        return config

    def build(self, input_shape):
        pass

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

class MaskedGlobalPooling1D(Layer):
    """Abstract class for different global pooling 1D layers.
    """

    def __init__(self, **kwargs):
        super(MaskedGlobalPooling1D, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])
    
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, inputs, mask=None):
        raise NotImplementedError


class _MaskedGlobalPooling1D(Layer):
    """Abstract class for different global pooling 1D layers.
    """

    def __init__(self, **kwargs):
        super(_MaskedGlobalPooling1D, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])
    
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, inputs, mask=None):
        raise NotImplementedError


class MaskedGlobalAveragePooling1D(_MaskedGlobalPooling1D):
    """Global average pooling operation for temporal data.
    # Input shape
        3D tensor with shape: `(batch_size, steps, features)`.
    # Output shape
        2D tensor with shape:
        `(batch_size, features)`
    """

    def call(self, inputs, mask=None):
        if mask is not None:
            s = K.sum(inputs, axis=1)
            c = K.sum(K.cast(K.expand_dims(mask), K.floatx()), axis=1)
            m = s / c
        else:
            m = K.mean(inputs, axis=1)
        return m


class KMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """
    def __init__(self, k=1, axis=1, **kwargs):
        super(KMaxPooling, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

        assert axis in [1,2],  'expected dimensions (samples, filters, convolved_values),\
                   cannot fold along samples dimension or axis not in list [1,2]'
        self.axis = axis

        # need to switch the axis with the last elemnet
        # to perform transpose for tok k elements since top_k works in last axis
        self.transpose_perm = [0,1,2] #default
        self.transpose_perm[self.axis] = 2
        self.transpose_perm[2] = self.axis

    def compute_output_shape(self, input_shape):
        input_shape_list = list(input_shape)
        input_shape_list[self.axis] = self.k
        return tuple(input_shape_list)

    def call(self, x):
        # swap sequence dimension to get top k elements along axis=1
        transposed_for_topk = tf.transpose(x, perm=self.transpose_perm)

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(transposed_for_topk, k=self.k, sorted=True, name=None)[0]

        # return back to normal dimension but now sequence dimension has only k elements
        # performing another transpose will get the tensor back to its original shape
        # but will have k as its axis_1 size
        transposed_back = tf.transpose(top_k, perm=self.transpose_perm)

        return transposed_back


class Folding(Layer):

    def __init__(self, **kwargs):
        super(Folding, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], int(input_shape[2]/2))

    def call(self, x):
        input_shape = x.get_shape().as_list()

        # split the tensor along dimension 2 into dimension_axis_size/2
        # which will give us 2 tensors
        splits = tf.split(x, num_or_size_splits=int(input_shape[2]/2), axis=2)

        # reduce sums of the pair of rows we have split onto
        reduce_sums = [tf.reduce_sum(split, axis=2) for split in splits]

        # stack them up along the same axis we have reduced
        row_reduced = tf.stack(reduce_sums, axis=2)
	return row_reduced


class AttentionWithContext(Layer):

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.kernel = self.add_weight((input_shape[2], 1,),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        # word context vector uw
        self.u = self.add_weight((input_shape[1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # in the paper refer equations (5) on page 3
        # (batch, time_steps, 40) x (40, 1)
        W_w_dot_h_it =  K.dot(x, self.kernel) # (batch, 40, 1)
        W_w_dot_h_it = K.squeeze(W_w_dot_h_it, -1) # (batch, 40)
        W_w_dot_h_it = W_w_dot_h_it + self.b # (batch, 40) + (40,)
        uit = K.tanh(W_w_dot_h_it) # (batch, 40)

        # in the paper refer equations (6) on page 3
        uit_dot_uw = uit * self.u # (batch, 40) * (40, 1) => (batch, 1)
        ait = K.exp(uit_dot_uw) # (batch, 1)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            mask = K.cast(mask, K.floatx()) #(batch, 40)
            ait = mask*ait #(batch, 40) * (batch, 40, )

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number epsilon to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        # sentence vector si is returned
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[-1]

    def compute_output_shape(self, input_shape):
        """Shape transformation logic so Keras can infer output shape
        """
	return (input_shape[0], input_shape[-1],)


class Attention(Layer):

    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim



def elapsed_time(start_time, end_time):
    elapsed_sec = end_time - start_time
    h = int(elapsed_sec / (60 * 60))
    m = int((elapsed_sec % (60 * 60)) / 60)
    s = int(elapsed_sec % 60)
    return "{}:{:>02}:{:>02}".format(h, m, s)


def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    
    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)   


def generate_embedding_matrix(word_index, word2vec, embedding_dim):
    # generate embedding matrix from word vectors
    logging.info('Preparing embedding matrix')
    nb_words = len(word_index) + 1        
    embedding_matrix = np.zeros((nb_words, embedding_dim))
    for word, i in word_index.items():
        if word in word2vec.vocab:
            embedding_matrix[i] = word2vec.word_vec(word)
    logging.info('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))    
    return embedding_matrix

# load GloVe word vectors
class glove_word2vec(object):
    def __init__(self, embedding_file):
        self.vocab = {}
        with open(embedding_file, 'rb') as f:
            for line in f:
                sline = line.split()
                self.vocab[sline[0]] = map(float, sline[1:])
    def word_vec(self, word):
        return self.vocab[word]


class Solver(object):   
    def __init__(self, config_file):
        self.name = re.search('(\S+)\.json', os.path.basename(config_file)).group(1)
        if not os.path.isdir(self.name):
            os.mkdir(self.name)
        self.output_dir = self.name + '/'
        
        self.timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")

        # load parameters from config json file
        with open(config_file, 'r') as f:
            self.params = json.load(f) 

        # random initialization seed
        self.seed = self.params.get('seed', np.random.random_integers(100))

        # for hyper-parameters specified as "random" in config file, assign random value.
        logging.info('Assign random value for hyper-parameters if necessary')
        if self.params.get('num_lstm', None) == "random":
            self.params['num_lstm'] = np.random.randint(175, 275)
            print("num_lstm: %d" % self.params['num_lstm'])
        if self.params.get('num_dense', None) == "random":
            self.params['num_dense'] = np.random.randint(100, 150)
            print("num_dense: %d" % self.params['num_dense'])
        if self.params.get('rate_drop_lstm', None) == "random":
            self.params['rate_drop_lstm'] = 0.15 + np.random.rand() * 0.25
            print("rate_drop_lstm: %f" % self.params['rate_drop_lstm'])
        if self.params.get('rate_drop_dense', None) == "random":
            self.params['rate_drop_dense'] = 0.15 + np.random.rand() * 0.25   
            print("rate_drop_dense: %f" % self.params['rate_drop_dense'])  


    def load_word2vec(self):
        # read pre-trained word vectors
        logging.info('Load word embeddings')
        
        if self.params['embedding_file_type'] == 'word2vec':
            self.word2vec = KeyedVectors.load_word2vec_format(self.params['embedding_file'], binary=True)


        elif self.params['embedding_file_type'] == 'glove':            
            self.word2vec = glove_word2vec(self.params['embedding_file'])
                
        logging.info('Found %s word vectors of word2vec' % len(self.word2vec.vocab))

 
	#def load_embedding(self):
        # prepare word embeddings
        #self.nb_words2 = len(self.word_index) + 1 
        #self.embedding_matrix = generate_embedding_matrix(self.word_index, self.word2vec2, self.params['embedding_dim2'])  

	

    def load_data(self):
        # process texts in datasets
        texts_1, texts_2, labels, ids = self.load_train_data(self.params['train_data_file'])
        logging.info('Found %s texts in train data' % len(texts_1))
    
        test_texts_1, test_texts_2, test_ids = self.load_test_data(self.params['test_data_file'])
        logging.info('Found %s texts in test data' % len(test_texts_1)) 
    
        tokenizer = Tokenizer(num_words=self.params['max_nb_words'])
        tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)        
    
        sequences_1 = tokenizer.texts_to_sequences(texts_1)
        sequences_2 = tokenizer.texts_to_sequences(texts_2)
        test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
        test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)
    
        word_index = tokenizer.word_index
        logging.info('Found %s unique tokens' % len(word_index))
    
        data_1 = pad_sequences(sequences_1, maxlen=self.params['max_seq_len'])
        data_2 = pad_sequences(sequences_2, maxlen=self.params['max_seq_len'])
        labels = np.array(labels)
        logging.info('Shape of data tensor: {}'.format(data_1.shape))
        logging.info('Shape of label tensor: {}'.format(labels.shape)) 

        feats, test_feats = self.build_features()
        logging.info('Built %s additional features' % self.feats_dim)
        
        train_data, val_data = self.train_val_split(data_1, data_2, feats, labels)
    
        test_data_1 = pad_sequences(test_sequences_1, maxlen=self.params['max_seq_len'])
        test_data_2 = pad_sequences(test_sequences_2, maxlen=self.params['max_seq_len'])
        test_ids = np.array(test_ids)
         
        test_data = {'X': [np.vstack((test_data_1, test_data_2)), np.vstack((test_data_2, test_data_1)), np.vstack((test_feats, test_feats))],
                     'ids': np.concatenate((test_ids, test_ids)),
                     }
        
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.word_index = word_index

    @staticmethod
    def load_train_data(train_data_file):
        texts_1 = [] 
        texts_2 = []
        labels = []
        ids = []
        with codecs.open(train_data_file, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)
            for values in reader:
                texts_1.append(text_to_wordlist(values[3]))
                texts_2.append(text_to_wordlist(values[4]))
                labels.append(int(values[5]))
                ids.append(values[0])
        return texts_1, texts_2, labels, ids

    @staticmethod
    def load_test_data(test_data_file):
        test_texts_1 = []
        test_texts_2 = []
        test_ids = []
        with codecs.open(test_data_file, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)
            for values in reader:
                test_texts_1.append(text_to_wordlist(values[1]))
                test_texts_2.append(text_to_wordlist(values[2]))
                test_ids.append(values[0])
        return test_texts_1, test_texts_2, test_ids   

    def build_features(self):
        # In addition to text sequence features, build additional statistical features into the model
        train_df = pd.read_csv(self.params['train_data_file'])
        test_df = pd.read_csv(self.params['test_data_file']) 
        
        ques = pd.concat([train_df[['word', 'visual_context']], \
                          test_df[['word', 'visual_context']]], axis=0).reset_index(drop='index')
        q_dict = collections.defaultdict(set)
        for i in range(ques.shape[0]):
            q_dict[ques.word[i]].add(ques.visual_context[i])
            q_dict[ques.visual_context[i]].add(ques.word[i])
                 
        #def word_freq(row):
	def word_freq(row):
            return(len(q_dict[row['word']]))
        
        #def visual_context_freq(row):
	def visual_context_freq(row):
            return(len(q_dict[row['visual_context']]))
        
        #def word_visual_context_intersect(row):
	def word_visual_context_intersect(row):
            return(len(set(q_dict[row['word']]).intersection(set(q_dict[row['visual_context']])))) 
        
	# overlaping layers with dictionary
        train_df['word_visual_context_intersect'] = train_df.apply(word_visual_context_intersect, axis=1, raw=True)
        train_df['word_freq'] = train_df.apply(word_freq, axis=1, raw=True)
        train_df['visual_context_freq'] = train_df.apply(visual_context_freq, axis=1, raw=True)
        
        test_df['word_visual_context_intersect'] = test_df.apply(word_visual_context_intersect, axis=1, raw=True)
        test_df['word_freq'] = test_df.apply(word_freq, axis=1, raw=True)
        test_df['visual_context_freq'] = test_df.apply(visual_context_freq, axis=1, raw=True)
        
        feats = train_df[['word_visual_context_intersect', 'word_freq', 'visual_context_freq']]
        test_feats = test_df[['word_visual_context_intersect', 'word_freq', 'visual_context_freq']]
        
        ss = StandardScaler()
        ss.fit(np.vstack((feats, test_feats)))
        feats = ss.transform(feats)
        test_feats = ss.transform(test_feats)
        
        self.feats_dim = feats.shape[1]
        
        return feats, test_feats

    #def train_val_split(self, data_1, data_2, feats, labels, validation_split=0.1):
    def train_val_split(self, data_1, data_2, feats, labels, validation_split=0.2):
        logging.info('Random seed for train validaiton data split: %s' % self.seed)
        np.random.seed(self.seed)
        perm = np.random.permutation(len(data_1))
        idx_train = perm[:int(len(data_1)*(1-validation_split))]
        idx_val = perm[int(len(data_1)*(1-validation_split)):]
        
        data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
        data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
        feats_train = np.vstack((feats[idx_train], feats[idx_train]))
        labels_train = np.concatenate((labels[idx_train], labels[idx_train]))
        
        data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
        data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
        feats_val = np.vstack((feats[idx_val], feats[idx_val]))
        labels_val = np.concatenate((labels[idx_val], labels[idx_val]))
        
        weight_val = np.ones(len(labels_val))
        if self.params['re_weight']:
            weight_val *= 0.472001959
            weight_val[labels_val==0] = 1.309028344 
            
        train_data = {'X': [data_1_train, data_2_train, feats_train],
                      'Y': labels_train
                      }    
        
        val_data = {'X': [data_1_val, data_2_val, feats_val],
                    'Y': labels_val,
                    'weight': weight_val
                    }
            
        return train_data, val_data

    def load_embedding(self):
        # prepare word embeddings
        self.nb_words = len(self.word_index) + 1 
        self.embedding_matrix = generate_embedding_matrix(self.word_index, self.word2vec, self.params['embedding_dim'])  

  
    def load_model(self):
        if self.params['model'] == 'LSTM':
            embedding_layer = Embedding(self.nb_words,
                                        self.params['embedding_dim'],
                                        weights=[self.embedding_matrix],
                                        input_length=self.params['max_seq_len'],
                                        trainable=self.params['embedding_trainable'])
            lstm_layer = LSTM(self.params['num_lstm'], dropout=self.params['rate_drop_lstm'], recurrent_dropout=self.params['rate_drop_lstm'])
            
            sequence_1_input = Input(shape=(self.params['max_seq_len'],), dtype='int32')
            embedded_sequences_1 = embedding_layer(sequence_1_input)
            x1 = lstm_layer(embedded_sequences_1)
            
            sequence_2_input = Input(shape=(self.params['max_seq_len'],), dtype='int32')
            embedded_sequences_2 = embedding_layer(sequence_2_input)
            y1 = lstm_layer(embedded_sequences_2)
            
            feats_input = Input(shape=(self.feats_dim,))
            feats_dense = Dense(self.params['num_dense']/2, activation=self.params['dense_activation'])(feats_input)                    
            
            merged = concatenate([x1, y1, feats_dense])
            merged = Dropout(self.params['rate_drop_dense'])(merged)
            merged = BatchNormalization()(merged)
            
            merged = Dense(self.params['num_dense'], activation=self.params['dense_activation'])(merged)
            merged = Dropout(self.params['rate_drop_dense'])(merged)
            merged = BatchNormalization()(merged)
            
            preds = Dense(1, activation='sigmoid')(merged)
            
            model = Model(inputs=[sequence_1_input, sequence_2_input, feats_input], outputs=preds)
            model.compile(loss='binary_crossentropy',
                          optimizer='nadam',
                          metrics=['acc']) 
         
 	# toy example with bidirectional_LSTM  
        elif self.params['model'] == "bidirectional_LSTM":
            embedding_layer = Embedding(self.nb_words,
                                        self.params['embedding_dim'],
                                        weights=[self.embedding_matrix],
                                        input_length=self.params['max_seq_len'],
                                        trainable=self.params['embedding_trainable'])
            lstm_layer = Bidirectional(LSTM(self.params['num_lstm'], dropout=self.params['rate_drop_lstm'], recurrent_dropout=self.params['rate_drop_lstm']))
            
            sequence_1_input = Input(shape=(self.params['max_seq_len'],), dtype='int32')
            embedded_sequences_1 = embedding_layer(sequence_1_input)
            x1 = lstm_layer(embedded_sequences_1)
            
            sequence_2_input = Input(shape=(self.params['max_seq_len'],), dtype='int32')
            embedded_sequences_2 = embedding_layer(sequence_2_input)
            y1 = lstm_layer(embedded_sequences_2)
            
            feats_input = Input(shape=(self.feats_dim,))
            feats_dense = Dense(self.params['num_dense']/2, activation=self.params['dense_activation'])(feats_input)                
            
            merged = concatenate([x1, y1, feats_dense])
            merged = Dropout(self.params['rate_drop_dense'])(merged)
            merged = BatchNormalization()(merged)
            
            merged = Dense(self.params['num_dense'], activation=self.params['dense_activation'])(merged)
            merged = Dropout(self.params['rate_drop_dense'])(merged)
            merged = BatchNormalization()(merged)
            
            preds = Dense(1, activation='sigmoid')(merged)
            
            model = Model(inputs=[sequence_1_input, sequence_2_input, feats_input], outputs=preds)
	    print(model.summary())
            keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,  write_graph=True, write_images=True)
            tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
            model.compile(loss='binary_crossentropy',
                          optimizer='nadam',
                          metrics=['acc'])
            


 	elif self.params['model'] == 'FDCLSTM':
                    
            embedding_layer1 = Embedding(self.nb_words,
                                        self.params['embedding_dim'],
                                        weights=[self.embedding_matrix],
                                        input_length=self.params['max_seq_len'],
                                        trainable=self.params['embedding_trainable'],
					mask_zero=False,
                                        name="features")     

		
           
	    embedding_layer2 = Embedding(self.nb_words,
                                        self.params['embedding_dim'],
                                        weights=[self.embedding_matrix],
                                        input_length=self.params['max_seq_len'],
                                        trainable=True,
					mask_zero=False)     

 	
            # Emedding 1 (fix) 
            sequence_1_input = Input(shape=(self.params['max_seq_len'],), dtype='int32')
            sequence_2_input = Input(shape=(self.params['max_seq_len'],), dtype='int32')
	
	    #
            sequence_1_input1 = Input(shape=(self.params['max_seq_len'],), dtype='int32')
            embedded_sequences_1_1 = embedding_layer1(sequence_1_input)
            
	    #
            sequence_2_input2 = Input(shape=(self.params['max_seq_len'],), dtype='int32')
            embedded_sequences_2_1 = embedding_layer1(sequence_2_input)
 
 
            
            # Embedding 2 (trainable) 
            sequence_1_input3 = Input(shape=(self.params['max_seq_len'],), dtype='int32')
            embedded_sequences_1_2 = embedding_layer2(sequence_1_input)
            
            sequence_2_input4 = Input(shape=(self.params['max_seq_len'],), dtype='int32')
            embedded_sequences_2_2 = embedding_layer2(sequence_2_input)
             

            # For Dual embedding static and dynamic 
            embedded_sequences_1 =  concatenate([embedded_sequences_1_1, embedded_sequences_1_2])
            embedded_sequences_2  = concatenate([embedded_sequences_2_1, embedded_sequences_2_2])
            embedded_sequences_3 = concatenate([embedded_sequences_1,embedded_sequences_2])

             
	    # MaskConv layer  k 3 
            sequence_merged1 = concatenate([embedded_sequences_1, embedded_sequences_2])
            conv1 = MaskedConv1D(filters=100, kernel_size=3, padding="same")(sequence_merged1)
	    batch_1 = BatchNormalization()(conv1)
            act1 = Activation('relu')(batch_1)
	    flat1 = Flatten()(act1)
            

	    #embedding2 = Embedding(vocab_size, 100)(inputs2)
            # Conv_1 layer k = 3
	    sequence_merged2 = concatenate([embedded_sequences_1, embedded_sequences_2])
	    conv2 = Conv1D(filters=100, kernel_size=3,padding="same")(embedded_sequences_2) # before was 5
            batch_2 = BatchNormalization()(conv2)
            act2 = Activation('relu')(batch_2)
	    flat2 = Flatten()(act2)

	    # Conv_2 layer k = 5 
            sequence_merged3 = concatenate([embedded_sequences_1, embedded_sequences_2])
	    conv3 = Conv1D(filters=100, kernel_size=5, padding="same")(sequence_merged3)
	    batch_3 = BatchNormalization()(conv3)
            act3 = Activation('relu')(batch_3)
	    flat3 = Flatten()(act3)
		
	    # Conv_3 layer k = 8  
	    sequence_merged4 = concatenate([embedded_sequences_1, embedded_sequences_2])
	    conv4 = Conv1D(filters=100, kernel_size=8, padding="same")(sequence_merged4)
	    batch_4 = BatchNormalization()(conv4)
            act4 = Activation('relu')(batch_4)
	    flat4= Flatten()(act4)
		

            # Conv joint layer 
	    ALL2 = concatenate([flat1, flat2,flat3, flat4]) # flat  
	    ALL = concatenate([act1, act2, act3, act4])
	    #ALL2 = concatenate([act1, act2, act3, act4])

            # LSTM layer 
	    rnn_layer= LSTM(64, return_sequences=True)(ALL)#25
            #rnn_layer= Bidirectional(LSTM(32, return_sequences=True))(ALL)#25 ## very bad
	    #rnn_layer= LSTM(64, return_sequences=True)(ALL)   
	    
	    # attention layer 
	    attention = Attention((self.params['max_seq_len']))(rnn_layer)
    	    #x = Dense(128, activation="relu")(attention)      
	    #output_layer = Dense(9, activation='softmax')(rnn_layer)
            #rnn_layer = Bidirectional(LSTM(64))(ALL)
	    output_layer = Dense(128, activation='relu')(rnn_layer) 
              

     
	    # overlapping layer --> FC             
            feats_input = Input(shape=(self.feats_dim,))
            feats_dense = Dense(self.params['num_dense']/2, activation=self.params['dense_activation'])(feats_input)
	    #feats_dense = Dense(self.params['num_dense']/2, activation=self.params['dense_activation'])(feats_input)  
            

	    # join layer 
	    merged = concatenate([feats_dense, attention, ALL2])
            #merged = concatenate([feats_dense, rnn_layer, ALL2])
            #merged = concatenate([conv1, feats_dense, rnn_layer])
            merged = BatchNormalization()(merged)
            
            # MLP  
            merged = Dense(200)(merged) 
            merged = BatchNormalization()(merged)   
            merged = Activation('relu')(merged)  
            #merged = Dropout(self.params['rate_drop_dense'])(merged)
	    #merged = Dropout(0.7)(merged)
            #merged = BatchNormalization()(merged)

 	    #merged = Dense(self.params['num_dense'], activation=self.params['dense_activation'])(merged)
            merged = Dense(200)(merged) 
            merged = BatchNormalization()(merged)
            merged = Activation('relu')(merged) 

	    preds = Dense(1, activation='sigmoid')(merged)
           

            model = Model(inputs=[sequence_1_input, sequence_2_input, feats_input], outputs=preds)
            print(model.summary())
            #tensorboard = TensorBoard(batch_size=2048,
            #              embeddings_freq=1,
            #              embeddings_layer_names=['features'],
            #              embeddings_metadata='metadata.tsv')
			  #embeddings_data=self.test_data['X'])

	    #keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,  write_graph=True, write_images=False)
           # tbCallBack = keras.callbacks.TensorBoard(log_dir='/Graph', histogram_freq=0, write_graph=True, write_images=False)
            
            model.compile(loss='binary_crossentropy',
                           optimizer='nadam',
                           #optimizer='adam',
                          metrics=['acc'])
   
        self.model = model
    
    def preprocess(self):
        self.load_word2vec()
        self.load_data()
        self.load_embedding()
        self.load_model()       
        
    def train(self):
        if self.params['re_weight']:
            class_weight = {0: 1.309028344, 1: 0.472001959}
        else:
            class_weight = None        

        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        best_model_fname = self.output_dir + 'model_' + self.name + '_%s'%(self.timestamp) + '.h5'
        model_checkpoint = ModelCheckpoint(best_model_fname, save_best_only=True)
        
        #hist = self.model.fit(self.train_data['X'], self.train_data['Y'], \
         #                     validation_data=(self.val_data['X'], self.val_data['Y'], self.val_data['weight']), \
         #                     epochs=self.params['nb_epoches'], batch_size=2048, shuffle=True, verbose=2, \
         #                     class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])
           
	#keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,  write_graph=True, write_images=False)
        #tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=2, write_graph=True, write_images=True)
        hist = self.model.fit(self.train_data['X'], self.train_data['Y'], \
                              validation_data=(self.val_data['X'], self.val_data['Y'], self.val_data['weight']), \
                              epochs=self.params['nb_epoches'], batch_size=2048, shuffle=True, verbose=2, \
                              class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])
                              #class_weight=class_weight, callbacks=[early_stopping, model_checkpoint, tbCallBack])
        

        
	
        self.model.load_weights(best_model_fname)
        self.best_val_score = min(hist.history['val_loss'])  

    def predict(self):
        # Write the submission
        logging.info('Writing prediction')
        
        preds = self.model.predict(self.test_data['X'], batch_size=2048, verbose=2)
        preds = preds.reshape((2,-1)).mean(axis=0)
        test_ids = self.test_data['ids'].reshape((2,-1))[0,:]
        
        score = pd.DataFrame({'test_id':test_ids, 'is_similar':preds.ravel()})
        score.to_csv(self.output_dir + 'scores_' + self.name + '_%.4f'%(self.best_val_score) + '_%s'%(self.timestamp) + '.csv', index=False)


def main():
    start_time = time.time()
    
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s]: %(message)s ',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        stream=sys.stdout,
                        filemode="w"
                        )        
    
    config_file = sys.argv[1]

    FDCLSTM_scorers  = Solver(config_file)

    FDCLSTM_scorers.preprocess()    
    FDCLSTM_scorers.train()
    FDCLSTM_scorers.predict()
     
    end_time = time.time()
    logging.info("Run complete: %s elapsed" % elapsed_time(start_time, end_time))

    
if __name__ == '__main__':
    main()
