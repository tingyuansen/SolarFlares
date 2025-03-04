# Start with importing packages, some explanations are provided.
# to suppress future warning from tf + np 1.17 combination.
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
#runtimewarning is from powertransformer
warnings.filterwarnings('ignore',category=RuntimeWarning)

# to avoid dividing by zero
epsilon = 1e-5
import numpy as np
import pandas as pd
from json import JSONDecoder, JSONDecodeError  # for reading the JSON data files
import re  # for regular expressions
import os  # for os related operations
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('fivethirtyeight')
# get_ipython().run_line_magic('matplotlib', 'inline')

# the target output y=0 for no solar flare and y=1 for solar flare
# the OneHotEncoder converts y=[0] to y=[1,0] and coverts y=[1] to y=[0,1]
# the new format of y is suitable for softmax
from sklearn.preprocessing import OneHotEncoder

# for splitting data set and cross validation
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# use tensorflow.keras to build our neural networks
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, TimeDistributed, Dropout
from tensorflow.keras.layers import Flatten, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

FLOAT_TYPE = 'float64'
K.set_floatx(FLOAT_TYPE)

# Function calculating the Fischer score
def fischer_ranking_score(x_all, x_P, x_N, median=False):
    '''
    x_all, x_P, and x_N are all pandas.Series. They contain all x's,
    x's in the positive class, and x's in the negative class.
    '''
    if median:
        # use median, robust to outliers
        xbar = x_all.median()
        xbar_P = x_P.median()
        xbar_N = x_N.median()
    else:
        # use mean, the usual definition of Fischer ranking score
        xbar = x_all.mean()
        xbar_P = x_P.mean()
        xbar_N = x_N.mean()
    # the numbers of positive-class samples and negative-class samples
    n_P = x_P.shape[0]
    n_N = x_N.shape[0]
    numerator = (xbar_P - xbar)**2.0 + (xbar_N - xbar)**2.0
    denominator = ((x_P-xbar)**2.0).sum()/(n_P - 1) + ((x_N-xbar)**2.0).sum()/(n_N - 1)
    fischer_score = numerator/denominator
    return fischer_score


# Packages for filling NaNs iteratively using linear regression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# Package for yeo-johnson transformation, supposed to cure skewness and normalize the data
from sklearn.preprocessing import PowerTransformer
# Scale a feature to [0,1] range by (x-x_min)/(x_max-x_min)
from sklearn.preprocessing import MinMaxScaler
# calculate Interquantile Range
from scipy.stats import iqr


class FeatureCurator():
    def __init__(self, X_3D):
        self.data = np.array(X_3D)
        self.n_samples = self.data.shape[0]
        self.time_steps = self.data.shape[1]
        self.num_features = self.data.shape[2]
        self.data_expand = self.data.reshape(-1, self.num_features)

    # fill in nan using linear regression iteratively
    def fill_nan(self):
        self.data_expand = IterativeImputer().fit_transform(self.data_expand)
        return self

    # clip extreme outliers
    def clip_outlier(self, out_num=8.0):
        nan_cnt = np.isnan(self.data_expand).sum()
        if nan_cnt >0:
            print('''There are nan in the data. Therefore,.clip_outlier() is not executed.\n Use .fill_nan() first before .clip_outlier().''')
            pass
        else:
            IQR = iqr(self.data_expand,axis=0)
            median = np.median(self.data_expand,axis=0)
            self.data_expand = np.clip(self.data_expand, median-out_num*IQR, median+out_num*IQR)
            return self

    # yeo-johnson transformation aiming at curing skewness and non-gaussianess
    # the transformation is applied to each column (feature) independently
    def power_transform(self):
        self.data_expand = PowerTransformer().fit_transform(self.data_expand)
        return self

    # squeeze the values of each feature into the range [0,1]
    def min_max_scaler(self):
        self.data_expand = MinMaxScaler().fit_transform(self.data_expand)
        return self

    # reshape the expanded 2D data_expand of shape(n_samples*time_steps, num_features)
    # back to the 3D shape(n_samples, time_steps, num_features)
    def back_to_3D(self):
        self.data = self.data_expand.reshape(self.n_samples,self.time_steps, self.num_features)
        return self



'''Loss function suitable for imbalanced classes: weighted_bce_with_gamma'''
# FN_penalty = alpha, FP_penalty = 1-alpha
# targets are y_true, inputs are y_pred

# focal loss designed for the case when the last layer of the networ has two neurons
# with softmax activation
def focal_loss_softmax(targets, outputs, fn_penalty, gamma):
#     targets = K.variable(targets) # these two lines are for the purpose of examining whether
#     outputs = K.variable(outputs) # focal_loss_softmax returns the correct answer. Comment
#     weights = K.variable([1.0-alpha,alpha]) # them away when used for training the model
    weights = [1.0, fn_penalty] # false negative penalty >1.0
    cce_weights = K.categorical_crossentropy(targets*weights, outputs)
    wce = K.mean(cce_weights)
#     y_pred = K.max(targets*outputs,axis=1)
#     wce = K.mean(K.pow((1.0-y_pred), gamma)*cce_weights)
    return wce

# need a proper way to wrap the function
from functools import partial, update_wrapper
def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

# wrapped focal loss
wrapped_focal_loss = wrapped_partial(focal_loss_softmax, fn_penalty=np.sqrt(2.0), gamma=2.0)


# Design a function upsampling minor class while downsampling major class.

from sklearn.utils import resample, shuffle

def balance_classes(X,y):
    '''
    Arguments: X--input features of the shape (n_sample, time_steps, num_features)
               y--target outputs of the shape (n_sample, 2)
               if y[i] = [1,0] means Class 0 (no flare)
               if y[i] = [0,1] means Class 1 (flare)
    Operation: Upsample minor class and Downsample major class keeping n_sample unchanged
    Returns: X_balanced, y_balanced of the same shapes as X and y
    '''
    class_0_index = (y[:,0] == 1)
    class_1_index = (y[:,1] == 1)
    num_0 = class_0_index.sum()
    num_1 = class_1_index.sum()
    tot_num = num_0 + num_1
    half_num = int(tot_num/2)
    another_half_num = tot_num - half_num

    X_class_0 = X[class_0_index]
    X_class_1 = X[class_1_index]

    X_class_0_resampled = resample(X_class_0, n_samples=half_num, random_state=10)
    X_class_1_resampled = resample(X_class_1, n_samples=another_half_num, random_state=0)
    X_resampled = np.concatenate((X_class_0_resampled,X_class_1_resampled),axis=0)

    class_0_resampled = np.tile(np.array([1,0]),(half_num,1))
    class_1_resampled = np.tile(np.array([0,1]),(another_half_num,1))
    y_resampled = np.concatenate((class_0_resampled, class_1_resampled), axis=0)

    X_shuffled, y_shuffled = shuffle(X_resampled, y_resampled)
    return  X_shuffled, y_shuffled



# Design a metric robust to class imbalancement.

'''Metric suitable for imbalanced classes: f1_score'''
# Sensitivity = true_positivies/actual_positives = tp/(tp+fn)
# tp is true positive, fn is false negative
# sensitivity is also called 'recall', or 'true positive rate'
def sensitivity(y_true, y_pred):
    y_pred = K.clip(y_pred, 0, 1)
    true_positives = K.sum(K.round(y_true * y_pred))
    # K.clip(x,a,b) x is a tensor, a and b are numbers, clip converts any element of x falling
    # below the range [a,b] to a, and any element of x falling above the range [a,b] to b.
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    # K.epsilon >0 & <<< 1, in order to avoid division by zero.
    sen = recall = true_positives / (possible_positives + K.epsilon())
    return sen

# Precision = true_positives/predicted_positives = tp/(tp+fp)
# tp is true positive, fp is false positive
def precision(y_true, y_pred):
    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)
    true_positives = K.sum(K.round(y_true * y_pred))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    prec = true_positives / (predicted_positives + K.epsilon())
    return prec

# f1 = 2/((1/precision) + (1/sensitivity))
def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    sen = sensitivity(y_true, y_pred)
    f1 = 2*((prec*sen)/(prec + sen + K.epsilon()))
    return f1


# # Building Models
#
# Long Short Term Memory (LSTM) networks are the state of art for dealing with time series. We build two different architectures:
# * One with two layers of LSTM.
# * The other with two layers of Bidirectional LSTM (BiLSTM).
#
# Note: Bidirectional LSTM allows information to flow both directions, while the conventional LSTM only allows information to flow from 'earlier' to 'later'.



# Define a nonlinear activation Function
# Mish: self-regularized activation function out-performs ReLU heuristically
# https://arxiv.org/pdf/1908.08681v1.pdf
# softplus(x) = ln(1+e^x)
# mish(x) = x*tanh(softplux(x))
def mish(x):
    return x*K.tanh(K.softplus(x))


# model 1: two lstm layers
def classifier(hidden_size, time_steps, feature_num, learning_rate, use_weighted_loss, dropout=0.5):
    model = Sequential()
    model.add(LSTM(units=hidden_size, input_shape=(time_steps,feature_num), return_sequences=True))
    model.add(LSTM(units=hidden_size, return_sequences=True))
    model.add(Dropout(dropout)) # against overfitting

    #TimeDistributed(layerX) assigns one layerX to each hidden memory cell (neuron) in the preceding LSTM layer
    #Make sure the preceding LSTM has its return_sequences=True, meaning output for every timestep
    model.add(TimeDistributed(Dense(int(hidden_size/2), activation=mish)))

    # Flatten takes input (batch_size, any size here) to output (batch_size, -1)
    # Flatten is needed here to get the input read for the following fully-connected (Dense) layer.
    model.add(Flatten())
    model.add(Dense(y_dim)) # Dense layer has y_dim=2 neurons.
    model.add(Activation('softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if use_weighted_loss:
        model.compile(loss=wrapped_focal_loss, optimizer=optimizer, metrics=[f1_score])
    else:
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[f1_score])
    return model

# model2: replace the lstm layers in model 1 by two bidirectional lstm layers
def classifier_bidirection(hidden_size, time_steps, feature_num, learning_rate, use_weighted_loss, dropout=0.5):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=hidden_size,return_sequences=True),input_shape=(time_steps,feature_num)))
    model.add(Bidirectional(LSTM(units=hidden_size,return_sequences=True)))
    model.add(Dropout(dropout))
    model.add(TimeDistributed(Dense(int(hidden_size/2), activation=mish)))
    model.add(Flatten())
    model.add(Dense(y_dim))
    model.add(Activation('softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if use_weighted_loss:
        model.compile(loss=wrapped_focal_loss, optimizer=optimizer, metrics=[f1_score])
    else:
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[f1_score])
    return model


# # Selecting Models by Cross Validation
#
# * Split data into train and test sets.
# * Use the train set for cross validation.
# * Adjust `batch_size`, `num_epoch` (total number of epochs for training), and `initial_lr` (initail learning rate) to reach better F1 scores.
# * We find both models, LSTM & BiLSTM with A
# ttention, achieve similar F1 scores. We decide to keep both of them and combine them into an ensumble.
#
# Below, we first read data from json file and convert them to np.ndarray and the right shape.



from read_json import convert_json_data_to_nparray

path_to_data = '../input'
file_name = "fold"+str(3)+"Training.json"
fname = os.path.join(path_to_data,file_name)

# Read in time series of 25 features into all_input, correct class labels into labels, and
# the unique ids for all data samples into ids.
all_input, class_labels, ids = convert_json_data_to_nparray(path_to_data, file_name)

# Change X and y to numpy.ndarray in the correct shape.
X_all = np.array(all_input)
y_all = np.array([class_labels]).T


# We use sklearn.model_selection.StratifiedShuffleSplit to split the data set into train and test sets.
# This method makes the train and test sets to have equally balanced classes.
# Because the train set will later be used for cross validation,
# I will call the inputs and labels for cross validation `X` and `y`.
from sklearn.model_selection import StratifiedShuffleSplit

# test set is 30% of the total data set.
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
sss.get_n_splits(X_all, y_all)

for cv_index, test_index in sss.split(X_all,y_all):
    X, X_test = X_all[cv_index], X_all[test_index]
    y, y_test = y_all[cv_index], y_all[test_index]

labels = y.copy()

"""
# these two lines below read in scaled data from .npy files which have been scaled using the
# time-consuming version of Feature Curation described above.
# read in scaled features and targeted (correct) outputs
use_time_saving_feature_curation = False
X = np.load('../input/X_scaled.npy')
y = np.load('../input/y.npy')
labels = y.copy()
print('There are {} NaN in y.'.format(np.isnan(y).sum()))
print('There are {} NaN in X.'.format(np.isnan(X).sum()))
"""

# use the time saving version of feature curation
use_time_saving_feature_curation = True

# upsampling minor class (Class 1) and downsampling major class (Class 0)
# so that they each occupy 50% of the training data.
# we decide not to use it either, because we find this does a worse job on the F1 score of the validation.
use_balance_classes = False

# since we find the weighted loss function we design works worse than categorical cross entropy
# the weighted loss function keeps growing after only a few epochs.
use_weighted_loss = False


# one-hot encode y, [0] --> [1,0]; [1] --> [0, 1]
onehot_encoder = OneHotEncoder(sparse=False)
y = np.asarray(onehot_encoder.fit_transform(labels), dtype=FLOAT_TYPE)
y_dim = np.shape(y)[1] # y_dim =2 after OneHotEncoder()


# Set some hyperparameters
num_epochs = 100
time_steps = X.shape[1]
batch_size = 256 # int(n_sample/30) # was 256 for fold3
feature_num = X.shape[2]
hidden_size = feature_num
initial_lr = 0.001 # initial learning for optimizer, default for Adam is 0.001;
# SGD keeps constant learning rate with default value 0.01

# Split X, y into training and validation sets
# define k-fold cross validation test harness
seed = 10 # random seed for reproductivity

n_splits = 5 # split data set into 0.2 validation & 0.8 training for cross validation.

# sklearn.model_selection.StratifiedKFold splits data while preserving percentage of samples for each class
# sklearn.model_selection.StratifiedKFold provides indices
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


"""
# cross validation for the bidirectional lstm model `classifier_bidirection`.
# since StratifiedKFold returns indices, NO need to pass in .split(X,y),
# instead passing in .split(y,y) is enough
cv_count = 0
for train, val in kfold.split(np.asarray(labels), np.asarray(labels)):
    if cv_count >0: # only run one iteration of cross validation for the purpose of exhibition.
        break
    X_train = X[train]
    X_val = X[val]
    y_train = y[train]
    y_val = y[val]

    '''
    If we use the time-saving version of feature curation, featured are not yet curated
    beforehand and thus we need to do feature curation here.
    '''
    if use_time_saving_feature_curation:

        # feature curation for the training set
        X_train_FC = FeatureCurator(X_train)
        X_train_FC.fill_nan().power_transform().min_max_scaler().back_to_3D()
        X_train = X_train_FC.data

        # feature curation for the validation set
        X_val_FC = FeatureCurator(X_val)
        X_val_FC.fill_nan().power_transform().min_max_scaler().back_to_3D()
        X_val = X_val_FC.data

    '''
    For X_train and y_train: upsample minor and downsample major class
    so that each class each occupies 50% of the X_train and y_train.
    Make sure you do not do this to X_val and y_val
    '''
    if use_balance_classes:
        X_train, y_train = balance_classes(X_train, y_train)

    # y does not need to be cured.

    # train and validate
    clf = KerasClassifier(classifier_bidirection, hidden_size=hidden_size, time_steps=time_steps,
                          feature_num=feature_num,learning_rate = initial_lr, use_weighted_loss=use_weighted_loss,\
                          epochs=num_epochs,batch_size=batch_size, verbose=1, validation_data=(X_val,y_val))
    history = clf.fit(X_train, y_train)
    cv_count += 1


# list all data in history
print(history.history.keys())
fig, axes = plt.subplots(1,2,figsize=(12,6))
axes[0].plot(history.history['loss'])
axes[0].plot(history.history['val_loss'])
axes[0].set_xlabel('epoch')
axes[0].set_ylabel('loss')
axes[0].legend(['train','validation'], loc='upper right')
axes[1].plot(history.history['f1_score'])
axes[1].plot(history.history['val_f1_score'])
axes[1].set_xlabel('epoch')
axes[1].set_ylabel('F1 score')
fig.suptitle('Loss (left) and F1 score (left) change with training Epoch')
plt.show()
"""


"""
# cross validation for the lstm model `classifier`.
# since StratifiedKFold returns indices, NO need to pass in .split(X,y),
# instead passing in .split(y,y) is enough
cv_count = 0
for train, val in kfold.split(np.asarray(labels), np.asarray(labels)):
    if cv_count >0: # only run one iteration of cross validation for the purpose of exhibition.
        break
    X_train = X[train]
    X_val = X[val]
    y_train = y[train]
    y_val = y[val]

    '''
    If we use the time-saving version of feature curation, featured are not yet curated
    beforehand and thus we need to do feature curation here.
    '''
    if use_time_saving_feature_curation:

        # feature curation for the training set
        X_train_FC = FeatureCurator(X_train)
        X_train_FC.fill_nan().power_transform().min_max_scaler().back_to_3D()
        X_train = X_train_FC.data

        # feature curation for the validation set
        X_val_FC = FeatureCurator(X_val)
        X_val_FC.fill_nan().power_transform().min_max_scaler().back_to_3D()
        X_val = X_val_FC.data

    '''
    For X_train and y_train: upsample minor and downsample major class
    so that each class each occupies 50% of the X_train and y_train.
    Make sure you do not do this to X_val and y_val
    '''
    if use_balance_classes:
        X_train, y_train = balance_classes(X_train, y_train)

    # train and validate
    clf = KerasClassifier(classifier, hidden_size=hidden_size, time_steps=time_steps,
                          feature_num=feature_num,learning_rate = initial_lr, use_weighted_loss=use_weighted_loss,\
                          epochs=num_epochs,batch_size=batch_size, verbose=1, validation_data=(X_val,y_val))
    history = clf.fit(X_train, y_train)
    cv_count += 1

pd.DataFrame(history.history).to_csv('classifier_cv_history.csv',index=False)

# list all data in history
print(history.history.keys())
fig, axes = plt.subplots(1,2,figsize=(12,6))
axes[0].plot(history.history['loss'])
axes[0].plot(history.history['val_loss'])
axes[0].set_xlabel('epoch')
axes[0].set_ylabel('loss')
axes[0].legend(['train','validation'], loc='upper right')
axes[1].plot(history.history['f1_score'])
axes[1].plot(history.history['val_f1_score'])
axes[1].set_xlabel('epoch')
axes[1].set_ylabel('F1 score')
fig.suptitle('Loss (left) and F1 score (left) change with training Epoch')
plt.show()
"""


# train 5 models using cross validation, and save the model with the best val_f1_score at each cross validation.
cv_cnt = 0
for train, val in kfold.split(np.asarray(labels), np.asarray(labels)):
    # count the number of cross validation
    cv_cnt += 1
    print('{}th cv:'.format(cv_cnt))
    # save the model achieving the best val_f1_score
    checkpoint = tf.keras.callbacks.ModelCheckpoint('bidirection_cv_'+str(cv_cnt)+'.h5', monitor='val_f1_score', \
                                                   mode='max', save_best_only=True, verbose=1)
    # assign train and validation data
    X_train = X[train]
    X_val = X[val]
    y_train = y[train]
    y_val = y[val]

    '''
    If we use the time-saving version of feature curation, featured are not yet curated
    beforehand and thus we need to do feature curation here.
    '''
    if use_time_saving_feature_curation:

        # feature curation for the training set
        X_train_FC = FeatureCurator(X_train)
        X_train_FC.fill_nan().power_transform().min_max_scaler().back_to_3D()
        X_train = X_train_FC.data

        # feature curation for the validation set
        X_val_FC = FeatureCurator(X_val)
        X_val_FC.fill_nan().power_transform().min_max_scaler().back_to_3D()
        X_val = X_val_FC.data

    '''
    For X_train and y_train: upsample minor and downsample major class
    so that each class each occupies 50% of the X_train and y_train.
    Make sure you do not do this to X_val and y_val
    '''
    if use_balance_classes:
        X_train, y_train = balance_classes(X_train, y_train)

    # define the model
    model = classifier_bidirection(hidden_size=hidden_size, time_steps=time_steps, feature_num=feature_num,\
                       learning_rate = initial_lr, use_weighted_loss=use_weighted_loss)

    # train the model
    history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size,
                        callbacks=[checkpoint], validation_data=(X_val,y_val))
    # save the trained model
    # model.save('bidirection_cv_'+str(cv_cnt)+'.h5')

# Test on test data.

# Scale features for the test data.
if use_time_saving_feature_curation:
    # feature curation for the training set
    X_test_FC = FeatureCurator(X_test)
    X_test_FC.fill_nan().power_transform().min_max_scaler().back_to_3D()
    X_test = X_test_FC.data

# OneHot code y_test
labels_test = y_test.copy()
y_test = np.asarray(onehot_encoder.fit_transform(labels_test),dtype=FLOAT_TYPE)

# make predictions using the five saved models
pred_ls = []
for i in range(1,6):
    model_name = '../trained_models/bidirection_cv_'+str(i)+'.h5'
    model = tf.keras.models.load_model(model_name,custom_objects={'mish':mish, 'f1_score': f1_score})
    y_pred = model.predict(X_test)
    pred_ls.append(y_pred)

# Take the mean of predictions of the 5 models
pred_mean = np.array(pred_ls).mean(axis=0)

# get f1 score
f1_score_test = f1_score(K.variable(y_test), K.variable(pred_mean)).numpy()
print("The F1 score on the test data is {:.4f}.".format(f1_score_test))
