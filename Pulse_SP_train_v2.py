#!/usr/local/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys,os

import numpy as np
import time
import h5py
import random

try:
    import matplotlib
    matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    print("Worked")
except:
    "Didn't work"
    pass

import numpy as np
import tensorflow as tf
import glob
import psrchive as psr

tf.logging.set_verbosity(tf.logging.INFO)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import merge as Merger
from keras.layers import Conv1D, Conv2D
from keras.layers import MaxPooling2D, MaxPooling1D, GlobalAveragePooling1D, BatchNormalization
from keras.optimizers import SGD
from keras.models import load_model

def construct_conv2d(features_only=False, fit=False,
                     train_data=None, train_labels=None,
                     eval_data=None, eval_labels=None,
                     nfreq=16, ntime=250, epochs=5,
                     nfilt1=32, nfilt2=64, batch_size=32):
    """ Build a two-dimensional convolutional neural network
    with a binary classifier. Can be used for, e.g.,
    freq-time dynamic spectra of pulsars, dm-time intensity array.

    Parameters:
    ----------
    features_only : bool 
        Don't construct full model, only features layers 
    fit : bool 
        Fit model 
    train_data : ndarray
        (ntrain, ntime, 1) float64 array with training data
    train_labels :  ndarray
        (ntrigger, 2) binary labels of training data [0, 1] = FRB, [1, 0]=RFI 
    eval_data : ndarray
        (neval, ntime, 1) float64 array with evaluation data
    eval_labels : 
        (neval, 2) binary labels of eval data 
    epochs : int 
        Number of training epochs 
    nfilt1 : int
        Number of neurons in first hidden layer 
    nfilt2 : int 
        Number of neurons in second hidden layer 
    batch_size : int 
        Number of batches for training   
       
    Returns
    -------
    model : XX

    score : np.float 
        accuracy, i.e. fraction of predictions that are correct 

    """

    if train_data is not None:
        nfreq=train_data.shape[1]
        ntime=train_data.shape[2]
    
    print(nfreq,ntime)
    model = Sequential()
    # this applies 32 convolution filters of size 5x5 each.
    model.add(Conv2D(nfilt1, (5, 5), activation='relu', input_shape=(nfreq, ntime, 1)))

    #model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Randomly drop some fraction of nodes (set weights to 0)
    model.add(Dropout(0.4))
    model.add(Conv2D(nfilt2, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())

    if features_only is True:
        model.add(BatchNormalization()) # hack
        return model, []

    model.add(Dense(256, activation='relu')) # should be 1024 hack

#    model.add(Dense(1024, activation='relu')) # remove for now hack
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # train_labels = keras.utils.to_categorical(train_labels)
    # eval_labels = keras.utils.to_categorical(eval_labels)

    if fit is True:
        print("Using batch_size: %d" % batch_size)
        print("Using %d epochs" % epochs)
        cb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0,
                                         batch_size=32, write_graph=True, write_grads=False,
                                         write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                                         embeddings_metadata=None)

        model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, callbacks=[cb])
        score = model.evaluate(eval_data, eval_labels, batch_size=batch_size)
        print("Conv2d only")
        print(score)

    return model, score

def get_classification_results(y_true, y_pred):
    """ Take true labels (y_true) and model-predicted 
    label (y_pred) for a binary classifier, and return 
    true_positives, false_positives, true_negatives, false_negatives
    """

    true_positives = np.where((y_true==1) & (y_pred==1))[0]
    false_positives = np.where((y_true==0) & (y_pred==1))[0]
    true_negatives = np.where((y_true==0) & (y_pred==0))[0]
    false_negatives = np.where((y_true==1) & (y_pred==0))[0]

    return true_positives, false_positives, true_negatives, false_negatives

def confusion_mat(y_true, y_pred):
    """ Generate a confusion matrix for a
    binary classifier based on true labels (
    y_true) and model-predicted label (y_pred)

    returns np.array([[TP, FP],[FN, TN]])
    """
    TP, FP, TN, FN = get_classification_results(y_true, y_pred)

    NTP = len(TP)
    NFP = len(FP)
    NTN = len(TN)
    NFN = len(FN)
    
    conf_mat = np.array([[NTP, NFP],[NFN, NTN]])

    return conf_mat

def print_metric(y_true, y_pred):
    """ Take true labels (y_true) and model-predicted 
    label (y_pred) for a binary classifier
    and print a confusion matrix, metrics, 
    return accuracy, precision, recall, fscore
    """
    conf_mat = confusion_mat(y_true, y_pred)

    NTP, NFP, NTN, NFN = conf_mat[0,0], conf_mat[0,1], conf_mat[1,1], conf_mat[1,0]

    print("Confusion matrix:")

    print('\n'.join([''.join(['{:8}'.format(item) for item in row])
      for row in conf_mat]))

    accuracy = float(NTP + NTN)/conf_mat.sum()
    precision = float(NTP) / (NTP + NFP + 1e-19)
    recall = float(NTP) / (NTP + NFN + 1e-19)
    fscore = 2*precision*recall/(precision+recall)

    print("accuracy: %f" % accuracy)
    print("precision: %f" % precision)
    print("recall: %f" % recall)
    print("fscore: %f" % fscore)

    return accuracy, precision, recall, fscore

def injectFRB(data):
    '''
    inject FRB in input numpy array
    '''
    nchan = data.shape[0]
    nbins = data.shape[1]    
    
    wid = 5 # Maximum width of the injected burst in number of bins

    st = random.randint(0, nbins-random.randint(0,wid)) # Random point to inject FRB

    select = data[...,st:st+wid]
    prof = np.mean(data,axis=0)
    
    #Very simple broadband pulse
    data[...,st:st+wid] = data[...,st:st+wid] + random.randint(6,50)*np.std(prof) 

    return data

def psr2np(fname,NCHAN,dm):
    #Get psrchive file as input and outputs numpy array
    fpsr = psr.Archive_load(fname)
    fpsr.dededisperse() 
    fpsr.set_dispersion_measure(dm)
    fpsr.dedisperse()

    fpsr.fscrunch_to_nchan(NCHAN)
    fpsr.remove_baseline()
    
    #-- apply weights for RFI lines --#
    ds = fpsr.get_data().squeeze()
    w = fpsr.get_weights().flatten()
    w = w/np.max(w)
    idx = np.where(w==0)[0]
    ds = np.multiply(ds, w[np.newaxis,:,np.newaxis])
    ds[:,idx,:] = np.nan

    #-- Get total intensity data (I) from the full stokes --#
    data = ds[0,:,:]

    #-- Get frequency axis values --#
    freq = np.linspace(fpsr.get_centre_frequency()-abs(fpsr.get_bandwidth()/2),fpsr.get_centre_frequency()+abs(fpsr.get_bandwidth()/2),fpsr.get_nchan())
    
    #-- Get time axis --#
    tbin = float(fpsr.integration_length()/fpsr.get_nbin())
    taxis = np.arange(0,fpsr.integration_length(),tbin)
    # Convert to time to msec
    taxis = taxis*1000

    return data


if __name__ == "__main__":

    # Read archive files and extract data arrays

    path = sys.argv[1] # Path and Pattern to find all the .ar files to read and train on
    NFREQ = 64
    NTINT = 512

    if path is not None:
        #files = glob.glob(path+"1stCand*.ar")
        files = glob.glob(path+"*.ar")
    else:    
        #files = glob.glob("1stCand*.ar")
        files = glob.glob("*.ar")
   
    ftdata = [] 
    label = []

    for fl in files:
        '''
        cmd = "pdv -t " + fl + " | awk '{print$4}' >  test.text"
        print(cmd)
        os.system(cmd)
        data = np.loadtxt("test.text",skiprows=1) 
        data = np.reshape(data,(NFREQ,NTINT)) 
        ftdata.append(data)
        '''
        #ar file with FRB
        data = []
        #data = psr2np(fl,NFREQ,30)
        ftdata.append(psr2np(fl,NFREQ,30))
        label.append(0)
        #ar file with injected FRB
        data1 = []
        data1 = injectFRB(psr2np(fl,NFREQ,30))
        ftdata.append(data1)
        label.append(1)

    ftdata = np.array(ftdata)

    if ftdata is not None:
          Nfl = ftdata.shape[0]
          nfreq=ftdata.shape[1]
          ntime=ftdata.shape[2]
  
    print(Nfl,nfreq,ntime)
    print(label)

    dshape = ftdata.shape

    # normalize data
    ftdata = ftdata.reshape(len(ftdata), -1)
    ftdata -= np.median(ftdata, axis=-1)[:, None]
    ftdata /= np.std(ftdata, axis=-1)[:, None]

    # zero out nans
    ftdata[ftdata!=ftdata] = 0.0
    ftdata = ftdata.reshape(dshape)

    #Get 4D vector for Keras
    ftdata = ftdata[..., None]

    NTRAIN = int(len(label)*0.5)

    ind = np.arange(Nfl)
    np.random.shuffle(ind)

    #print(ind,NTRAIN)

    ind_train = ind[:NTRAIN]
    ind_eval = ind[NTRAIN:]

    #print(ind_train)
    #print(ind_eval)

    #train_labels, eval_labels = label[ind_train], label[ind_eval]
    #train_labels = [1,0,1]
    #eval_labels = [1,0]

    '''
    Just a short cut because no internet 
    print(ind_train,ind_eval)
    train_labels = label[ind_train]
    eval_labels = label[ind_eval]
    '''

    train_labels = []
    eval_labels = []

    for i in ind_train: train_labels.append(label[i])
    for j in ind_eval: eval_labels.append(label[j])

    #print(train_labels,ind_train)
    #print(eval_labels,ind_eval)
    eval_label1 = np.array(eval_labels)

    train_labels = keras.utils.to_categorical(train_labels)
    eval_labels = keras.utils.to_categorical(eval_labels)

    train_data_freq, eval_data_freq = ftdata[ind_train], ftdata[ind_eval]

    # Fit convolution neural network to the training data
    model_freq_time, score_freq_time = construct_conv2d(
                            features_only=False, fit=True,
                            train_data=train_data_freq, eval_data=eval_data_freq,
                            train_labels=train_labels, eval_labels=eval_labels,
                            epochs=32, nfilt1=32, nfilt2=64,
                            nfreq=NFREQ, ntime=NTINT)
    
    y_pred_prob1 = model_freq_time.predict(eval_data_freq)
    y_pred_prob = list(y_pred_prob1[:,1])
    rfi_prob    = list(y_pred_prob1[:,0])
    prob_threshold=0.5
    y_pred_freq_time = np.array(list(np.round(y_pred_prob)))
    print_metric(eval_label1, y_pred_freq_time)

    ind_frb = np.where(y_pred_prob>prob_threshold)[0]

    #print(y_pred_prob)
    #print(np.argmax(y_pred_prob))
    #print(np.shape(ftdata[0]))
    data1 = eval_data_freq[...,0][np.argmax(y_pred_prob)]
    data2 = eval_data_freq[...,0][np.argmax(rfi_prob)]

    plt.imshow(data1,aspect='auto',interpolation='none')
    plt.savefig('frb.png')
    plt.figure(2)
    plt.imshow(data2,aspect='auto',interpolation='none')
    plt.savefig('rfi.png')
    plt.show()



