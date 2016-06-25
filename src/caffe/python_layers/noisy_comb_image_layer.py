"""
Computes sum of probabilities for noise model

"""

import caffe
import numpy as np
import yaml
from multiprocessing import Process, Queue
import multiprocessing
import h5py
import math
import code
import os

class NoisyCombImageLayer(caffe.Layer):
    def setup(self, bottom, top):
        """Setup the NoisyCombImageLayer."""
        assert( len(top) == 1), 'Only one top'
        assert( len(bottom) == 5), 'Only five bottom'

        
        pmil = bottom[0].data;
        q00 = bottom[1].data;
        q01 = bottom[2].data;
        q10 = bottom[3].data;
        q11 = bottom[4].data;

        assert(q00.shape == pmil.shape);
        assert(q00.shape == q01.shape);
        assert(q00.shape == q10.shape);
        assert(q00.shape == q11.shape);
        assert(pmil.shape[0] == 1);         

    def forward(self, bottom, top):
        """Get blobs and sum them"""
        pmil = bottom[0].data;
        q00 = bottom[1].data;
        q01 = bottom[2].data;
        q10 = bottom[3].data;
        q11 = bottom[4].data;
        topData = np.multiply(q10, (1-pmil)) + np.multiply(q11, pmil);
        top[0].data[...] = topData.astype(dtype=np.float32, copy=False);

    def backward(self, top, propagate_down, bottom):
        pmil = bottom[0].data;
        q00 = bottom[1].data;
        q01 = bottom[2].data;
        q10 = bottom[3].data;
        q11 = bottom[4].data;
        tdiff = top[0].diff;    

        if propagate_down[0]:
            bottom[0].diff[...] = np.multiply(tdiff, q11 - q10).astype(dtype=np.float32, copy=False);
        if propagate_down[1]:
            bottom[1].diff[...] = np.zeros(tdiff.shape, dtype=np.float32).astype(dtype=np.float32, copy=False);
        if propagate_down[2]:
            bottom[2].diff[...] = np.zeros(tdiff.shape, dtype=np.float32).astype(dtype=np.float32, copy=False);
        if propagate_down[3]:
            bottom[3].diff[...] = np.multiply(tdiff, (1-pmil)).astype(dtype=np.float32, copy=False);
        if propagate_down[4]:
            bottom[4].diff[...] = np.multiply(tdiff, pmil).astype(dtype=np.float32, copy=False);

    def reshape(self, bottom, top):        
        bottom_shape = bottom[0].data.shape;        
        top[0].reshape(*(bottom_shape)); 
