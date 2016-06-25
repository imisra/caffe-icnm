"""
top_shape[0] = 1;
sums values from bottom across batch
DOES NOT BACK PROPAGATE
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
import traceback as tb
import code

class MeanAcrossBottomsLayer(caffe.Layer):
    def setup(self, bottom, top):
        """Setup the MeanAcrossBottomsLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)
        
        self._name_to_top_map = {
            'meanTop': 0,}        

        assert( len(top) == 1), 'Only one top'
        assert( len(bottom) >= 2), 'At least two bottoms'
        numBottom = len(bottom);
        bot0shape = bottom[0].data.shape;
        for b in range(numBottom):
            assert(bot0shape == bottom[b].data.shape), 'All bottoms must have same shape: {} {}'.format(bot0shape, bottom[b].data.shape);

    def forward(self, bottom, top):
        """Get blobs and sum then along axis=0 (i.e. batch). Then copy into this layer's top blob vector."""
        numBottom = len(bottom);
        assert(numBottom == 5);
        blah = np.vstack((bottom[0].data, bottom[1].data, bottom[2].data, bottom[3].data, bottom[4].data));
        meanblah = blah.mean(axis=0);
        top[0].data[...] = np.reshape(meanblah,top[0].data.shape).astype(dtype=np.float32);

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        raise ValueError('sum_across_batch layer cannot backpropagate')        

    def reshape(self, bottom, top):        
        bottom_shape = bottom[0].data.shape;
        top_shape = list(bottom_shape);
        top_shape = tuple(top_shape)
        top[0].reshape(*(top_shape));        
