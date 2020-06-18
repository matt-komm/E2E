import tensorflow as tf
import h5py
import numpy
import os
import math
import sys
import vtx

inputFiles = vtx.InputFiles(['../Samples/hybrid_tt100k.hdf5'])
pipeline = vtx.Pipeline(inputFiles,testFraction=0.1)

sess = tf.Session()

weights_graph_def = tf.GraphDef()
weights_graph_def.ParseFromString(open("hybridv2_weight.pb","rb").read())
track_input,weights_output = (tf.graph_util.import_graph_def(weights_graph_def, name="weights", return_elements=['track_input','weights_output']))

position_graph_def = tf.GraphDef()
position_graph_def.ParseFromString(open("hybridv2_position.pb","rb").read())
hists_input,pv_position_output = (tf.graph_util.import_graph_def(position_graph_def, name="position", return_elements=['hists_input','pv_position_output']))


for batch in pipeline.generate(
    1,
    nFiles=max(3,len(inputFiles)),
    isTraining=True
):
    
    weights = sess.run(
        weights_output.outputs[0],
        feed_dict = {track_input.outputs[0]: batch['X'][0]}
    )
    hist,bin_edges = numpy.histogram(batch['X'][0,:,0],256,range=(-15,15),weights=weights[:,0])
    
    #print (batch['X'].shape)
    hist = numpy.expand_dims(numpy.expand_dims(hist,axis=2),axis=0)
    
    print (sess.run(
        pv_position_output.outputs[0],
        feed_dict = {hists_input.outputs[0]: hist})
    )
    print (batch['y_avg'][0])

    print ()





