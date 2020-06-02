import time
import os
import math
import sys
import shutil
import numpy
import h5py
import logging
import argparse

import vtx


parser = argparse.ArgumentParser(description='Training of networks')
parser.add_argument('-i','--input',nargs="+", dest='inputFiles', default=[], action='append',
                    help='Input files',required=True)
parser.add_argument('-v', dest='logLevel', default='Info', type=str,
                    help='Verbosity level: Debug, Info, Warning, Error, Critical')

args = parser.parse_args()

logging.basicConfig(format='%(levelname)s: %(message)s', level=getattr(logging, args.logLevel.upper(), None))

inputFiles = [f for fileList in args.inputFiles for f in fileList]

if len(inputFiles)==0:
    logging.critical("No input files specified")
    sys.exit(1)
    
pipeline = vtx.Pipeline(inputFiles,testFraction=0.0)

fhAlgo = vtx.FastHisto()

predictedZ0FH = []
trueZ0 = []

for batch in pipeline.generate(
    batchSize=100,
    nFiles=1,
    isTraining=True
):
    predictedZ0FH.append(fhAlgo.predictZ0(
        batch['X'][:,:,0],
        batch['X'][:,:,1]
    ))
    trueZ0.append(batch['y_avg'])
    
predictedZ0FH = numpy.concatenate(predictedZ0FH,axis=0)
trueZ0 = numpy.concatenate(trueZ0,axis=0)
        
print ("Quantiles: ",list(map(lambda x: "%6.1f%%"%x,[5.,15.87,50.,84.13,95.])))
print ("FastHisto: ",list(map(lambda x: "%+6.4f"%x,numpy.percentile(predictedZ0FH-trueZ0,[5.,15.87,50.,84.13,95.]))))

        
