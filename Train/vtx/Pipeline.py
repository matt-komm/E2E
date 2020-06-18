import numpy
import h5py
import random

class Accessor:
    @staticmethod
    def createEmptyBatch():
        return {"X":[],"assoc":[],"y":[],"y_avg":[]}
    
    @staticmethod
    def fileSize(f):
        return f['X'].shape[0]
        
    @staticmethod
    def batchSize(batch):
        return sum(map(lambda x: x.shape[0],batch['X']))
       
    @staticmethod 
    def readFile(batch, f, indices):
        batch['X'].append(f['X'][indices])
        batch['assoc'].append(f['assoc'][indices])
        batch['y'].append(f['y'][indices])
        batch['y_avg'].append(f['y_avg'][indices])
        
    @staticmethod
    def concatenateBatch(batch):
        batch['X'] = numpy.concatenate(batch['X'],axis=0)
        batch['assoc'] = numpy.concatenate(batch['assoc'],axis=0)
        batch['y'] = numpy.concatenate(batch['y'],axis=0)
        batch['y_avg'] = numpy.concatenate(batch['y_avg'],axis=0)
        
class Pipeline():
    def __init__(self,inputFileList,testFraction,dataAccessor=Accessor):
        self.inputFileList = inputFileList   
        self.testFraction = testFraction
        self.dataAccessor = dataAccessor
    
    def generate(self,batchSize=100,nFiles=2,isTraining=True,shuffle=True):
        fileIndices = list(range(len(self.inputFileList)))
        readIndex = 0
        if shuffle:
            random.shuffle(fileIndices)
        openFiles = []
        
        def totalsize(fileList):
            n = 0
            for f in fileList:
                n+=len(f['indices'])-f['index']
            return n
          
        while readIndex==0 or readIndex<len(self.inputFileList) or totalsize(openFiles)>batchSize:
            while (len(openFiles)<min(len(self.inputFileList),nFiles) or totalsize(openFiles)<batchSize) and readIndex<(len(self.inputFileList)):
                #print 'loading ',inputFileList[fileIndices[readIndex]]
                filePath = self.inputFileList[fileIndices[readIndex]]
                f = h5py.File(filePath,'r')
                if isTraining:
                    fileDescription = {
                        'path':filePath,
                        'handle':f,
                        'start':0,
                        'index':0,
                        'end':int(self.dataAccessor.fileSize(f)*min(1.,max(0.,1.-self.testFraction)))
                    }
                    
                else:
                    fileDescription = {
                        'path':filePath,
                        'handle':f,
                        'start':int(self.dataAccessor.fileSize(f)*min(1.,max(0.,1.-self.testFraction))),
                        'index':0,
                        'end':self.dataAccessor.fileSize(f)
                    }
                fileDescription['indices'] = numpy.arange(fileDescription['start'],fileDescription['end'])
                
                #numpy.random.shuffle(fileDescription['indices'])
                    
                    
                openFiles.append(fileDescription)
                
                readIndex+=1
        
            if totalsize(openFiles)>batchSize:  
                batch = self.dataAccessor.createEmptyBatch()
                while (self.dataAccessor.batchSize(batch)<batchSize):
                    currentBatchSize = self.dataAccessor.batchSize(batch)
                    #print '  batch size ',currentBatchSize
                    chosenFileIndex = random.randint(0,len(openFiles)-1)
                    f = openFiles[chosenFileIndex]
                    nread = min([len(f['indices'])-f['index'],max(1,int(1.*batchSize/nFiles)),batchSize-currentBatchSize])
                    
                    #print ('  reading ',f['path'],f['index'],f['indices'][f['index']:f['index']+nread])
                    
                    self.dataAccessor.readFile(batch,f['handle'],numpy.sort(f['indices'][f['index']:f['index']+nread]))
                    f['index']+=nread
                    if (f['index']+1)>=len(f['indices']):
                        elem = openFiles.pop(chosenFileIndex)
                        #print 'dequeue ',elem,len(openFiles)
                    
                self.dataAccessor.concatenateBatch(batch)
                yield batch
            
            
            
