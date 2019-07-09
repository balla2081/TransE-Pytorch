import re
import torch
import logging
import numpy as np
from collections import Counter
import numpy as np
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def normalize(d, target=1.0):
    raw = sum(d.values())
    factor = target/raw
    return {key:value*factor for key,value in d.items()}


# if dataset is big enoughy, use this reading_library
class bigDataSet(Dataset):
    def __init__(self, inDir):
        self.inDir = inDir
        self.headIndicies = []
        self.headDistribution = []
        self.tailIndicies = []
        self.tailDistribution = []

        self.readTrainTriples()
        self.readEntityNumber()
        self.readRelationNumber()
    
    def __len__(self):
        return self.numOfTriple

    def __getitem__(self, item):
        return self.trainTriples[item]


    def readTrainTriples(self):
        logging.info ("-----Reading train tiples from " + self.inDir + "/-----")
        inputData = np.load(self.inDir + "/train_triples.npy")
        logging.info(inputData.shape)
        logging.info("Data loading complete")
        
        
        self.numOfTriple = len(inputData)
        self.trainTriples = torch.LongTensor(inputData)
        
        inputData = inputData.T
        headCounter = Counter(inputData[0])
        tailCounter = Counter(inputData[1])
        
        logging.info("making head Dist")
        headDistributionDict = {k:np.power(v, 0.75) for k,v in headCounter.items()}
        headDistributionDict = normalize(headDistributionDict)
        self.headIndicies = torch.LongTensor(list(headDistributionDict.keys()))
        self.headDistribution = torch.Tensor(list(headDistributionDict.values()))
        
        logging.info("making tail Dist")
        tailDistributionDict = {k:np.power(v, 0.75) for k,v in tailCounter.items()}
        tailDistributionDict = normalize(tailDistributionDict)
        self.tailIndicies = torch.LongTensor(list(tailDistributionDict.keys()))
        self.tailDistribution = torch.Tensor(list(tailDistributionDict.values()))
        
        return

    def readEntityNumber(self):
        logging.info ("-----Reading entity2id.txt from " + self.inDir + "/-----")
        inputData = open(self.inDir + "/entity2id.txt")
        line = inputData.readline()
        self.numOfEntity = int(re.findall(r"\d+", line)[0])
        
        return

    def readRelationNumber(self):
        logging.info ("-----Reading relation2id.txt from " + self.inDir + "/-----")
        inputData = open(self.inDir + "/relation2id.txt")
        line = inputData.readline()
        self.numOfRelation = int(re.findall(r"\d+", line)[0])
        inputData.close()
        
        return 
    
    def generateCorruptedBatch(self, batch):
        corruptedBatch = {}
        logging.info('making rand array')
        batch_length = len(batch)
        
        logging.info('making corrupted haeds and tails')
        corrputed_heads_index = torch.multinomial(self.headDistribution, batch_length, replacement=True)
        corrupted_heads = self.headIndicies[corrputed_heads_index].view((-1,1))
        corrputed_tails_index = torch.multinomial(self.tailDistribution, batch_length, replacement=True)
        corrupted_tails = self.tailIndicies[corrputed_tails_index].view((-1,1))
        logging.info('random complete')
        corruptedBatch = torch.cat((torch.cat((corrupted_heads,batch[:,1:]), 1), torch.cat((batch[:,:1],corrupted_tails,batch[:,2:]), 1)))
        print(corruptedBatch.size())
        return torch.LongTensor(corruptedBatch)
