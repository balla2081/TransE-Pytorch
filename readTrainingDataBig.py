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
        return (self.trainTriples["h"][item], self.trainTriples["r"][item], self.trainTriples["t"][item])


    def readTrainTriples(self):
        logging.info ("-----Reading train tiples from " + self.inDir + "/-----")
        inputData = np.load(self.inDir + "/train_triples.npy").T
        logging.info(inputData.shape)
        logging.info("Data loading complete")
        
        
        self.numOfTriple = len(inputData[0])
        self.trainTriples = {}
        self.trainTriples["h"] = torch.tensor(inputData[0])
        self.trainTriples["r"] = torch.tensor(inputData[2])
        self.trainTriples["t"] = torch.tensor(inputData[1])
        

        
        headCounter = Counter(inputData[0])
        tailCounter = Counter(inputData[1])
        
        headDistributionDict = {k:np.power(v, 0.75) for k,v in headCounter.items()}
        headDistributionDict = normalize(headDistributionDict)
        self.headIndicies = list(headDistributionDict.keys())
        self.headDistribution = list(headDistributionDict.values())
        
        tailDistributionDict = {k:np.power(v, 0.75) for k,v in tailCounter.items()}
        tailDistributionDict = normalize(tailDistributionDict)
        self.tailIndicies = list(tailDistributionDict.keys())
        self.tailDistribution = list(tailDistributionDict.values())
        
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
        
        batch_length = len(batch[0])
        rand_array = torch.ge(torch.rand(batch_length), 0.5)
        
        corrupted_heads = np.random.choice(self.headIndicies, batch_length, p=self.headDistribution, replace=True)
        corrupted_tails = np.random.choice(self.tailIndicies, batch_length, p=self.tailDistribution, replace=True)
        
        corruptedBatch["h"] = [corrupted_heads[i] if rand_array[i] == 1 else head.item() for i, head in enumerate(batch[0])]
        corruptedBatch["r"] = batch[1]
        corruptedBatch["t"] = [corrupted_tails[i] if rand_array[i] == 0 else tail.item() for i, tail in enumerate(batch[2])]
            
        for aKey in corruptedBatch:
            corruptedBatch[aKey] = torch.LongTensor(corruptedBatch[aKey])
                
        return [corruptedBatch['h'], corruptedBatch['r'], corruptedBatch['t']]
        
