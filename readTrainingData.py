import re
import torch
import logging
import numpy as np
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class smallDataSet(Dataset):
    def __init__(self, inDir):
        self.inDir = inDir
        self.headRelation2Tail = defaultdict(lambda: defaultdict(set))
        self.tailRelation2Head = defaultdict(lambda: defaultdict(set))
        
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
        
        for count, (head, tail, rel) in enumerate(inputData):
            self.headRelation2Tail[head][rel].add(tail)
            self.tailRelation2Head[tail][rel].add(head)
        logging.info("Making count dict complete")
        
        self.numOfTriple = len(inputData)
        self.trainTriples = torch.LongTensor(inputData)

    

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
        corruptedBatch = []
        for head, tail, rel in batch:
            head = head.item()
            rel = rel.item()
            tail = tail.item()
            
            if torch.rand(1).item() >= 0.5:
                not_list = self.tailRelation2Head[tail][rel].union(set([head]))
                while True:
                    CorruptedHead = torch.FloatTensor(1).uniform_(0, self.numOfEntity).long().item()
                    if CorruptedHead not in not_list:
                        head = CorruptedHead
                        break
            else:
                not_list = self.headRelation2Tail[head][rel].union(set([tail]))
                while True:
                    CorruptedTail = torch.FloatTensor(1).uniform_(0, self.numOfEntity).long().item()
                    if CorruptedTail not in not_list:
                        tail = CorruptedTail
                        break
                        
            corruptedBatch.append([head, tail, rel])
            

        corruptedBatch = torch.LongTensor(corruptedBatch)
                
        return corruptedBatch
        
            
        
        
    