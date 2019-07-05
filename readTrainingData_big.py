import re
import torch
import logging
import numpy as np
from collections import Counter
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def normalize(d, target=1.0):
    raw = sum(d.values())
    factor = target/raw
    return {key:value*factor for key,value in d.items()}

# if dataset is big enoughy, use this reading_library
class readData:
    def __init__(self, inAdd, train2id, entity2id, id2entity, relation2id, id2relation, nums):
        self.inAdd = inAdd
        self.train2id = train2id
        self.headCounter = {}
        self.tailCounter = {}
        self.nums = nums
        self.entity2id = entity2id
        self.id2entity = id2entity
        self.relation2id = relation2id
        self.id2relation = id2relation
        self.numOfTriple = 0
        self.numOfEntity = 0
        self.numOfRelation = 0

        self.readTrain2id()
        logging.info ("number of triples: " + str(self.numOfTriple))

        self.readEntity2id()
        logging.info ("number of entities: " + str(self.numOfEntity))

        self.readRelation2id()
        logging.info ("number of relations: " + str(self.numOfRelation))

        self.nums[0] = self.numOfTriple
        self.nums[1] = self.numOfEntity
        self.nums[2] = self.numOfRelation




    def readTrain2id(self):
        logging.info ("-----Reading train2id.txt from " + self.inAdd + "/-----")
        inputData = np.load(self.inAdd + "/train_triples.npy").T
        logging.info("Data loading complete")
        
        self.numOfTriple = len(inputData[0])
        self.train2id["h"] = torch.tensor(inputData[0])
        self.train2id["r"] = torch.tensor(inputData[2])
        self.train2id["t"] = torch.tensor(inputData[1])
        

        
        self.headCounter = Counter(inputData[0])
        self.tailCounter = Counter(inputData[1])
        
        self.headCounter = {k:np.power(v, 0.75) for k,v in self.headCounter.items()}
        self.headCounter = normalize(self.headCounter)
        self.tailCounter = {k:np.power(v, 0.75) for k,v in self.tailCounter.items()}
        self.tailCounter = normalize(self.tailCounter)
     
        
        return

    def readEntity2id(self):
        logging.info ("-----Reading entity2id.txt from " + self.inAdd + "/-----")
        count = 0
        inputData = open(self.inAdd + "/entity2id.txt")
        line = inputData.readline()
        self.numOfEntity = int(re.findall(r"\d+", line)[0])
        line = inputData.readline()
        while line and line not in ["\n", "\r\n", "\r"]:
            reR = re.search(r"(.+)\t(\d+)", line)
            if reR:
                entity = reR.group(1)
                Eid = reR.group(2)
                self.entity2id[entity] = int(Eid)
                self.id2entity[int(Eid)] = entity
                count += 1
                line = inputData.readline()
            else:
                logging.info ("error in entity2id.txt at line " + str(count + 2))
                line = inputData.readline()
        inputData.close()
        if count == self.numOfEntity:
            return
        else:
            logging.info ("error in entity2id.txt")
            return

    def readRelation2id(self):
        logging.info ("-----Reading relation2id.txt from " + self.inAdd + "/-----")
        count = 0
        inputData = open(self.inAdd + "/relation2id.txt")
        line = inputData.readline()
        self.numOfRelation = int(re.findall(r"\d+", line)[0])
        line = inputData.readline()
        while line and line not in ["\n", "\r\n", "\r"]:
            reR = re.search(r"(.+)\t(\d+)", line)
            if reR:
                relation = reR.group(1)
                Rid = int(reR.group(2))
                self.relation2id[relation] = Rid
                self.id2relation[Rid] = relation
                line = inputData.readline()
                count += 1
            else:
                logging.info ("error in relation2id.txt at line " + str(count + 2))
                line = inputData.readline()
        inputData.close()
        if count == self.numOfRelation:
            return
        else:
            logging.info ("error in relation2id.txt")
            return
