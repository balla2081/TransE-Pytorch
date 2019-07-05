import re
import sys
import time
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from TransE import TransE
from readTrainingData import readData as readData
from readTrainingData_big import readData as readBigData
from generatePosAndCorBatch import generateBatches, dataset
from generatePosAndCorBatch_big import generateBatches as generateBigBatches
from collections import defaultdict, Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class trainTransE:

    def __init__(self, in_dir, out_dir, pre_out_dir, preOrNot, bigOrSmall):
        
        self.dataType = bigOrSmall
        
        self.inAdd = in_dir
        self.outAdd = out_dir
        self.preAdd = pre_out_dir
        self.preOrNot = preOrNot
        self.entityDimension = 100
        self.relationDimension = 100
        self.sizeOfBachtes = 1000
        self.numOfEpochs = 100
        self.outputFreq = 5
        self.learningRate = 0.01  # 0.01
        self.weight_decay = 0.001  # 0.005  0.02
        self.margin = 1.0
        self.norm = 2
        self.top = 10

        self.train2id = {}
        self.entity2id = {}
        self.id2entity = {}
        self.relation2id = {}
        self.id2relation = {}
        self.nums = [0, 0, 0]
        self.numOfTriple = 0
        self.numOfEntity = 0
        self.numOfRelation = 0
        # it is for small data_set
        self.headRelation2Tail = defaultdict(lambda: defaultdict(set))
        self.tailRelation2Head = defaultdict(lambda: defaultdict(set))
        
        # it is for large data_set
        self.headCounter = {}
        self.tailCounter = {}
        
        
        self.positiveBatch = {}
        self.corruptedBatch = {}
        self.entityEmbedding = None
        self.relationEmbedding = None
        

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.start()
        self.train()
        self.end()


    def start(self):
        logging.info ("-----Training Started -----")
        logging.info ("input address: " + self.inAdd)
        logging.info ("output address: " +self.outAdd)
        logging.info ("entity dimension: " + str(self.entityDimension))
        logging.info ("relation dimension: " + str(self.relationDimension))
        logging.info ("number of epochs: " + str(self.numOfEpochs))
        logging.info ("output training results every " + str(self.outputFreq) + " epochs")
        logging.info ("learning rate: " + str(self.learningRate))
        logging.info ("weight decay: " + str(self.weight_decay))
        logging.info ("margin: " + str(self.margin))
        logging.info ("norm: " + str(self.norm))
        logging.info ("is a continued learning: " + str(self.preOrNot))
        if self.preOrNot:
            logging.info ("pre-trained result address: " + self.preAdd)
        print ("device: " + str(self.device))

    def end(self):
        logging.info ("-----Training Finished at -----")

    def train(self):
        if self.dataType == True:
            read = readBigData(self.inAdd, self.train2id,self.entity2id, self.id2entity, self.relation2id, self.id2relation, self.nums)
            self.headCounter = {'indicies': list(read.headCounter.keys()), 'probs': list(read.headCounter.values())}
            self.tailCounter = {'indicies': list(read.tailCounter.keys()), 'probs': list(read.tailCounter.values())}
        
        else:
            read = readData(self.inAdd, self.train2id, self.headRelation2Tail, self.tailRelation2Head,
                          self.entity2id, self.id2entity, self.relation2id, self.id2relation, self.nums)   
        self.numOfTriple = self.nums[0]
        self.numOfEntity = self.nums[1]
        self.numOfRelation = self.nums[2]

        logging.info('Model initialize')
        transE = TransE(self.numOfEntity, self.numOfRelation, self.entityDimension, self.relationDimension, self.norm)

        if self.preOrNot:
            self.preRead(transE)

        criterion = nn.MarginRankingLoss(self.margin, False).to(self.device)
        optimizer = optim.SGD(transE.parameters(), lr=self.learningRate, weight_decay=self.weight_decay)

        dataSet = dataset(self.numOfTriple)
        batchSize = self.sizeOfBachtes
        dataLoader = DataLoader(dataSet, batchSize, True)
 
        logging.info('training start')
        for epoch in range(self.numOfEpochs):
            epochLoss = 0
            for batch in dataLoader:
                self.positiveBatch = {}
                self.corruptedBatch = {}
                if self.dataType == True:
                    generateBigBatches(batch, self.train2id, self.positiveBatch, self.corruptedBatch, self.numOfEntity,
                                    self.headCounter, self.tailCounter)
                else:
                    generateBatches(batch, self.train2id, self.positiveBatch, self.corruptedBatch, self.numOfEntity,
                                    self.headRelation2Tail, self.tailRelation2Head)
                    
                optimizer.zero_grad()
                positiveBatchHead = self.positiveBatch["h"].to(self.device)
                positiveBatchRelation = self.positiveBatch["r"].to(self.device)
                positiveBatchTail = self.positiveBatch["t"].to(self.device)
                corruptedBatchHead = self.corruptedBatch["h"].to(self.device)
                corruptedBatchRelation = self.corruptedBatch["r"].to(self.device)
                corruptedBatchTail = self.corruptedBatch["t"].to(self.device)
                output = transE(positiveBatchHead, positiveBatchRelation, positiveBatchTail, corruptedBatchHead,
                                   corruptedBatchRelation, corruptedBatchTail).to(self.device)
                positiveLoss = output.view(2, -1)[0].to(self.device)
                negativeLoss = output.view(2, -1)[1].to(self.device)
                tmpTensor = torch.tensor([-1], dtype=torch.float).to(self.device)
                batchLoss = criterion(positiveLoss, negativeLoss, tmpTensor)
                batchLoss.backward()
                optimizer.step()
                epochLoss += batchLoss

            logging.info ("epoch " + str(epoch) + ": , loss: " + str(epochLoss))

            if (epoch+1)%self.outputFreq == 0 or (epoch+1) == self.numOfEpochs:
                self.write(epoch, transE)
            print ("")


    def write(self, epoch, transE):
        logging.info ("-----Writing Training Results at " + str(epoch) + " to " + self.outAdd + "-----")
        entity2vecAdd = self.outAdd + "/entity2vec" + str(epoch) + ".pickle"
        relation2vecAdd = self.outAdd + "/relation2vec" + str(epoch) + ".pickle"
        entityOutput = open(entity2vecAdd, "wb")
        relationOutput = open(relation2vecAdd, "wb")
        pickle.dump(transE.entityEmbeddings, entityOutput)
        pickle.dump(transE.relationEmbeddings, relationOutput)
        entityOutput.close()
        relationOutput.close()

    def preRead(self, transE):
        logging.info ("-----Reading Pre-Trained Results from " + self.preAdd + "-----")
        entityInput = open(self.preAdd + "/entity2vec.pickle", "rb")
        relationInput = open(self.preAdd + "/relation2vec.pickle", "rb")
        tmpEntityEmbedding = pickle.load(entityInput)
        tmpRelationEmbedding = pickle.load(relationInput)
        entityInput.close()
        relationInput.close()
        
        before_size = tmpEntityEmbedding.shape[0]
        dummy = transE.entity_embeddings.weight[before_size:].data
        transE.entity_embeddings.weight.data = torch.cat((tmpEntityEmbedding, dummy))
        transE.relation_embeddings.weight.data = tmpRelationEmbedding


if __name__ == '__main__':
    IN_DIR = sys.argv[1]
    OUT_DIR = sys.argv[2]
    PRE_OUT_DIR = sys.argv[3] if sys.argv[3] != "None" else None
    PRE_OR_NOT = True if sys.argv[3] != "None" else False
    BIG_OR_SMALL = True if sys.argv[4] == 'big' else False
    print(BIG_OR_SMALL)
    
    trainTransE = trainTransE(IN_DIR, OUT_DIR, PRE_OUT_DIR, PRE_OR_NOT, BIG_OR_SMALL)

