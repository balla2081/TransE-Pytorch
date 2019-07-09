import re
import sys
import time
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from TransE import TransE
from readTrainingData import smallDataSet
from readTrainingDataBig import bigDataSet
from collections import defaultdict, Counter
import logging
import torch.nn.functional as F


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class trainTransE:
    def __init__(self, in_dir, out_dir, pre_out_dir, preOrNot, isBig):
        
        self.isBig = isBig
        
        self.inDir = in_dir
        self.outDir = out_dir
        self.preDit = pre_out_dir
        self.preOrNot = preOrNot
        self.entityDimension = 100
        self.relationDimension = 100
        self.sizeOfBatches = 10000
        self.numOfEpochs = 100
        self.outputFreq = 5
        self.learningRate = 0.01  # 0.01
        self.weight_decay = 0.001  # 0.005  0.02
        self.margin = 1.0
        self.norm = 2
        self.top = 10

        self.train2id = {}
        self.nums = [0, 0, 0]
        
        self.numOfTriple = 0
        self.numOfEntity = 0
        self.numOfRelation = 0
        

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.start()
        self.train()
        self.end()


    def start(self):
        logging.info ("-----Training Started -----")
        logging.info ("input address: " + self.inDir)
        logging.info ("output address: " +self.outDir)
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
        logging.info ("device: " + str(self.device))

    def end(self):
        logging.info ("-----Training Finished at -----")

    def train(self):
        if self.isBig == True:
            self.dataSet = bigDataSet(self.inDir)
        else:
            self.dataSet = smallDataSet(self.inDir)
            
        logging.info('Model initialize')
        transE = TransE(self.dataSet.numOfEntity, self.dataSet.numOfRelation, self.entityDimension, self.relationDimension, self.norm, self.device)
        logging.info("embeding matrix initialize complete")
        if self.preOrNot:
            self.preRead(transE)
            
        criterion = nn.MarginRankingLoss(self.margin, False).to(self.device)
        optimizer = optim.SGD(transE.parameters(), lr=self.learningRate, weight_decay=self.weight_decay)

        dataLoader = DataLoader(dataset=self.dataSet, batch_size=self.sizeOfBatches, shuffle=True, num_workers=4, pin_memory=True)
 
        logging.info('training start')
        for epoch in range(self.numOfEpochs):
            epochLoss = 0
            count = 0
            for batch in dataLoader:
                logging.info("making corrupted_batch")
                corruptedBatch = self.dataSet.generateCorruptedBatch(batch)
                optimizer.zero_grad()
                logging.info("forawrd calculation")
                positiveLoss, negativeLoss = transE(batch, corruptedBatch)
                logging.info("seperate results")
                logging.info("tmp tensor")
                positiveLoss = positiveLoss.to(self.device)
                negativeLoss = negativeLoss.to(self.device)
                tmpTensor = torch.tensor([-1], dtype=torch.float).to(self.device)
                logging.info("calculate loss")
                batchLoss = criterion(positiveLoss, negativeLoss, tmpTensor).to(self.device)
                logging.info("backward")
                batchLoss.backward()
                logging.info("optimizer step")
                optimizer.step()
                epochLoss += batchLoss
                count += 1
                logging.info("step_ended")
                if count%100 == 0:
                    logging.info ("epoch " + str(epoch) + + "-" + str(count) + ": , loss: " + str(epochLoss))

            logging.info ("epoch " + str(epoch) + ": , loss: " + str(epochLoss))

            if (epoch+1)%self.outputFreq == 0 or (epoch+1) == self.numOfEpochs:
                self.write(epoch, transE)
            print ("")


    def write(self, epoch, transE):
        logging.info ("-----Writing Training Results at " + str(epoch) + " to " + self.outDir + "-----")
        entity2vecAdd = self.outDir + "/entity2vec" + str(epoch) + ".pickle"
        relation2vecAdd = self.outDir + "/relation2vec" + str(epoch) + ".pickle"
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
    IS_BIG = True if sys.argv[4] == 'big' else False
    
    trainTransE = trainTransE(IN_DIR, OUT_DIR, PRE_OUT_DIR, PRE_OR_NOT, IS_BIG)

