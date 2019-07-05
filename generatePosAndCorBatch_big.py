import torch
from torch.utils.data import Dataset
import numpy as np



class generateBatches:

    def __init__(self, batch, train2id, positiveBatch, corruptedBatch, numOfEntity, headCounter, tailCounter):
        self.batch = batch
        self.train2id = train2id
        self.positiveBatch = positiveBatch
        self.corruptedBatch = corruptedBatch
        self.numOfEntity = numOfEntity
        self.headCounter = headCounter
        self.tailCounter = tailCounter

        self.generatePosAndCorBatch()

    ## generate corrupted batches from distribution, does not check from dataset.
    def generatePosAndCorBatch(self):                
        self.positiveBatch["h"] = self.train2id["h"][self.batch]
        self.positiveBatch["r"] = self.train2id["r"][self.batch]
        self.positiveBatch["t"] = self.train2id["t"][self.batch]
        
        batch_length = len(self.batch)
        
        rand_array = torch.ge(torch.rand(batch_length), 0.5)
        
        corrupted_heads = np.random.choice(self.headCounter['indicies'], batch_length, p=self.headCounter['probs'], replace=True)
        corrupted_tails = np.random.choice(self.tailCounter['indicies'], batch_length, p=self.tailCounter['probs'], replace=True)

        self.corruptedBatch["h"] = [corrupted_heads[i] if rand_array[i] == 1 else head.item() for i, head in enumerate(self.positiveBatch["h"])]
        self.corruptedBatch["r"] = self.positiveBatch["r"]
        self.corruptedBatch["t"] = [corrupted_tails[i] if rand_array[i] == 0 else tail.item() for i, tail in enumerate(self.positiveBatch["t"])]
        
        for aKey in self.positiveBatch:
            self.positiveBatch[aKey] = torch.LongTensor(self.positiveBatch[aKey])
        for aKey in self.corruptedBatch:
            self.corruptedBatch[aKey] = torch.LongTensor(self.corruptedBatch[aKey])
