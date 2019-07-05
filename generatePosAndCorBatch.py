import torch
from torch.utils.data import Dataset


class dataset(Dataset):

    def __init__(self, numOfTriple):
        self.tripleList = torch.LongTensor(range(numOfTriple))
        self.numOfTriple = numOfTriple

    def __len__(self):
        return self.numOfTriple

    def __getitem__(self, item):
        return self.tripleList[item]


class generateBatches:

    def __init__(self, batch, train2id, positiveBatch, corruptedBatch, numOfEntity, headRelation2Tail, tailRelation2Head):
        self.batch = batch
        self.train2id = train2id
        self.positiveBatch = positiveBatch
        self.corruptedBatch = corruptedBatch
        self.numOfEntity = numOfEntity
        self.headRelation2Tail = headRelation2Tail
        self.tailRelation2Head = tailRelation2Head

        self.generatePosAndCorBatch()

    def generatePosAndCorBatch(self):                
        self.positiveBatch["h"] = self.train2id["h"][self.batch]
        self.positiveBatch["r"] = self.train2id["r"][self.batch]
        self.positiveBatch["t"] = self.train2id["t"][self.batch]
        
        self.corruptedBatch["h"] = []
        self.corruptedBatch["r"] = []
        self.corruptedBatch["t"] = []
        
        for head, rel, tail in zip(self.positiveBatch["h"], self.positiveBatch["r"], self.positiveBatch["t"]):
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
            self.corruptedBatch["h"].append(head)
            self.corruptedBatch["r"].append(rel)
            self.corruptedBatch["t"].append(tail)        
                    
        
        for aKey in self.positiveBatch:
            self.positiveBatch[aKey] = torch.LongTensor(self.positiveBatch[aKey])
        for aKey in self.corruptedBatch:
            self.corruptedBatch[aKey] = torch.LongTensor(self.corruptedBatch[aKey])
