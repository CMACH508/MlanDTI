import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import re
import numpy as np
import pandas as pd
import os
import requests
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

proteinmaxlens = 1024

step = proteinmaxlens//2


def sequencechange(sequence):
    '''
    input a protien sequence and return a sequence with blank intervals
    :param sequence:eg: "MSPLNQ"
    :return: eg: "M S P L N Q"
    '''
    new_seq = ""
    count = 0
    for i in sequence:
        if i == ' ':
            continue
        new_seq += i
        count += 1
        if count == len(sequence):
            continue
        new_seq += ' '
    return new_seq
class mydata(Dataset):
    def __init__(self,protdata):
        self.protiens=protdata
        self.len=len(protdata)
    def __getitem__(self, idx):
        return self.protiens[idx]
    def __len__(self):
        return self.len
def preparedata(data_path):
    f=pd.read_csv(data_path+"/protein.csv")
    N = f.protein.values.shape[0]
    proteins= []
    for i in range(N):
        print('/'.join(map(str, [i + 1, N])))
        proteins.append(f.protein.values[i])
    print(str(len(proteins[0]))+' unique protein sequences in total!')
    proteins = [re.sub(r"[UZOB]", "X", sequence) for sequence in proteins]
    return proteins,N
def preparedataset(DATASET):
    proteins,N=preparedata(DATASET)
    proteinloader=DataLoader(mydata(proteins),shuffle=False,batch_size=1,drop_last=False)
    return proteinloader,N
def process(prt):
    #protein maxlength=1024
    prt=prt[0]
    list=[]
    N=len(prt)
    if N>proteinmaxlens:
        i=0
        while (i+1)*step<N:
            slice = prt[i*step:min(i*step+proteinmaxlens,N)]
            slice = sequencechange(slice)
            list.append(slice)
            i+=1
    else:
        prt = sequencechange(prt)
        list.append(prt)
    return list

def get_protein_embeddings(dir_path,device):
    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = AutoModel.from_pretrained("Rostlab/prot_bert")
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fe = pipeline('feature-extraction', model=model, tokenizer=tokenizer, device=device)
    proteinloader, N = preparedataset(dir_path)
    features = []
    i = 0
    for prt in tqdm(proteinloader):

        print('/'.join(map(str, [i + 1, N])))
        getprt = process(prt)
        if len(getprt)>1:
            j = 1
            for protein in getprt:
                embedding = fe([protein])
                embedding = np.array(embedding)
                embedding = embedding.reshape(-1, 1024)
                embed = embedding[1:embedding.shape[0] - 1]
                if j==1:
                    fullembed = embed
                else:
                    fullembed[-step:]=(embed[0:step]+fullembed[-step:])/2
                    fullembed = np.append(fullembed,embed[step:],axis=0)
                if j==len(getprt):
                    features.append(fullembed)
                j += 1
            i += 1
        else:
            embedding = fe(getprt)
            embedding = np.array(embedding)
            embedding = embedding.reshape(-1, 1024)
            embed = embedding[1:embedding.shape[0] - 1]
            features.append(embed)
            i += 1
    features =np.array(features,dtype=object)
    np.save(dir_path + '/proteinsembeddings', features,allow_pickle=True)
    print('The preprocess of dataset has finished!')
if __name__ == "__main__":
    DATASET = "human"
    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = AutoModel.from_pretrained("Rostlab/prot_bert")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    fe = pipeline('feature-extraction', model=model, tokenizer=tokenizer, device=device)
    proteinloader, N = preparedataset(DATASET)
    features = []
    i = 0
    for prt in tqdm(proteinloader):
        print(type(prt),len(prt))
        print('/'.join(map(str, [i + 1, N])))
        getprt = process(prt)
        if len(getprt)>1:
            j = 1
            for protein in getprt:
                embedding = fe([protein])
                embedding = np.array(embedding)
                embedding = embedding.reshape(-1, 1024)
                embed = embedding[1:embedding.shape[0] - 1]
                if j==1:
                    fullembed = embed
                else:
                    fullembed[-step:]=(embed[0:step]+fullembed[-step:])/2
                    fullembed = np.append(fullembed,embed[step:],axis=0)
                if j==len(getprt):
                    features.append(fullembed)
                j += 1
            i += 1
        else:
            embedding = fe(getprt)
            embedding = np.array(embedding)
            embedding = embedding.reshape(-1, 1024)
            embed = embedding[1:embedding.shape[0] - 1]
            features.append(embed)
            i += 1
        print(features[i-1].shape)

    '''sequences_Example = ["A E T C Z A O"]
    sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]
    embedding = fe(sequences_Example)
    embedding = np.array(embedding)
    embedding = embedding.reshape(-1, 1024)
    print(embedding.shape)'''

    dir_input = ('dataset/' + DATASET + '/')
    os.makedirs(dir_input, exist_ok=True)
    np.save(dir_input + 'proteinsembeddings', features,allow_pickle=True)
    print('The preprocess of ' + DATASET + ' dataset has finished!')

