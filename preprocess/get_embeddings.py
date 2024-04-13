import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import re
import numpy as np
import pandas as pd
import os
import requests
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import argparse
from protembedding import get_protein_embeddings
from smilesembedding import get_smiles_embeddings


def get_data_list(data_path):
    full_data_path = os.path.join(data_path,"fulldata.csv")
    full=pd.read_csv(full_data_path)
    smiledictnum2str={};smiledictstr2num={}
    sqddictnum2str={};sqdictstr2num={}
    list_of_exmaple=[]
    pos,neg=0,0
    smilelist=[]
    proteinlist=[]
    for no,data in enumerate(full.values):
        smiles,protein,interaction,_,_ =data
        if interaction==1:
            pos+=1
        else:
            neg+=1
        if smiledictstr2num.get(smiles) == None:
            smiledictstr2num[smiles] = len(smiledictstr2num)
            smiledictnum2str[str(len(smiledictnum2str))] = smiles
            smilelist.append(smiles)

        if sqdictstr2num.get(protein) == None:
            sqdictstr2num[protein] = len(sqdictstr2num)
            sqddictnum2str[str(len(sqddictnum2str))] = protein
            proteinlist.append(protein)
    column=['smiles','protein','interactions']

    print(f'number of smiles: {len(smilelist)}, number of proteins: {len(proteinlist)}')
    trainsmiles=pd.DataFrame(columns=['smiles'],data=smilelist)
    trainsequence=pd.DataFrame(columns=['protein'],data=proteinlist)
    trainsmiles.to_csv(data_path+'/smiles.csv')
    trainsequence.to_csv(data_path+'/protein.csv')
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='settings')
    parser.add_argument('--dataset', type=str, default="bindingdb", choices=["human","celegans","bindingdb","biosnap"],help='select dataset for training')
    parser.add_argument('--device', type=str, default="cuda:0",help='cuda or cpu') 
 
    args = parser.parse_args()

    
    data_path = os.path.join('./data',args.dataset)
    get_data_list(data_path)
    get_protein_embeddings(data_path,args.device)
    get_smiles_embeddings(data_path,args.device)


    


