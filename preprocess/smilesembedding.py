from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from torch import nn

class mydata(Dataset):
   def __init__(self, compounddata):
      self.compounds = compounddata
      self.len = len(compounddata)

   def __getitem__(self, idx):
      return self.compounds[idx]

   def __len__(self):
      return self.len


def preparedata(dir_path):
   f = pd.read_csv(dir_path+"/smiles.csv")
   N = f.smiles.values.shape[0]
   compounds = []
   for i in range(N):
      print('/'.join(map(str, [i + 1, N])))
      mol = Chem.MolFromSmiles(f.smiles.values[i])
      smiles = Chem.MolToSmiles(mol)
      compounds.append(smiles)

   return compounds, N

def adjacent_matrix(mol):
   adjacency = Chem.GetAdjacencyMatrix(mol)
   return np.array(adjacency) + np.eye(adjacency.shape[0])

def preparedataset(DATASET):
   compounds, N = preparedata(DATASET)
   compoundsloader = DataLoader(mydata(compounds), shuffle=False, batch_size=1, drop_last=False)
   return compoundsloader, N

def get_smiles_embeddings(dir_path, device):
   tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
   model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MLM")

   # device = 'cuda:' if torch.cuda.is_available() else 'cpu'
   model.to(device)
   model = model.eval()
   compoundsloader, N = preparedataset(dir_path)
   compounds = []
   i = 0
   ln = nn.LayerNorm(768).to(device)
   for data in tqdm(compoundsloader):
      print(str(i+1) + '/' + str(N))
      tokens = tokenizer.tokenize(data[0])
      string = ''.join(tokens)
      if len(string) > 512:
         j = 0
         flag = True
         output = torch.zeros(1, 384).to(device)
         while flag:
            input = tokenizer(string[j:min(len(string), j + 511)], return_tensors='pt').to(device)
            if len(string) <= j + 511:
               flag = False
            with torch.no_grad():
               hidden_states = model(**input, return_dict=True, output_hidden_states=True).hidden_states
               output_hidden_state = torch.cat([(hidden_states[-1] + hidden_states[1]).mean(dim=1),(hidden_states[-2] + hidden_states[2]).mean(dim=1)],dim=1)  # first last layers average add
               output_hidden_state = ln(output_hidden_state)
            output = torch.cat((output, output_hidden_state), dim=0)
            j += 256
            print(output.shape)
         output = output[1:-1].mean(dim=0).unsqueeze(dim=0).to('cpu').data.numpy()
      else:
         input = tokenizer(data[0], return_tensors='pt').to(device)
         with torch.no_grad():
            hidden_states = model(**input, return_dict=True, output_hidden_states=True).hidden_states
            output_hidden_state = torch.cat([(hidden_states[-1] + hidden_states[1]).mean(dim=1),(hidden_states[-2] + hidden_states[2]).mean(dim=1)],dim=1)  # first last layers average add   
            output_hidden_state = ln(output_hidden_state)
         output = output_hidden_state.to('cpu').data.numpy()
      compounds.append(output)
      i+=1
   compounds =np.array(compounds,dtype=object)
   np.save(dir_path+'/smilesembeddings', compounds, allow_pickle=True)
   print('The preprocess of dataset has finished!')



if __name__ == "__main__":
   DATASET = "Data/human/"
   path = "dataset/" + DATASET
   tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
   model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MLM")

   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   model.to(device)
   model = model.eval()
   compoundsloader, N = preparedataset(path)
   compounds = []
   i = 0
   ln = nn.LayerNorm(768).to(device)
   for data in tqdm(compoundsloader):
      print(str(i+1) + '/' + str(N))
      tokens = tokenizer.tokenize(data[0])
      string = ''.join(tokens)
      if len(string) > 512:
         j = 0
         flag = True
         output = torch.zeros(1, 384).to(device)
         while flag:
            input = tokenizer(string[j:min(len(string), j + 511)], return_tensors='pt').to(device)
            if len(string) <= j + 511:
               flag = False
            with torch.no_grad():
               hidden_states = model(**input, return_dict=True, output_hidden_states=True).hidden_states
               output_hidden_state = torch.cat([(hidden_states[-1] + hidden_states[1]).mean(dim=1),(hidden_states[-2] + hidden_states[2]).mean(dim=1)],dim=1)  # first last layers average add
               output_hidden_state = ln(output_hidden_state)
            output = torch.cat((output, output_hidden_state), dim=0)
            j += 256
            print(output.shape)
         output = output[1:-1].mean(dim=0).unsqueeze(dim=0).to('cpu').data.numpy()
      else:
         input = tokenizer(data[0], return_tensors='pt').to(device)
         with torch.no_grad():
            hidden_states = model(**input, return_dict=True, output_hidden_states=True).hidden_states
            output_hidden_state = torch.cat([(hidden_states[-1] + hidden_states[1]).mean(dim=1),(hidden_states[-2] + hidden_states[2]).mean(dim=1)],dim=1)  # first last layers average add
            # output_hidden_state = torch.cat([hidden_states[-1].mean(dim=1),hidden_states[1].mean(dim=1)],dim=0)
            output_hidden_state = ln(output_hidden_state)
            # print(output_hidden_state,output_hidden_state.shape)
         output = output_hidden_state.to('cpu').data.numpy()
      print(output.shape)
      compounds.append(output)
      i+=1
   print(len(compounds))
   np.save(path+'smilesembeddings', compounds, allow_pickle=True)
   print('The preprocess of dataset has finished!')
