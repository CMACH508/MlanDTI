
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader

class Mydataset(Dataset):
    def __init__(self,samples,compounds,proteins):


        self.samples=samples
        self.len=len(samples)
        self.compounds = compounds
        self.proteins = proteins
    def __getitem__(self, idx):
        #return self.compound[idx],self.adj[idx],self.protein[idx],self.correct_interaction[idx]
        one_sample = self.samples[idx]
        return self.compounds[int(one_sample[1])],self.proteins[int(one_sample[2])],torch.tensor([self.samples[idx,3]]),one_sample
    def __len__(self):
        return self.len
class MultiDataLoader(object):
    def __init__(self, dataloaders, n_batches):
        if n_batches <= 0:
            raise ValueError("n_batches should be > 0")
        self._dataloaders = dataloaders
        self._n_batches = np.maximum(1, n_batches)
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):
        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                new_dl = iter(self._dataloaders[di])
                self._iterators[di] = new_dl
                batch = next(new_dl)
            return batch

        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        for _ in range(self._n_batches):
            yield self._get_nexts()
        self._init_iterators()

    def __len__(self):
        return self._n_batches
    
def load_tensor(file_name, dtype):
    return [dtype(d) for d in np.load(file_name + '.npy',allow_pickle=True)]



def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2



def preparedata(type,dataset):
    dir_input = ('dataset/'+dataset+'/')
    compounds = load_tensor(dir_input + 'smilesembeddings', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteinsembeddings', torch.FloatTensor) 
    trainfiles = pd.read_csv(dir_input + type + '/train/' + 'samples.csv')
    validfiles = pd.read_csv(dir_input + type + '/valid/' + 'samples.csv')
    testfiles = pd.read_csv(dir_input + type + '/test/' + 'samples.csv')
    

    return trainfiles.values,validfiles.values,testfiles.values,compounds,proteins


def collatef(batch):
    batchlist=[]
    for item in batch:
        compound,protein,interaction,sample=item
        list=[compound,protein,interaction,sample]
        batchlist.append(list)
    return batchlist

def preparedataset(batch_size,type,dataset):
    trainsamples,validsamples,testsamples,compounds,proteins=preparedata(type,dataset)
    trainloader = DataLoader(Mydataset(trainsamples,compounds,proteins),shuffle=True,batch_size=batch_size,collate_fn=collatef, drop_last=False)
    validloader = DataLoader(Mydataset(validsamples, compounds, proteins), shuffle=False, batch_size=batch_size,
                            collate_fn=collatef, drop_last=False)
    testloader = DataLoader(Mydataset(testsamples, compounds, proteins), shuffle=False, batch_size=batch_size,
                             collate_fn=collatef, drop_last=False)
    return trainloader,validloader,testloader,compounds,proteins