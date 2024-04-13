
import copy
import time
from random import sample as SAMPLE
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score,precision_recall_curve, auc, average_precision_score,accuracy_score
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from dataset.getdataset import Mydataset,collatef
from utils.attention import FeedForward,SelfAttention
#from Radam import *




def get_nlayers(module,N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim,kernel_size , dropout, device):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.conv =nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size, padding=(kernel_size - 1) // 2)# for _ in range(self.n_layers)])  # convolutional layers
        self.dropout = nn.Dropout(dropout)

    def forward(self, conv_input):

        #protein = [batch size, protein len,protein_dim]
        #permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)
        #conv_input = [batch size, hid dim, protein len]
            #pass through convolutional layer
        conved = self.conv(self.dropout(conv_input))
            #conved = [batch size, 2*hid dim, protein len]

            #pass through GLU activation function
        conved = F.glu(conved, dim=1)
            #conved = [batch size, hid dim, protein len]

            #apply residual connection / high way
        conved = (conved + conv_input) * self.scale
            #conved = [batch size, hid dim, protein len]
            #set conv_input to conved for next loop iteration
        conved = conved.permute(0, 2, 1)
        # conved = [batch size,protein len,hid dim]
        return conved
class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.sa = SelfAttention(hid_dim, n_heads, dropout, device)
        self.ea = SelfAttention(hid_dim, n_heads, dropout, device)
        self.ff = FeedForward(hid_dim, hid_dim, glu=True, dropout=dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        #self attetion
        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))
        #cross attention
        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))
        #feed forward
        trg = self.ln(trg + self.do(self.ff(trg)))

        return trg

def Multilevelblock(in_layers,out_layer):
    blk = nn.Conv2d(in_layers,out_layer,1,bias=False)
    return blk
class HierarchyT(nn.Module):
    def __init__(self, protein_dim, hid_dim,atom_dim, n_heads,n_enlayers,n_delayers, dropout, device):
        super().__init__()
        self.input_dim=protein_dim
        self.hid_dim=hid_dim
        self.n_enlayers=n_enlayers
        self.n_heads=n_heads

        self.bn=nn.BatchNorm1d(atom_dim)
        self.n_delayers=n_delayers
        self.Encoder = nn.ModuleList(
            [EncoderLayer(hid_dim,i*2+3,dropout , device) for i
             in
             range(n_enlayers)])
        self.Decoder = nn.ModuleList(
            [DecoderLayer(hid_dim,n_heads,dropout,device) for i
             in
             range(n_delayers)])
        Multi_level_att_block = []
        for i in range(n_enlayers):
            Multi_level_att_block.append(Multilevelblock(1+i,1))
        self.multilevelattention = nn.ModuleList(Multi_level_att_block)
        self.device = device
        self.protein_attention = nn.Linear(hid_dim, hid_dim)
        self.compound_attention = nn.Linear(hid_dim, hid_dim)
        self.inter_attention = nn.Linear(hid_dim, hid_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.relu=nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()
        self.fc = nn.Linear(self.input_dim, self.hid_dim)
        self.ft = nn.Linear(atom_dim, hid_dim)
        self.fc1 = nn.Linear(hid_dim*2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)
        self.protmaxpool=nn.AdaptiveMaxPool1d(1)
        self.drugmaxpool=nn.AdaptiveMaxPool1d(1)
    def forward(self,protein,trg,trg_mask=None,src_mask=None):

        src = self.fc(protein)
        trg = self.ft(trg)
        #Multilevel att & talking heads

        talkingsrc=torch.zeros([1,src.shape[1],self.hid_dim]).to(self.device)
        for i in range(self.n_enlayers):
            src = self.Encoder[i](src)
            talkingsrc = torch.cat([talkingsrc,src],dim=0)
            usesrc = self.multilevelattention[i](talkingsrc[1:].unsqueeze(dim=0)).squeeze(dim=0)
            trg = self.Decoder[i](trg,usesrc,trg_mask,src_mask)

        #classifier
        src_att=self.protein_attention(talkingsrc[-1].unsqueeze(dim=0))
        trg_att = self.compound_attention(trg)
        c_att = torch.unsqueeze(trg_att, 2).repeat(1, 1, src.shape[1], 1)
        p_att = torch.unsqueeze(src_att, 1).repeat(1, trg.shape[1], 1, 1)
        Attention_matrix = self.inter_attention(self.relu(c_att+p_att))
        Compound_attetion = torch.mean(Attention_matrix, 2)
        Protein_attetion = torch.mean(Attention_matrix, 1)
        Compound_attetion = self.sigmoid(Compound_attetion.permute(0, 2, 1))
        Protein_attetion = self.sigmoid(Protein_attetion.permute(0, 2, 1))
        CompoundConv = trg.permute(0, 2, 1) * 0.5 + trg.permute(0, 2, 1) * Compound_attetion
        ProteinConv = talkingsrc[-1].unsqueeze(dim=0).permute(0, 2, 1) * 0.5 + talkingsrc[-1].unsqueeze(dim=0).permute(0, 2, 1) * Protein_attetion
        CompoundConv = self.drugmaxpool(CompoundConv).squeeze(2)
        ProteinConv = self.protmaxpool(ProteinConv).squeeze(2)
        pair = torch.cat([CompoundConv, ProteinConv], dim=1)
        Pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(Pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        label = self.out(fully2)

        return label


class Classifier(nn.Module):
    def __init__(self,protein_dim,atom_dim,hid_dim,dropout):
        super().__init__()
        self.protein_dim = protein_dim
        self.atom_dim = atom_dim
        self.hid_dim = hid_dim
        self.protein_attention = nn.Linear(hid_dim, hid_dim)
        self.compound_attention = nn.Linear(hid_dim, hid_dim)
        self.inter_attention = nn.Linear(hid_dim, hid_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()
        self.fp = nn.Linear(self.protein_dim, self.hid_dim)
        self.fc = nn.Linear(self.atom_dim, self.hid_dim)
        self.fc1 = nn.Linear(hid_dim * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.protmaxpool = nn.AdaptiveMaxPool1d(1)
        self.drugmaxpool = nn.AdaptiveMaxPool1d(1)
        self.out = nn.Linear(512, 2)
    def forward(self,compound,protein):
        compound = compound.reshape(-1,32).unsqueeze(dim=0)
        protein = protein.unsqueeze(dim=0)
        #print(compound.shape, protein.shape)
        trg = self.fc(compound)
        src = self.fp(protein)
        src_att = self.protein_attention(src)
        trg_att = self.compound_attention(trg)
        c_att = torch.unsqueeze(trg_att, 2).repeat(1, 1, src.shape[1], 1)
        p_att = torch.unsqueeze(src_att, 1).repeat(1, trg.shape[1], 1, 1)
        Attention_matrix = self.inter_attention(self.relu(c_att + p_att))
        Compound_attetion = torch.mean(Attention_matrix, 2)
        Protein_attetion = torch.mean(Attention_matrix, 1)
        Compound_attetion = self.sigmoid(Compound_attetion.permute(0, 2, 1))
        Protein_attetion = self.sigmoid(Protein_attetion.permute(0, 2, 1))
        # print(trg.shape, Compound_atte.shape)
        CompoundConv = trg.permute(0, 2, 1) * 0.5 + trg.permute(0, 2, 1) * Compound_attetion
        ProteinConv = src.permute(0, 2, 1) * 0.5 + src.permute(
            0, 2, 1) * Protein_attetion
        CompoundConv = self.drugmaxpool(CompoundConv).squeeze(2)
        ProteinConv = self.protmaxpool(ProteinConv).squeeze(2)
        pair = torch.cat([CompoundConv, ProteinConv], dim=1)
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        label = self.out(fully2)
        return label




class Predictor(nn.Module):
    def __init__(self, endecoder,classifier, device, atom_dim=32):
        super().__init__()

        self.endecoder = endecoder
        self.device = device
        self.classifier = classifier
        self.weight = nn.Parameter(torch.FloatTensor(atom_dim, atom_dim))
        nn.init.xavier_uniform_(self.weight)
        #self.Loss = PolyLoss(weight_loss=torch.FloatTensor([0.5, 0.5]).to(device), DEVICE=device, epsilon=1.0)
        #self.init_weight()
    def init_weight(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def classify(self,compound,   protein):
        #protein = torch.mean(protein,dim=0).reshape(1,-1)
        compound  = compound.reshape(1,-1)
        out = self.classifier(compound,protein)
        return out
    def forward(self, compound,   protein):
        # compound = [atom_num, atom_dim]
        # adj = [atom_num, atom_num]
        # protein = [protein len, 100]
        #compound = self.gcn(compound, adj)
        compound = compound.reshape(-1,32)
        compound = torch.unsqueeze(compound, dim=0)
        # compound = [batch size=1 ,atom_num, atom_dim]

        protein = torch.unsqueeze(protein, dim=0)
        # protein =[ batch size=1,protein len, protein_dim]
        out = self.endecoder(protein,compound)
        #out_cls = self.classifier(compound, protein)
        # out = [batch size, 2]
        # out = torch.squeeze(out, dim=0)
        return out

    def __call__(self, compound,protein,correct_interaction, train=True,confuse=False):

        Loss = nn.CrossEntropyLoss()
        Loss2 = nn.BCELoss()

        if train:
            predicted_interaction= self.forward(compound,  protein)
            classifier_pred_interation = self.classifier(compound, protein)
            if confuse:
                predicted_interaction = F.softmax(predicted_interaction,dim=0)
                classifier_pred_interation = F.softmax(classifier_pred_interation,dim=0)
                loss = Loss2(predicted_interaction, correct_interaction)
                loss_classifier = Loss2(classifier_pred_interation, correct_interaction)
            else:
                loss = Loss(predicted_interaction, correct_interaction)
                loss_classifier = Loss(classifier_pred_interation,correct_interaction)
            return predicted_interaction,loss,classifier_pred_interation,loss_classifier

        else:
            predicted_interaction= self.forward(compound,  protein)
            classifier_pred_interation = self.classifier(compound, protein)
            correct_labels = correct_interaction.to('cpu').data.numpy().item()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            cls_ys = F.softmax(classifier_pred_interation, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(ys)
            predicted_scores = ys[0, 1]
            cls_pred_labels = np.argmax(cls_ys)
            cls_pred_scores = cls_ys[0, 1]
            return correct_labels, predicted_labels, predicted_scores,cls_pred_labels,cls_pred_scores


class Trainer(object):
    def __init__(self, model, lr, weight_decay, batch ):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)
        self.batch = batch

    def train(self, tranloader, device,confuse=False):
        batch=len(tranloader)
        self.model.train()
        loss_total = 0
        loss_cls_total = 0
        i = 0
        for item in tqdm(tranloader):
            for data in item:
                compounds,protein,interaction,sample=data
                compounds=compounds.to(device)
                protein=protein.to(device)
                interaction=interaction.to(device)
                if confuse:
                    interaction = torch.tensor([[0.5,0.5]]).to(device)
                y_pred,loss,cls_y_pred,loss_cls =self.model(compounds,protein,interaction,confuse=confuse)
                loss.backward()
                loss_cls.backward()
                loss_cls_total+=loss_cls.item()
                loss_total+=loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss_total,loss_cls_total


def pseudo_labels(Y,S,Y_cls,S_cls,samples,epoch,type=1):
    N = len(Y)
    pseudo =[]
    pseudodata = []
    confuse = []
    confusion = []
    epoch = epoch-1
    for i in range(N):
        if Y[i] == Y_cls[i]:
            pseudo.append([Y[i],S[i],S_cls[i],S[i]+S_cls[i],samples[i]])
        else:
            confuse.append([Y[i],S[i],S_cls[i],S[i]+S_cls[i],samples[i]])
    pseudo = sorted(pseudo,key = lambda x : x[3])
    print(len(pseudo),len(confuse))
    if type == 1:
        list = pseudo[:min(10 + 20 * epoch, len(pseudo))] + pseudo[max(-10 - 20 * epoch, -len(pseudo)):]
        list = SAMPLE(list, min(20 + 10 * epoch, len(list)))
        mix = SAMPLE(confuse, min(5 + 5 * epoch, len(confuse)))
    else:
        list = pseudo[:25 + 50 * epoch] + pseudo[-25 - 50 * epoch:]
        list = SAMPLE(list, 50 + 25 * epoch)
        mix = SAMPLE(confuse, 20 + 20 * epoch)
    count =0
    for idx,sp in enumerate(list):
        y, s ,s_cls,s_sum,sample = sp
        pseudodata.append([idx,sample[0],sample[1],y])
        if int(y)==int(sample[2]):
            count+=1
    for idx,sp in enumerate(mix):
        y, s ,s_cls,s_sum,sample = sp
        confusion.append([idx,sample[0],sample[1],y])
    confusion = np.array(confusion)
    pseudodata = np.array(pseudodata)
    return pseudodata,confusion, count,pseudo

class Tester(object):
    def __init__(self, model):
        self.model = model
    def pseudotrain(self,compounds,proteins,testloader,batch_size,device,epoch,type):
        self.model.eval()
        T, Y, S,Y_cls,S_cls,Samples = [], [], [],[],[],[]
        for item in tqdm(testloader):
            for data in item:
                compound, protein, interaction,sample = data
                #print(type(sample),sample,sample.shape)
                compound = compound.to(device)
                protein = protein.to(device)
                interaction = interaction.to(device)
                correct_labels, predicted_labels, predicted_scores,cls_pred_labels,cls_pred_scores = self.model(compound, protein, interaction,
                                                                                train=False)
                T.append(correct_labels)
                Y.append(predicted_labels)
                S.append(predicted_scores)
                Y_cls.append(cls_pred_labels)
                S_cls.append(cls_pred_scores)
                Samples.append([sample[1],sample[2],sample[3]])
        pseudodata ,confusiondata,count,pseudo= pseudo_labels(Y,S,Y_cls,S_cls,Samples,epoch,type)
        pseudoloader = DataLoader(Mydataset(pseudodata, compounds, proteins), shuffle=True, batch_size=batch_size,
                                 collate_fn=collatef, drop_last=False)
        if len(confusiondata)!=0:
           confusionloader = DataLoader(Mydataset(confusiondata, compounds, proteins), shuffle=True, batch_size=batch_size,
                                  collate_fn=collatef, drop_last=False)
        else:
            confusionloader=None
        AUC = roc_auc_score(T, S)
        AUC_cls = roc_auc_score(T, S_cls)
        tpr, fpr, _ = precision_recall_curve(T, S)
        PRAUC = auc(fpr, tpr)
        AUPRC = average_precision_score(T, S)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        acc = accuracy_score(T, Y)
        return AUC,AUC_cls, PRAUC, AUPRC, precision, recall, acc, pseudoloader,confusionloader


    def test(self, testloader,device):
        self.model.eval()
        T, Y, S,Y_cls,S_cls= [], [], [],[],[]
        for item in tqdm(testloader):
            for data in item:
                compounds, protein, interaction,sample = data
                compounds=compounds.to(device)
                protein=protein.to(device)
                interaction=interaction.to(device)
                correct_labels, predicted_labels, predicted_scores,cls_pred_labels,cls_pred_scores =self.model(compounds,protein,interaction,train=False)
                T.append(correct_labels)
                Y.append(predicted_labels)
                S.append(predicted_scores)
                Y_cls.append(cls_pred_labels)
                S_cls.append(cls_pred_scores)
        AUC = roc_auc_score(T, S)
        AUC_cls = roc_auc_score(T,S_cls)
        tpr, fpr, _ = precision_recall_curve(T, S)
        PRAUC = auc(fpr,tpr)
        AUPRC = average_precision_score(T,S)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        acc = accuracy_score(T, Y)
        return AUC,AUC_cls, PRAUC, AUPRC, precision, recall,acc,S,S_cls

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

