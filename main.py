# -*- coding: utf-8 -*-
"""
@Time:Created on 2023/8/18 4:--
@author: Zhou San Xie
@Filename: main.py
@Software: PyCharm
"""
import torch
import numpy as np
import os
import time
from model.Mlanmodel import *
import timeit
import argparse
import utils.misc as misc
from dataset.getdataset import preparedataset




if __name__ == "__main__":

    """ you can change the hyperparameters here"""
    parser = argparse.ArgumentParser(description='settings')
    parser.add_argument('--batch', type=int, default=64, help='batch size')
    parser.add_argument('--epoch', type=int, default=40, help='epochs')
    parser.add_argument('--dataset', type=str, default="bindingdb", choices=["human","celegans","bindingdb","biosnap"],help='select dataset for training')
    parser.add_argument('--cuda', type=int, default=0, help='device,cuda:0,1,2...')
    parser.add_argument('--type', type=str,default="cluster", choices=["cluster","cold","random"],help='cluster,cold or random')
    parser.add_argument('--usepl', type=str, default="False", choices=["True","False"],help='Use pseudo labeling or not')
    parser.add_argument('--output_file', type=str, default="train1",help='output file name')
    args = parser.parse_args()
    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda:'+str(args.cuda))
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')



    """ param settings """
    if args.type =="cluster":
        config = misc.load_config('./configs/cluster.yaml')
    else :
        config = misc.load_config('./configs/random&cold.yaml')
    misc.seed_all(config.seed)

    config.lr.weight_decay = 1e-4
    config.lr.decay_interval = 5
    config.lr.lr_decay = 0.5

    if args.dataset =='celegans' or args.dataset =='human':
        datatype = 1
    else :
        datatype = 2
    endecoder = HierarchyT(**config.settings,device=device)
    classifier = Classifier(**config.settings_cls)
    model = Predictor(endecoder, classifier,device)
    print(args.usepl,type(args.usepl))
    if args.usepl=="False":
        tranloader, validloader, testloader,_,_ = preparedataset(args.batch, args.type,args.dataset)
        trainer = Trainer(model, config.lr.lr, config.lr.weight_decay, args.batch)
        tester = Tester(model)
        AUCs = (
            'Epoch\tTime(sec)\tLoss_train\tAUC_dev\tAUC_test\tPRAUC\tAUPRC_DEV\tAUPRC_test\tAccuracy\tPrecision_test\tRecall_test\t ACC')
    elif args.usepl=="True" :
        tranloader, validloader, testloader, compounds, proteins = preparedataset(args.batch,args.type,args.dataset)
        trainer = Trainer(model, config.lr.lr, config.lr.weight_decay, args.batch)
        tester = Tester(model)
        AUCs = (
            'Epoch\tTime(sec)\tLoss_train\tAUC_dev\tAUC_test\tPRAUC\tAccuracy\tPrecision_test\tRecall_test\tF1')
    model.to(device)


    """Output files."""
    dir_result ='output/'+str(args.dataset)+'/result/'
    dir_model = 'output/' + str(args.dataset) + '/model/'
    os.makedirs(dir_result, exist_ok=True)
    os.makedirs(dir_model, exist_ok=True)
    file_AUCs = dir_result+args.output_file+'.txt'
    file_model = dir_model+args.output_file+'.pt'
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')
    """Start training."""
    print('Training...')
    print(AUCs)
    start = timeit.default_timer()

    max_AUC_dev = 0

    epoch_label = 0
    for epoch in range(1, args.epoch+1):
        if epoch % config.lr.decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= config.lr.lr_decay
        if args.usepl == "False":
            loss_train = trainer.train(tranloader, device)
            AUC_dev, AUC_clsdev, PRAUC_dev, AUPRC_dev, precision_dev, recall_dev, acc_dev, _, _, _ = tester.test(validloader, device)
            AUC_test, AUC_clstest, PRAUC1, _, precision_test, recall_test, ACC, _, _ = tester.test(testloader, device)
            AUCs = [epoch, time, loss_train, AUC_dev, PRAUC_dev,precision_dev,recall_dev,acc_dev]
            if AUC_dev > max_AUC_dev:
                tester.save_model(model, file_model)
                max_AUC_dev = AUC_dev
                epoch_label = epoch
        elif args.usepl=="True":
            loss_train= trainer.train(tranloader, device)
            AUC_dev,AUC_clsdev, PRAUC_dev, AUPRC_dev, precision_dev, recall_dev, acc_dev, pseudoloader,confusionloader = tester.pseudotrain(compounds,proteins,validloader,args.batch, device,epoch,datatype)
            if AUC_dev > max_AUC_dev:
                tester.save_model(model, file_model)
                max_AUC_dev = AUC_dev
                epoch_label = epoch
            loss_dev = trainer.train(pseudoloader,device)
            if confusionloader!=None:
               _ = trainer.train(confusionloader, device,confuse=True)
            AUC_dev2,AUC_clsdev2, _, _, _,_, _,S2,S2_cls = tester.test(validloader, device)
            if AUC_dev2 > max_AUC_dev:
                tester.save_model(model, file_model)
                max_AUC_dev = AUC_dev2
                epoch_label = epoch

            AUCs = [epoch, time, loss_train, AUC_dev, AUC_dev2,PRAUC_dev,precision_dev,recall_dev,acc_dev]

        end = timeit.default_timer()
        time = end - start
        tester.save_AUCs(AUCs, file_AUCs)
        print('\t'.join(map(str, AUCs)))

    model.load_state_dict(torch.load(file_model))
    AUC_test, AUC_clstest, PRAUC1, _, precision_test, recall_test, ACC, _, _ = tester.test(testloader, device)
    results = (
        'Best Epoch\tAUC_test\tPRAUC\tAccuracy\tPrecision_test\tRecall_test\tF1')
    metric = [epoch_label, AUC_test,  PRAUC1,ACC,
                    precision_test, recall_test,2*precision_test*recall_test/(precision_test+recall_test+1e-5)]
    with open(file_AUCs, 'w') as f:
        f.write(results+ '\n')
    tester.save_AUCs(metric, file_AUCs)
    print("The best model is epoch",epoch_label)

