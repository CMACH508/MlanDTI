import numpy as np
import pandas as pd
import os
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='settings')
    parser.add_argument('--dataset', type=str, default="bindingdb", choices=["human","celegans","bindingdb","biosnap"],help='select dataset for training')
    parser.add_argument('--split_settings', type=str, default="random", choices=["random","cold","cluster"],help='select split settings')   
    args = parser.parse_args()

    data_path = os.path.join('./data',args.dataset)
    dir_path = os.path.join(data_path, args.split_settings)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if not os.path.exists(dir_path+"/train/"):
        os.makedirs(dir_path+"/train/")
    if not os.path.exists(dir_path+"/valid/"):
        os.makedirs(dir_path+"/valid/")
    if not os.path.exists(dir_path+"/test/"):
        os.makedirs(dir_path+"/test/")
    data_path = os.path.join(data_path,"fulldata.csv")
    full=pd.read_csv(data_path)
    if args.split_settings is 'cluster':
        source = []
        target = []

        trg_cluster=np.array(list(set(full['target_cluster'].values)))
        drug_cluster=np.array(list(set(full['drug_cluster'].values)))
        print(trg_cluster.shape)
        print(drug_cluster.shape)
        print(max(full['target_cluster'].values))
        print(max(full['drug_cluster'].values))
        np.random.shuffle(trg_cluster)
        np.random.shuffle(drug_cluster)
        trg_src,trg_trg=np.split(trg_cluster,[int(0.6*trg_cluster.shape[0])])
        drug_src,drug_trg=np.split(drug_cluster,[int(0.6*drug_cluster.shape[0])])
        print(full.values.shape)

        smiledictnum2str={};smiledictstr2num={}
        sqddictnum2str={};sqdictstr2num={}
        trainsamples=[]
        valtest_example=[]
        smilelist=[]
        sequencelist=[]
        for no,data in enumerate(full.values):
            smiles,sequence,interaction,d_cluster,t_cluster =data
            if smiledictstr2num.get(smiles) == None:
                smiledictstr2num[smiles] = len(smiledictstr2num)
                smiledictnum2str[str(len(smiledictnum2str))] = smiles
                smilelist.append(smiles)
            if sqdictstr2num.get(sequence) == None:
                sqdictstr2num[sequence] = len(sqdictstr2num)
                sqddictnum2str[str(len(sqddictnum2str))] = sequence
                sequencelist.append(sequence)
            if d_cluster in drug_src and t_cluster in trg_src:
                smilesidx = int(smiledictstr2num.get(smiles))
                sequenceidx = int(sqdictstr2num[sequence])
                trainsamples.append([smilesidx, sequenceidx, int(interaction)])
                source.append(data)
            if d_cluster in drug_trg and t_cluster in trg_trg:
                target.append(data)
                smilesidx = int(smiledictstr2num.get(smiles))
                sequenceidx = int(sqdictstr2num[sequence])
                valtest_example.append([smilesidx, sequenceidx, int(interaction)])
        print(len(source),len(target))

        valsamples=valtest_example[:int(0.8*len(valtest_example))]
        testsamples=valtest_example[int(0.8*len(valtest_example)):]
        target_train=target[0:int(0.8*len(target))]
        target_test=target[int(0.8*len(target)):]
        column=['SMILES','Protein','Y','drug_cluster','target_cluster']
        sourcesamples=pd.DataFrame(columns=column,data=source)
        sourcesamples.to_csv(dir_path+'/source_train.csv',index=False)
        targetsamples=pd.DataFrame(columns=column,data=target_train)
        targetsamples.to_csv(dir_path+'/target_train.csv',index=False)
        targetsamples=pd.DataFrame(columns=column,data=target_test)
        targetsamples.to_csv(dir_path+'/target_test.csv',index=False)
        column=['smiles','sequence','interactions']
        Tr=pd.DataFrame(columns=column, data=trainsamples)
        Tr.to_csv(dir_path+'/train/samples.csv')
        val=pd.DataFrame(columns=column,data=valsamples)
        val.to_csv(dir_path+'/valid/samples.csv')
        Ts=pd.DataFrame(columns=column,data=testsamples)
        Ts.to_csv(dir_path+'/test/samples.csv')
        print(len(target_train),len(target_test))
    elif args.split_settings is 'cold':
        train = []
        valtest = []

        smiledictnum2str={};smiledictstr2num={}
        sqddictnum2str={};sqdictstr2num={}
        trainsamples=[]
        valtest_example=[]
        smilelist=[]
        sequencelist=[]

        for no,data in enumerate(full.values):
            smiles,sequence,interaction,_,_ =data
            if smiledictstr2num.get(smiles) == None:
                smiledictstr2num[smiles] = len(smiledictstr2num)
                smiledictnum2str[str(len(smiledictnum2str))] = smiles
                smilelist.append(smiles)
            if sqdictstr2num.get(sequence) == None:
                sqdictstr2num[sequence] = len(sqdictstr2num)
                sqddictnum2str[str(len(sqddictnum2str))] = sequence
                sequencelist.append(sequence)
        trg_cluster=np.array(list(sqddictnum2str.keys()),dtype=int).squeeze()
        drug_cluster=np.array(list(smiledictnum2str.keys()),dtype=int).squeeze()
        print(trg_cluster.shape)
        print(drug_cluster.shape)
        np.random.shuffle(trg_cluster)
        np.random.shuffle(drug_cluster)
        trg_src,trg_trg=np.split(trg_cluster,[int(0.7*trg_cluster.shape[0])])
        drug_src,drug_trg=np.split(drug_cluster,[int(0.7*drug_cluster.shape[0])])
        for no,data in enumerate(full.values):
            smiles,sequence,interaction,_,_ =data
            # smiles, sequence, interaction, d_cluster, t_cluster = data
            if smiledictstr2num.get(smiles) in drug_src and sqdictstr2num.get(sequence) in trg_src:
                train.append(data)
                smilesidx = int(smiledictstr2num.get(smiles))
                sequenceidx = int(sqdictstr2num[sequence])
                trainsamples.append([smilesidx, sequenceidx, int(interaction)])
            if smiledictstr2num.get(smiles) in drug_trg and sqdictstr2num.get(sequence) in trg_trg:
                valtest.append(data)
                smilesidx = int(smiledictstr2num.get(smiles))
                sequenceidx = int(sqdictstr2num[sequence])
                valtest_example.append([smilesidx, sequenceidx, int(interaction)])
        print(len(train),len(valtest))
        val=valtest[0:int(0.3*len(valtest))]
        test=valtest[int(0.3*len(valtest)):]
        valsamples=valtest_example[0:int(0.3*len(valtest_example))]
        testsamples=valtest_example[int(0.3*len(valtest_example)):]
        print(len(val),len(valsamples),len(test),len(testsamples))
        column=['SMILES','Protein','Y','drug_cluster','target_cluster']
        sourcesamples=pd.DataFrame(columns=column,data=train)
        sourcesamples.to_csv(dir_path+'/train.csv',index=False)
        targetsamples=pd.DataFrame(columns=column,data=val)
        targetsamples.to_csv(dir_path+'/val.csv',index=False)
        targetsamples=pd.DataFrame(columns=column,data=test)
        targetsamples.to_csv(dir_path+'/test.csv',index=False)
        column=['smiles','sequence','interactions']
        Tr=pd.DataFrame(columns=column, data=trainsamples)
        Tr.to_csv(dir_path+'/train/samples.csv')
        val=pd.DataFrame(columns=column,data=valsamples)
        val.to_csv(dir_path+'/valid/samples.csv')
        Ts=pd.DataFrame(columns=column,data=testsamples)
        Ts.to_csv(dir_path+'/test/samples.csv')
    elif args.split_settings is 'random':
        smiledictnum2str={};smiledictstr2num={}
        sqddictnum2str={};sqdictstr2num={}
        trainsamples=[]
        valtest_example=[]
        smilelist=[]
        sequencelist=[]
        samples = []
        for no,data in enumerate(full.values):
            smiles,sequence,interaction,_,_ =data
            smilesidx=0;sequenceidx=0
            if smiledictstr2num.get(smiles) == None:
                smiledictstr2num[smiles] = len(smiledictstr2num)
                smiledictnum2str[str(len(smiledictnum2str))] = smiles
                smilesidx=int(smiledictstr2num[smiles])
                smilelist.append(smiles)
            else:
                smilesidx=int(smiledictstr2num.get(smiles))
            if sqdictstr2num.get(sequence) == None:
                sqdictstr2num[sequence] = len(sqdictstr2num)
                sqddictnum2str[str(len(sqddictnum2str))] = sequence
                sequenceidx=int(sqdictstr2num[sequence])
                sequencelist.append(sequence)
            else:
                sequenceidx=int(sqdictstr2num[sequence])
            samples.append([smilesidx,sequenceidx,int(interaction)])
        samples = np.array(samples)
        N = samples.shape[0]
        if args.dataset == 'bindingdb' or args.dataset == 'biosnap':
            trainsamples,valsamples,testsamples = np.split(samples,[int(0.7*N),int(0.8*N)])
        elif args.dataset == 'human' or args.dataset == 'celegans': 
            trainsamples,valsamples,testsamples = np.split(samples,[int(0.8*N),int(0.9*N)])
        column=['smiles','sequence','interactions']
        Tr=pd.DataFrame(columns=column, data=trainsamples)
        Tr.to_csv(dir_path+'/train/samples.csv')
        val=pd.DataFrame(columns=column,data=valsamples)
        val.to_csv(dir_path+'/valid/samples.csv')
        Ts=pd.DataFrame(columns=column,data=testsamples)
        Ts.to_csv(dir_path+'/test/samples.csv')   


    






