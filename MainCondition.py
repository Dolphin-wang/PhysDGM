import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from TrainCondition import train, sample
import sys
from data.CMAPSSDataset import CMAPSSDataset
import wandb
from utils import wandb_record,torch_seed
#from eva_regressor_reweight import predictive_score_metrics
from eva_regressor import predictive_score_metrics
from eva_classifier import discrimative_score_metrics
from data_process import load_train_data,load_test_data,load_test_data_rul,load_train_data_rul
#from measure_score.Utils.discriminative_metric import discriminative_score_metrics
#from measure_score.Utils.discriminative_metric import discriminative_score_metrics
from measure_score.Utils.context_fid import Context_FID
from measure_score.Utils.cross_correlation import CrossCorrelLoss
# from data.data_process import load_RUL2012

from args import args
import torch

os.environ["WANDB_MODE"] = "offline"
rmse_list,score_list,acc_list, FID_list, CorrelLoss_list, mae_list ,r2_list= [],[],[],[],[],[],[]


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    if len(sys.argv)==1:
        print('-------no prompt--------')
        args.epoch = 70
        args.dataset = 'FD001'
        args.lr = 2e-3
        args.state = 'eval' # all,train,sample,eval
        args.model_name = 'DiffUnet_fre' 
        args.T = 50
        args.window_size = 48
        args.w = 0
        args.input_size = 18
        args.warmup = 'true'
        args.downstream = 1
    if args.model_name == 'dit':
        wandb.init(project="DiffFre", tags=['EXP-compare'], config=args )
    else:
        wandb.init(project="DiffFre", tags=['Fre-w/o_syn'], config=args )
    train_loop = 1
    torch_seed(6)
    if args.downstream == 1:
        args.model_path =  'weights/' + args.model_name + '_' + args.dataset + '_' + str(args.window_size) + '_' + args.warmup + '.pth'
    else:
        args.model_path =  'weights/' + args.model_name + '_' + args.dataset + '_' + str(args.window_size) + '_' + args.warmup + '_' + str(args.downstream) + '.pth'
    if args.downstream == 1:
        args.syndata_path =  './weights/syn_data/syn_'+ args.dataset+'_'+args.model_name + '_' + str(args.window_size) + args.warmup + '_' + args.sample_type  +'.npz'
    else:
        args.syndata_path =  './weights/syn_data/syn_'+ args.dataset+'_'+args.model_name + '_' + str(args.window_size) + args.warmup + '_' + args.sample_type + '_' + str(args.downstream) +'.npz'

    if 's0' in args.dataset:  
        MAX_rul = True
        args.input_size = 19 if MAX_rul else 18   
        stride = 50
        args.window_size = 50
        if MAX_rul:
            train_data,train_label = load_train_data_rul(sequence_length = args.window_size, stride = stride, dataset=args.dataset) 
            test_data,test_label = load_test_data_rul(sequence_length = args.window_size, stride = stride, dataset=args.dataset)
        else:
            train_data,train_label = load_train_data(sequence_length = args.window_size, stride = stride, dataset=args.dataset) 
            test_data,test_label = load_test_data(sequence_length = args.window_size, stride = stride, dataset=args.dataset)
    elif args.dataset == 'tep':
        args.input_size = 52
        train_data = torch.from_numpy(np.load(f'data/precessd_data/Classification/train_data{args.window_size}_new.npy')).float()
        train_label = torch.from_numpy(np.load(f'data/precessd_data/Classification/train_labels{args.window_size}.npy')).float() 
        test_data = torch.from_numpy(np.load(f'data/precessd_data/Classification/test_data{args.window_size}_new.npy')).float()
        test_label = torch.from_numpy(np.load(f'data/precessd_data/Classification/test_labels{args.window_size}.npy')).float()
    elif args.dataset in ['2c', '3c', 'r25', 'r3', 'rw', 'sate', 'bat', 'battery']:
        args.input_size = 17
        train_data = torch.from_numpy(np.load(f'data/battery/traindata_{args.dataset}_{args.window_size}.npy')).float()
        train_label = torch.from_numpy(np.load(f'data/battery/trainlabel_{args.dataset}_{args.window_size}.npy')).float() 
        test_data = torch.from_numpy(np.load(f'data/battery/testdata_{args.dataset}_{args.window_size}.npy')).float()
        test_label = torch.from_numpy(np.load(f'data/battery/testlabel_{args.dataset}_{args.window_size}.npy')).float()
    else:
        args.input_size = 14
        datasets = CMAPSSDataset(fd_number=args.dataset, sequence_length=args.window_size, deleted_engine=[1000])
        train_data = datasets.get_train_data()
        train_data,train_label = datasets.get_feature_slice(train_data), datasets.get_label_slice(train_data)
        
        test_data = datasets.get_test_data()
        test_data,test_label = datasets.get_last_data_slice(test_data)
    
    train_data,train_label = train_data[0:len(train_data)], train_label[0:len(train_label)]
    if args.downstream > 1:
        train_label_max = train_label.repeat(args.downstream, 1)
    print("train_data.shape:",train_data.shape,"      test_data.shape:",test_data.shape)
    
    
    if args.state == "train" or args.state == "all":
        if args.downstream >= 1:
            train(args,train_data,train_label)
            sample(args,train_label)
        else:
            percentage = args.downstream
            torch.manual_seed(42)
            num_samples = train_data.shape[0]
            num_subset = int(num_samples * percentage) # 10000 * 0.05 = 500
            indices = torch.randperm(num_samples)
            subset_indices = indices[:num_subset]
            data_subset = train_data[subset_indices]
            np.save(f'data_plot/{percentage*100}%train_data_{args.dataset}.npy',data_subset)
            label_subset = train_label[subset_indices]
            np.save(f'data_plot/{percentage*100}%train_label_{args.dataset}.npy',label_subset)
            train(args,data_subset,label_subset)
            sample(args,train_label)
    elif args.state == "sample":
        if args.downstream <= 1:
            sample(args,train_label)
        else:
            sample(args,train_label_max)
    if args.state == "eval" or args.state == "all" or args.state == "sample":
        syn_dataset = np.load(args.syndata_path)
        syn_data = syn_dataset['data']
        syn_label = syn_dataset['label']
        original_data_test = {'data':test_data,'label':test_label}
        original_data_train = {'data':train_data,'label':train_label}
        concat_data = {}
        random_indices = np.random.choice(train_data.shape[0], size=len(train_data) // 10 , replace=False)

        for i in range(train_loop):
            rmse,mae, score,r2= predictive_score_metrics(args, original_data_test, syn_dataset) #syn_dataset)
            #rmse,mae, score= predictive_score_metrics(args, original_data_test, syn_dataset) #syn_dataset)
            Context_FID_score = Context_FID(train_data.cpu().numpy(), syn_data) # Context_FID分数计算
            loss_function = CrossCorrelLoss(train_data,name=args.dataset)
            CrossCorrel_Loss = loss_function(torch.tensor(syn_data))
            discriminative_score = discrimative_score_metrics(args, original_data_train , syn_dataset)
            rmse_list.append(rmse); score_list.append(score); mae_list.append(mae); r2_list.append(r2); acc_list.append(discriminative_score)
            FID_list.append(Context_FID_score); CorrelLoss_list.append(CrossCorrel_Loss)

        print("loss_list",rmse_list,"acc list",acc_list)
        #wandb_record(rmse_list,mae_list,score_list, acc_list,FID_list,CorrelLoss_list)
        #wandb.finish()

