
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader,TensorDataset,random_split
import numpy as np
from DiffusionFreeGuidence.Unet1D_fre_c import UNet1D_fre_c
from DiffusionFreeGuidence.Unet1D_fre_t import UNet1D_fre_t
from DiffusionFreeGuidence.Unet1D_fre_syn import UNet1D_fre_syn
from DiffusionFreeGuidence.Unet1D_fre_syn_ori import UNet1D_fre_syn_ori
from DiffusionFreeGuidence.Unet1D_fre_woct import UNet1D_fre_woct
from DiffusionFreeGuidence.Unet1D import UNet1D
from DiffusionFreeGuidence.Unet1D_fre import UNet1D_fre
from DiffusionFreeGuidence.Diffwave import DiffWave
from DiffusionFreeGuidence.tabddpm import MLPDiffusion
from DiffusionFreeGuidence.Dit import DiT

from DiffusionFreeGuidence.diffwaveimputer import DiffWaveImputer
from DiffusionFreeGuidence.csdi import diff_CSDI

import data.CMAPSSDataset as CMAPSSDataset
import wandb
from GaussianDiffusion import GaussianDiffusion1D_cls_free,GradualWarmupScheduler

def train(args, train_data, train_label):
    device = args.device
    best_loss = 9999

    train_dataset = TensorDataset(train_data.permute(0,2,1).to(device), train_label.to(device))
    if args.downstream < 0.5:
        dataloader= DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)#256
    elif args.downstream < 1:
        dataloader= DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)#256
    else: 
        dataloader= DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)#256
    
    # model setup
    if args.model_name == 'DiffUnet': 
        net_model = UNet1D(dim = 32, dim_mults = (1, 2), cond_drop_prob = 0.2, channels = args.input_size).to(device) #cmapss  dim = 32 dim_mults = (1, 2, 2) cond_drop_prob = 0.5
    if args.model_name == 'DiffUnet_c': 
        net_model = UNet1D_fre_c(dim = 32, dim_mults = (1, 2), cond_drop_prob = 0.2, channels = args.input_size, length = args.window_size).to(device) 
    if args.model_name == 'DiffUnet_t': 
        net_model = UNet1D_fre_t(dim = 32, dim_mults = (1, 2), cond_drop_prob = 0.2, channels = args.input_size, length = args.window_size).to(device) 
    if args.model_name == 'DiffUnet_fre': 
        net_model = UNet1D_fre(dim = 32, dim_mults = (1,2), cond_drop_prob = 0.3, channels = args.input_size, length = args.window_size).to(device) 
    if args.model_name == 'DiffUnet_woct': 
        net_model = UNet1D_fre_woct(dim = 32, dim_mults = (1,2), cond_drop_prob = 0.2, channels = args.input_size, length = args.window_size).to(device) 
    if args.model_name == 'DiffUnet_wosyn': 
        net_model = UNet1D_fre_syn(dim = 32, dim_mults = (1,2), cond_drop_prob = 0.2, channels = args.input_size, length = args.window_size).to(device) 
    if args.model_name == 'DiffUnet_wosyn_ori': 
        net_model = UNet1D_fre_syn_ori(dim = 32, dim_mults = (1,2), cond_drop_prob = 0.2, channels = args.input_size, length = args.window_size).to(device) 
    if args.model_name == 'dit': 
        net_model = DiT(input_size=args.input_size,hidden_size=128, num_heads=1).to(device) 
    if args.model_name == 'DiffWave': 
        net_model = DiffWaveImputer(in_channels=args.input_size, out_channels=args.input_size, seq_length=args.window_size)
        # net_model = DiffWave(residual_channels=64,window_size=args.window_size,dilation_cycle_length=10,
        #        unconditional=False,residual_layers=5, time_step=args.T)
    if args.model_name == 'SSSD': 
        net_model = SSSDS4Imputer(seq_length=args.window_size, s4_lmax=args.window_size )
    if args.model_name == 'tabddpm':     
        net_model = MLPDiffusion( d_in=14, num_classes=0, is_y_cond=True, window_size=args.window_size, rtdl_params={'d_layers': [128, 256, 256, 128], 'dropout': 0.3}, dim_t = 512)
    if args.model_name == 'csdi': 
        net_model = diff_CSDI(inputdim=14)   
                 
    optimizer = torch.optim.AdamW(net_model.parameters(), lr=args.lr, weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epoch, eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=args.multiplier,
                                             warm_epoch=args.epoch // 10 + 1, after_scheduler=cosineScheduler)
    print(args.T)
    trainer = GaussianDiffusion1D_cls_free(net_model, seq_length=args.window_size, args = args, channels = args.input_size, timesteps =  args.T, objective='pred_noise').to(device)

    # start training
    for e in range(args.epoch):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            args.now_epoch = e
            loss_list=[]
            for images, labels in tqdmDataLoader:
                # train
                b = images.shape[0]
                optimizer.zero_grad()
                x_0 = images.to(device)
                labels = labels.to(device)
                if np.random.rand() < 0.1:
                    labels = torch.zeros_like(labels).to(device)
                loss = trainer(x_0, classes = labels).sum()
                loss_list.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), args.grad_clip)
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": sum(loss_list)/len(loss_list),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        current_loss = sum(loss_list)/len(loss_list)
        wandb.log({"Diffusion_Loss":current_loss})
        if e > 5 and current_loss < best_loss:
            torch.save(net_model.state_dict(), args.model_path)
            print('*******imporove!!!********')

def sample(args, train_label = None):
    if train_label == None:
        datasets = CMAPSSDataset.CMAPSSDataset(fd_number=args.dataset, sequence_length=args.window_size ,deleted_engine=[1000])
        train_data = datasets.get_train_data()
        train_label = datasets.get_label_slice(train_data)        
    device = args.device
    # load model and evaluate
    with torch.no_grad():
        if args.model_name == 'DiffUnet': 
            net_model = UNet1D(dim = 32, dim_mults = (1, 2), cond_drop_prob = 0.2, channels = args.input_size).to(device) #cmapss  dim = 32 dim_mults = (1, 2, 2) cond_drop_prob = 0.5
        if args.model_name == 'DiffUnet_c': 
            net_model = UNet1D_fre_c(dim = 32, dim_mults = (1, 2), cond_drop_prob = 0.2, channels = args.input_size, length = args.window_size).to(device) 
        if args.model_name == 'DiffUnet_t': 
            net_model = UNet1D_fre_t(dim = 32, dim_mults = (1, 2), cond_drop_prob = 0.2, channels = args.input_size, length = args.window_size).to(device) 
        if args.model_name == 'DiffUnet_fre': 
            net_model = UNet1D_fre(dim = 32, dim_mults = (1,2), cond_drop_prob = 0.3, channels = args.input_size, length = args.window_size).to(device) 
        if args.model_name == 'DiffUnet_woct': 
            net_model = UNet1D_fre_woct(dim = 32, dim_mults = (1,2), cond_drop_prob = 0.2, channels = args.input_size, length = args.window_size).to(device) 
        if args.model_name == 'DiffUnet_wosyn': 
            net_model = UNet1D_fre_syn(dim = 32, dim_mults = (1,2), cond_drop_prob = 0.2, channels = args.input_size, length = args.window_size).to(device) 
        if args.model_name == 'DiffUnet_wosyn_ori': 
            net_model = UNet1D_fre_syn_ori(dim = 32, dim_mults = (1,2), cond_drop_prob = 0.2, channels = args.input_size, length = args.window_size).to(device)
        if args.model_name == 'DiffWave': 
            net_model = DiffWaveImputer(in_channels=args.input_size, out_channels=args.input_size, seq_length=args.window_size)
            # net_model = DiffWave(residual_channels=64,window_size=args.window_size,dilation_cycle_length=10,
            #        unconditional=False,residual_layers=5, time_step=args.T)
        if args.model_name == 'SSSD': 
            net_model = SSSDS4Imputer(seq_length=args.window_size, s4_lmax=args.window_size )
        if args.model_name == 'tabddpm':     
            net_model = MLPDiffusion( d_in=14, num_classes=0, is_y_cond=True, window_size=args.window_size, rtdl_params={'d_layers': [128, 256, 256, 128], 'dropout': 0.0}, dim_t = 512)
        if args.model_name == 'csdi':     
            net_model = diff_CSDI(inputdim=14)
        if args.model_name == 'dit': 
            net_model = DiT(input_size=args.input_size,hidden_size=128, num_heads=1).to(device)                               
        ckpt = torch.load(args.model_path)
        net_model.load_state_dict(ckpt)
        print("model load weight done.")
        net_model.eval()
        diffusion = GaussianDiffusion1D_cls_free(net_model, seq_length=args.window_size, channels = args.input_size,args = args,  timesteps =  args.T, objective='pred_noise').to(device)
        # Sampled from standard normal distribution
        if args.downstream <= 1:
            if args.dataset =='FD004':
                total_size = train_label.size(0)
                split_points = [total_size // 2, total_size - total_size // 2]  
                split_labels = torch.split(train_label, split_points)
                sample_data = []
                for i, label in enumerate(split_labels):
                    sampled = diffusion.sample(classes=label.to(device)).permute(0, 2, 1)
                    sample_data.append(sampled)
                    #print(f"Sampled data {i+1} shape: {sampled.shape}")
                sampledata = torch.cat(sample_data, dim=0)
                sampledata = sampledata.cpu().numpy()
            else:
                sampledata = diffusion.sample(classes = train_label.to(device)).permute(0,2,1)
                sampledata = sampledata.cpu().numpy()
        else:
            total_size = train_label.size(0)
            args.downstream = args.downstream * 2
            split_size = total_size // args.downstream
            split_points = [split_size] * (args.downstream - 1) + [total_size - split_size * (args.downstream - 1)]
            split_labels = torch.split(train_label, split_points)
            sample_data = []
            for i, label in enumerate(split_labels):
                sampled = diffusion.sample(classes=label.to(device)).permute(0, 2, 1)  # 生成样本
                sample_data.append(sampled)
                # print(f"Sampled data {i+1} shape: {sampled.shape}")
            sampledata = torch.cat(sample_data, dim=0)
            sampledata = sampledata.cpu().numpy()
        if args.warmup != 'none' and args.dataset == 'FD001' or args.dataset == 'FD003':
            if args.dataset == 'FD001':
                discrete_num = 12
            elif args.dataset == 'FD003':
                discrete_num = 11
            sampledata[:, :, 11] = np.round(sampledata[:, :, 11] * discrete_num) / discrete_num
        np.savez(args.syndata_path,data=sampledata, label =  train_label.cpu().numpy())      

        #np.savez(args.syndata_path,data=sampledata.cpu().numpy(), label =  train_label.cpu().numpy())
    return sampledata