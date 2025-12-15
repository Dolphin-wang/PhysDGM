import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from visualizationMetrics import visualization
import data.CMAPSSDataset as CMAPSSDataset
# from DiffusionFreeGuidence.DiffusionCondition import GaussianDiffusionTrainer,extract
from GaussianDiffusion import GaussianDiffusion1D_cls_free
# from DiffusionFreeGuidence.Unet1D_TD import moving_maxavg,SinusoidalPosEmb
import torch 

def plot_curve1(train_data,fake_data,model_name,type='real'):
    #Reshaping the data
    cols_name = ['s2', 's3','s4',  's7', 's8', 's9',  's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    cols = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    cols = [1,2,8,9,12]
    
    #Plotting some generated samples. Both Synthetic and Original data are still standartized with values between [0,1]
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 4))
    axes=axes.flatten()
    
    cycle = 50
    for i,col in enumerate(cols):
        df = pd.DataFrame({'Real': train_data[cycle][:, col],
                    'Synthetic': fake_data[cycle][:, col],
                    })
        if 'GAN' in model_name:
            df.plot(ax=axes[i], title = cols_name[col], lw = 3,color=['black', 'green'], style=['-', '--'])
        else:
            df.plot(ax=axes[i], title = cols_name[col], lw = 3,color=['black', 'orange'], style=['-', '--'])
        axes[i].legend(prop={'size': 18},loc=2)
        axes[i].set_title(cols_name[col],fontsize=24)
    fig.tight_layout()
    plt.savefig('./submits/figure/' + model_name + str(cycle)+ '.png', dpi=100)
    
def plot_curve2(train_data,fake_data,model_name,type='real'):
    #Reshaping the data
    cols_name = ['s2', 's3','s4',  's7', 's8', 's9',  's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    cols = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    cols = [1,2,8,9,12]
    
    #Plotting some generated samples. Both Synthetic and Original data are still standartized with values between [0,1]
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 4))
    axes=axes.flatten()
    
    cycle = 60
    for i,col in enumerate(cols):
        
        # train_data += np.random.randn(41539, 48, 14) / 50

        df_real = pd.DataFrame({'syn': train_data[cycle][:, col],})
        df_Synthetic = pd.DataFrame({'Synthetic': fake_data[cycle][:, col],})   
        
        if type == 'real':
            df_real.plot(ax=axes[i], title = cols_name[col], lw = 3,color=['black'], style=['-'])            
        elif 'GAN' in model_name:
            df_Synthetic.plot(ax=axes[i], title = cols_name[col], lw = 3,color=['green'], style=['--'])
        elif 'Unet' in model_name:
            df_Synthetic.plot(ax=axes[i], title = cols_name[col], lw = 3,color=['orange'], style=['--'])
        else:
            df_Synthetic.plot(ax=axes[i], title = cols_name[col], lw = 3,color=['blue'], style=['--'])
        axes[i].legend(prop={'size': 18},loc=2)
        axes[i].set_title(cols_name[col],fontsize=24)
        axes[i].set_ylim(0,1)
    fig.tight_layout()
    if type == 'real':
        plt.savefig('./submits/figure/' + str(cycle)+ '.png', dpi=100)
    else:     
        plt.savefig('./submits/figure/' + model_name + str(cycle)+ '.png', dpi=100)
    plt.show()
    
def load_data(model_name,datasets,syn=True):
    syndata = 0
    if syn:
        syndata_path =  './weights/syn_data/syn_'+ dataset+'_'+model_name + '_' + str(window_size)+'ddim' +'.npz'
        loaded_data = np.load(syndata_path)
        syndata = loaded_data['data']
    
    train_data = datasets.get_train_data()
    train_data,train_label = datasets.get_feature_slice(train_data), datasets.get_label_slice(train_data)  
    
    return syndata,train_data

def noise_signal(x_0,t):
    Trainer = GaussianDiffusion1D_cls_free(1, beta_1=1e-4 ,beta_T=0.028, T=500 )
    t = torch.tensor(t).unsqueeze(0)      
    noise = np.random.rand(x_0.shape[0],x_0.shape[1])
    x_0 = x_0.unsqueeze(0)
    x_t = extract(Trainer.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
        extract(Trainer.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise    
    return x_t

def plot_two_normalizationdata(sample,window_size,str):

    print(sample.shape)
    x_range = range(window_size)

    # plt.xticks([])# 去掉x轴
    fig,ax = plt.subplots(figsize=(8,5),tight_layout=True)
    plt.plot(x_range,sample[:,5],linewidth=10,color='royalblue' )
    plt.plot(x_range,sample[:,7],linewidth=10,color ='#00A086')
    # ax.patch.set_facecolor('whitesmoke')

    plt.ylim(0,1)
    plt.xticks([])
    plt.yticks([])

    plt.grid()
    plt.tight_layout()
    plt.savefig('./submits/figure/nomalization' + str+ '.png')
    plt.show() 
    
def plot_normalizationdata(sample,dataset='FD003'):
    """
    plot multisensor time series fig which contain some sensor
    """
    # syndata[952]
    print(sample.shape)

    x_range = range(sample.shape[0])

    # plt.xticks([])# 去掉x轴
    fig,ax = plt.subplots(figsize=(16,5),tight_layout=True)
    for i in [1,2,5,6,8,9]:
        plt.plot(x_range,sample[:,i],linewidth=2.5)

    ax.set_xlabel('Time Steps',fontsize=30)
    ax.set_ylabel('Normalization Data',fontsize=30)

    plt.ylim(0,0.75)
    plt.xticks(fontproperties = 'Times New Roman', size = 24)
    plt.yticks(fontproperties = 'Times New Roman', size = 24)

    plt.grid()
    plt.tight_layout()
    plt.savefig('./submits/figure/nomalizationb'+dataset+'.png')
    plt.show()   


def plot_ada_mmd():
    groups = ["FD001-48", "FD002-48", "FD003-48", "FD004-48"]
    RMSE16 = [14.667, 25.815, 17.950, 28.926]
    RMSE24 = [13.976, 25.596, 18.350, 27.220]
    RMSE32 = [13.616, 24.408, 17.184, 27.072]

    xticks = np.arange(0,7.2,1.8)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    bar_width = 0.5
    ax.bar(xticks, RMSE16, width=bar_width, label="DiffIMTS w/o Ada-MMD", color="#4dbbd6", yerr=0.1)
    ax.bar(xticks + bar_width, RMSE24, width=bar_width, label="DiffIMTS w/o Ada", color="#3d5488", yerr=0.1)
    ax.bar(xticks + 2 * bar_width, RMSE32, width=bar_width, label="DiffIMTS", color="#e64b35", yerr=0.1)

    # plt.text(0.0, RMSE16[0], '*', ha='center', va='bottom', fontsize=30)
    # plt.text(2.3, RMSE32[1], '*', ha='center', va='bottom', fontsize=30)
    # plt.text(4.1, RMSE32[2], '*', ha='center', va='bottom', fontsize=30)
    
    plt.xticks(fontproperties = 'Times New Roman', size = 18)
    plt.yticks(fontproperties = 'Times New Roman', size = 18)
    ax.set_xlabel("Dataset", fontsize=22)
    ax.set_ylabel("Predictive Score", fontsize=22)
    ax.legend(fontsize=18)

    ax.set_xticks(xticks + 0.5)
    ax.set_ylim(10,30)
    ax.set_xticklabels(groups)
    plt.tight_layout()
    plt.grid()
    plt.savefig('submits/figure/ablation-mmd.png',dpi=400)
    plt.show()
        
if __name__ == '__main__':
    
    # plotTimestepEembed()
    
    dataset = 'FD001'
    window_size = 96
    
    diff_t = 499
    datasets = CMAPSSDataset.CMAPSSDataset(fd_number=dataset, sequence_length=window_size ,deleted_engine=[1000])
    model_name = 'DiffUnet'
    model_list = ['dit']
    # model_list = ['DiffUnet_fre']
    for model_name in model_list:
        
        syndata, train_data = load_data(model_name,datasets,syn = True) 
        print(syndata.shape)
        
        # x_t = noise_signal(train_data[500],0).squeeze(0)
        # plot_two_normalizationdata(x_t,window_size,str='t='+str(diff_t))
        
        # plot_maxavg( train_data[0],type='recontruct') # 'max' 'avg' 'recontruct'
        # plot_normalizationdata(syndata[952])
        # plot_curve2(train_data,syndata,model_name+dataset+'_cycle',type='rea')
        visualization(train_data, syndata, 'pca', model_name+'-pca')  
        visualization(train_data, syndata, 'tsne', model_name+'-tsne')
        # plot_ada_mmd()