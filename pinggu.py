import numpy as np
import tensorflow as tf
from  measure_score.Utils.predictive_metric import predictive_score_metrics
from measure_score.Utils.discriminative_metric import discriminative_score_metrics
from measure_score.Utils.predictive_metric_lstm import predictive_score_metrics_lstm
from measure_score.Utils.Data_utils.real_datasets import CustomDataset
from measure_score.Utils.predictive_metric_RUL import predictive_score_metrics_RUL
import matplotlib.pyplot as plt
from measure_score.Utils.context_fid import Context_FID
from measure_score.Utils.cross_correlation import CrossCorrelLoss
import torch

def read_and_modify_npy_file(data):
    # 获取数据形状
    shape = data.shape
    # 确保数据形状为（10003，50，19）
    if len(shape) != 3 or shape[1] != 50 or shape[2] != 19:
        raise ValueError("数据形状必须为(10003,50,19)")
    # 将最后一列数据乘以100
    data[:, :, -1] *= 100
    return data

# 加载原始数据和生成的数据
#original_data = np.load('D:/python/TimeGANPyTorch_easy/model_ETTh_new/ori_data_ETTh.npy')
#generated_data = np.load('D:/python/TimeGANPyTorch_easy/model_ETTh_new/generated_data_ETTh.npy')

# original_data = np.load('./OUTPUT/energy/samples/energy_norm_truth_24_train.npy')
#original_data = np.load('D:/python/bishe/Diffusion-TS-main/Data/train_data.npy')
#original_data = np.load('D:/python/bishe/TimeGAN_PytorchRebuild-master/output/energy/ori_data_energy.npy')

#generated_data = np.load('./OUTPUT/s02/ddpm_fake_s02.npy')
#generated_data = np.load('D:/python/TimeGANPyTorch_easy/fd001_model/generated_data_fd001.npy')
# generated_data = np.load('./OUTPUT/energy/ddpm_fake_energy.npy')
#generated_data = np.load('D:/python/bishe/TimeGAN_PytorchRebuild-master/output/energy/generated_data_energy.npy')

original_data = np.zeros((10000, 48, 19))+0.01
generated_data = np.random.randn(10000, 48, 19)

print("generated_data.shape",generated_data.shape)

'''generated_data = read_and_modify_npy_file(generated_data)
ground_data = read_and_modify_npy_file(ground_data)'''
'''new_fake_data = np.load('D:/python/TimeGANPyTorch_easy/model_s02_new/generated_data_s02.npy')
new_fake_data = read_and_modify_npy_file(new_fake_data)'''

# 计算预测得分
# score_mae, score_rmse = predictive_score_metrics(original_data,generated_data)
# print("预测得分MAE:", score_mae)
# print("预测得分RMSE", score_rmse)

# '''score_mae_lstm, score_rmse_lstm = predictive_score_metrics_lstm(original_data,generated_data)
# print("预测得分MAE_LSTM:", score_mae_lstm)
# print("预测得分RMSE_LSTM:", score_rmse_lstm)'''

# score_mae_rul, score_rmse_rul, y_mb, y_pred= predictive_score_metrics_RUL(original_data,generated_data)
# print("预测得分MAE:", score_mae_rul)
# print("预测得分RMSE", score_rmse_rul)

# #分类器得分
# discriminative_score, fake_accuracy, real_accuracy = discriminative_score_metrics(original_data, generated_data)
# #discriminative_score, fake_accuracy = discriminative_score_metrics(original_data, generated_data)
# print("Discriminative Score:", discriminative_score)
# print("Fake Accuracy:", fake_accuracy)
# print("Real Accuracy:", real_accuracy)

#Context_FID分数计算
Context_FID_score = Context_FID(original_data,generated_data) 
print("Context_FID_score:", Context_FID_score)

#Correlation分数计算
loss_function = CrossCorrelLoss(original_data,name='s02')
loss = loss_function(torch.tensor(generated_data))
print("损失值:", loss.item())