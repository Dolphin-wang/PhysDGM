import numpy as np
import torch
import os

def windowarray(array, windowsize, fpt, array_rul=0):
    ###
    # 1、读取数据
    # 2、
    array = torch.from_numpy(array).float()
    res = torch.randn(1, windowsize, array.size(1))
    cur_res = array[0].unsqueeze(0).float()
    for i in range(1, array.size(0)):
        temp = array[i].unsqueeze(0)
        if i < windowsize - 1:
            cur_res = torch.cat((cur_res, temp), 0)
        else:
            cur_res = torch.cat((cur_res, temp), 0)
            res = torch.cat((res, cur_res.unsqueeze(0)), 0)
            cur_res = cur_res[1:]

    l = res[1:].size(0) + array_rul
    label = []
    for i in range(res[1:].size(0)):
        # label.append((l - 1 - i)/l)  #线性

        # if i <= l - fpt - 1:
        #     label.append(1)
        # else:
        #     label.append((l - 1 - i)/fpt)  #0-1

        if i <= l - fpt - 1:
            label.append(fpt)
        else:
            label.append(l - 1 - i)  #125-0
    res = res[1:]
    # start = torch.ones((res.size(0), 1, res.size(2)))
    # res = torch.cat((res, start), 1) #加标志位
    # # res = torch.cat((start, res), 1)
    return res.float(), torch.tensor(label).float().view(-1, 1)

def load_train_data_rul(sequence_length = 50, stride = 1, dataset = 's02'):
    dataset_path = 'N-CMAPSS/' + dataset + '/train'
    file_lists = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path)]
    data = np.load(file_lists[0])
    max_rul = np.full((data.shape[0],1),fill_value=data[:,0].max())/100 
    data = np.concatenate((data,max_rul),axis=1)
    for index_file in file_lists:
        index_data = np.load(index_file)
        max_rul = np.full((index_data.shape[0],1),fill_value=index_data[:,0].max())/100 
        index_data = np.concatenate((index_data,max_rul),axis=1)
        data = np.concatenate((data,index_data), axis=0)

    window_list = []
    label_list = []
    num_samples = int((data.shape[0] - sequence_length)/stride) + 1

    for i in range(num_samples):
        window = data[i*stride:i*stride + sequence_length, 4:]  # each individual window
        label = data[i*stride + sequence_length-1, 0]
        window_list.append(window)
        label_list.append([label])

    sample_array = np.array(window_list).astype(np.float32)
    
    label_array = np.array(label_list).astype(np.float32)

    data = torch.from_numpy(sample_array)
    label = torch.from_numpy(label_array)
    return data,label

def load_test_data_rul(sequence_length = 50, stride = 1, dataset='s02'):
    dataset_path = 'N-CMAPSS/' + dataset + '/test'
    file_lists = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path)]
    data = np.load(file_lists[0])
    max_rul = np.full((data.shape[0],1),fill_value=data[:,0].max())/100
    data = np.concatenate((data,max_rul),axis=1)
    for index_file in file_lists:
        index_data = np.load(index_file)
        max_rul = np.full((index_data.shape[0],1),fill_value=index_data[:,0].max())/100 
        index_data = np.concatenate((index_data,max_rul),axis=1)
        data = np.concatenate((data,index_data), axis=0)
    window_list = []
    label_list = []    
    num_samples = int((data.shape[0] - sequence_length)/stride) + 1

    for i in range(num_samples):
        window = data[i*stride:i*stride + sequence_length, 4:]  # each individual window
        label = data[i*stride + sequence_length-1, 0]
        window_list.append(window)
        label_list.append([label])

    sample_array = np.array(window_list).astype(np.float32) 
    label_array = np.array(label_list).astype(np.float32)

    data = torch.from_numpy(sample_array)
    label = torch.from_numpy(label_array)
    return data,label

def load_train_data(sequence_length = 50, stride = 1, dataset='s02'):
    dataset_path = 'N-CMAPSS/' + dataset + '/train'
    file_lists = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path)]
    data = np.load(file_lists[0])
    for index_file in file_lists:
        index_data = np.load(index_file)
        data = np.concatenate((data,index_data), axis=0)

    window_list = []
    label_list = []
    num_samples = int((data.shape[0] - sequence_length)/stride) + 1

    for i in range(num_samples):
        window = data[i*stride:i*stride + sequence_length, 4:]  # each individual window
        label = data[i*stride + sequence_length-1, 0]
        window_list.append(window)
        label_list.append([label])

    sample_array = np.array(window_list).astype(np.float32)
    
    label_array = np.array(label_list).astype(np.float32)

    data = torch.from_numpy(sample_array)
    label = torch.from_numpy(label_array)
    return data,label

def load_test_data(sequence_length = 50, stride = 1, dataset='s02'):
    dataset_path = 'N-CMAPSS/' + dataset + '/test'
    file_lists = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path)]
    data = np.load(file_lists[0])
    for index_file in file_lists:
        index_data = np.load(index_file)
        data = np.concatenate((data,index_data), axis=0)
    window_list = []
    label_list = []    
    num_samples = int((data.shape[0] - sequence_length)/stride) + 1

    for i in range(num_samples):
        window = data[i*stride:i*stride + sequence_length, 4:]  # each individual window
        label = data[i*stride + sequence_length-1, 0]
        window_list.append(window)
        label_list.append([label])

    sample_array = np.array(window_list).astype(np.float32) 
    label_array = np.array(label_list).astype(np.float32)

    data = torch.from_numpy(sample_array)
    label = torch.from_numpy(label_array)
    return data,label