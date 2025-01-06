import numpy as np
import glob
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
# 导入所需的库：numpy用于数值计算，glob用于文件路径匹配，scipy.io用于读取和写入数据文件，torch用于深度学习，torch.utils.data用于数据加载器。

def UT_HAR_dataset(root_dir):
    """
        读取UT_HAR数据集的csv文件，并将数据和标签存储到字典中返回
        参数:
        - root_dir: 数据集根目录
        返回值:
        - WiFi_data: 包含数据和标签的字典
        """
    # 获取数据文件和标签文件的路径列表
    data_list = glob.glob(root_dir+'/UT_HAR/data/*.csv')
    label_list = glob.glob(root_dir+'/UT_HAR/label/*.csv')
    # 创建一个空字典来存储数据和标签
    WiFi_data = {}
    # 遍历数据文件路径列表
    for data_dir in data_list:
        # 提取数据文件名
        data_name = data_dir.split('\\')[-1].split('.')[0]
        with open(data_dir, 'rb') as f:
            data = np.load(f)
            data = data.reshape(len(data),1,250,90)
            # 数据归一化
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
            # 将归一化后的数据存储到字典中
        WiFi_data[data_name] = torch.Tensor(data_norm)
    for label_dir in label_list:
        # 提取标签文件名
        label_name = label_dir.split('\\')[-1].split('.')[0]
        with open(label_dir, 'rb') as f:
            label = np.load(f)
        WiFi_data[label_name] = torch.Tensor(label)
    return WiFi_data

# 定义了一个名为UT_HAR_dataset的函数，它接受一个根目录root_dir作为参数。在函数内部，首先使用glob.glob函数获取所有匹配
# root_dir+'/UT_HAR/data/*.csv'和root_dir+'/UT_HAR/label/*.csv'模式的数据文件和标签文件的路径，并存储在data_list和label_list中。
#
# 然后，创建一个空字典WiFi_data用于存储数据。接下来，通过一个循环遍历data_list中的每个数据文件路径。在循环中，
# 首先从路径中提取数据文件的名称，并使用open函数打开数据文件。然后，使用np.load函数加载数据文件中的数据，并对数据进行重塑，
# 使其具有形状(len(data), 1, 250, 90)。接着，对数据进行归一化处理，将其缩放到0到1之间。最后，将处理后的数据转换为torch.Tensor类型，
# 并将其存储在WiFi_data字典中，以数据文件名称作为键。
#
# 接下来，使用类似的方式处理label_list中的标签文件，将标签数据加载、处理和存储到WiFi_data字典中。
#
# 最后，函数返回WiFi_data字典，其中包含了处理后的数据和标签
# dataset: /class_name/xx.mat
class CSI_Dataset(Dataset):
    """CSI dataset."""

    def __init__(self, root_dir, modal='CSIamp', transform=None, few_shot=False, k=5, single_trace=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            modal (CSIamp/CSIphase): CSI data modal
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.modal=modal
        self.transform = transform
        self.data_list = glob.glob(root_dir+'/*/*.mat')
        self.folder = glob.glob(root_dir+'/*/')
        self.category = {self.folder[i].split('\\')[-2]:i for i in range(len(self.folder))}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('\\')[-2]]
        x = sio.loadmat(sample_dir)[self.modal]
        
        # normalize
        x = (x - 42.3199)/4.9802
        
        # sampling: 2000 -> 500
        x = x[:,::4]
        x = x.reshape(3, 114, 500)
        
        if self.transform:
            x = self.transform(x)
        
        x = torch.FloatTensor(x)

        return x,y


