from dataset import *
from UT_HAR_model import *
from NTU_Fi_model import *
import torch


def load_data_n_model(dataset_name, model_name, root):
    # 定义数据集的类别数
    classes = {'UT_HAR_data':7,'NTU-Fi-HumanID':14,'NTU-Fi_HAR':6,'Widar':22}

    # 根据数据集名字选择对应的数据集类和模型类
    if dataset_name == 'UT_HAR_data':
        num_classes = classes['UT_HAR_data']
        print('using dataset: UT-HAR DATA')
        # 加载UT-HAR数据集
        data = UT_HAR_dataset(root)
        # 将训练集数据和标签封装成一个TensorDataset对象
        train_set = torch.utils.data.TensorDataset(data['X_train'],data['y_train'])
        # 将验证集和测试集数据和标签合并，并封装成一个TensorDataset对象
        test_set = torch.utils.data.TensorDataset(torch.cat((data['X_val'],data['X_test']),0),torch.cat((data['y_val'],data['y_test']),0))
        # 创建训练集和测试集的数据加载器
        train_loader = torch.utils.data.DataLoader(train_set,batch_size=64,shuffle=True, drop_last=True) # shuffle 是否打乱顺序 drop_last=True
        test_loader = torch.utils.data.DataLoader(test_set,batch_size=256,shuffle=False)
        # 根据模型名字选择对应的模型类和训练轮数_scSE
        if model_name == 'ResNet18':
            print("using model: ResNet18")
            model = UT_HAR_ResNet18()
            train_epoch = 200 #70
        elif model_name == 'ResNet18_scSE':
            print("using model: ResNet18_scSE")
            model = UT_HAR_ResNet18_scSE()
            train_epoch = 200  # 70

        elif model_name == 'ResNet18_dual':
            print("using model: ResNet18_dual")
            model = UT_HAR_ResNet18_dual()
            train_epoch = 200  # 70
        elif model_name == 'ResNet18_sd':
            print("using model: ResNet18_sd")
            model = UT_HAR_ResNet18_sd()
            train_epoch = 200  # 70
        
        elif model_name == 'GRU':
            print("using model: GRU")
            model = UT_HAR_GRU()
            train_epoch = 200
        elif model_name == 'GRU_Attention':
            print("using model: GRU_Attention")
            model = UT_HAR_GRU_Attention()
            train_epoch = 200
        elif model_name == 'GRU_scSE':
            print("using model: GRU_scSE")
            model = UT_HAR_GRU_scSE()
            train_epoch = 200
        
        elif model_name == 'GRU_A':
            print("using model: GRU_A")
            model = UT_HAR_GRU_A()
            train_epoch = 200
        
        elif model_name == 'CNN+GRU':
            print("using model: CNN+GRU")
            model = UT_HAR_CNN_GRU()
            train_epoch = 200 #20
        elif model_name == 'ViT':
            print("using model: ViT")
            model = UT_HAR_ViT()
            train_epoch = 200 #100,'CNN+BiLSTM'
        elif model_name == 'CNN_BiLSTM':
            print("using model: CNN_BiLSTM")
            model = UT_HAR_CNN_BiLSTM()
            train_epoch = 200

        elif model_name == 'ViT_scSE':
            print("using model: ViT_scSE")
            model = UT_HAR_ViT_scSE()
            train_epoch = 200 #100
       
        return train_loader, test_loader, model, train_epoch,num_classes
    
    
    
    
    elif dataset_name == 'NTU-Fi_HAR':
        print('using dataset: NTU-Fi_HAR')
        num_classes = classes['NTU-Fi_HAR']
        train_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi_HAR/train_amp/'), batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi_HAR/test_amp/'), batch_size=64, shuffle=False)
        if model_name == 'ResNet18':
            print("using model: ResNet18")
            model = NTU_Fi_ResNet18(num_classes)
            train_epoch = 30
        elif model_name == 'ResNet18_dual':
            print("using model: ResNet18_dual")
            model = NTU_Fi_ResNet18_dual(num_classes)
            train_epoch = 30
        elif model_name == 'ResNet18_scSE':
            print("using model: ResNet18_scSE")
            model = NTU_Fi_ResNet18_scSE(num_classes)
            train_epoch = 30
        elif model_name == 'ResNet18_sd':
            print("using model: ResNet18_sd")
            model = NTU_Fi_ResNet18_sd(num_classes)
            train_epoch = 30
        elif model_name == 'GRU':
            print("using model: GRU")
            model = NTU_Fi_GRU(num_classes)
            train_epoch = 30 #20
        elif model_name == 'GRU_Attention':
            print("using model: GRU_Attention")
            model = NTU_Fi_GRU_Attention(num_classes)
            train_epoch = 30
        elif model_name == 'CNN+GRU':
            print("using model: CNN+GRU")
            model = NTU_Fi_CNN_GRU(num_classes)
            train_epoch = 100 #20
        elif model_name == 'ViT':
            print("using model: ViT")
            model = NTU_Fi_ViT(num_classes=num_classes)
            train_epoch = 30
        elif model_name == 'CNN_BiLSTM':
            print("using model: CNN_BiLSTM")
            model = NTU_Fi_CNN_BiLSTM()
            train_epoch = 30
        return train_loader, test_loader, model, train_epoch,num_classes

