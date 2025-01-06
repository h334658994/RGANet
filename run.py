import numpy as np
import torch
import torch.nn as nn
import argparse
import os
from util import load_data_n_model
# from torch.utils.tensorboard import SummaryWriter  # 后期增加
import csv   # 后期增加



# def train(model, tensor_loader, num_epochs, learning_rate, criterion, device):
def train(model, tensor_loader, num_epochs, learning_rate, criterion, device,name):  # 后期增加
    """
        训练模型函数
        参数：
        - model: 模型
        - tensor_loader: 数据加载器
        - num_epochs: 训练轮数
        - learning_rate: 学习率
        - criterion: 损失函数
        - device: 设备
        - name: 模型名称
        """
    model = model.to(device) # 将模型移动到设备上
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Adam优化器


    training_data = []  # 后期增加 记录训练过程中的准确率和损失值
    for epoch in range(num_epochs):
        model.train() # 设置模型为训练模式
        epoch_loss = 0
        epoch_accuracy = 0
        for data in tensor_loader:
            inputs, labels = data
            # print(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)# 将标签转换为长整型

            optimizer.zero_grad()# 梯度清零
            outputs = model(inputs) # 前向传播
            outputs = outputs.to(device)
            outputs = outputs.type(torch.FloatTensor)
            loss = criterion(outputs, labels)# 计算损失值
            loss.backward()  # 求反向传播导数
            optimizer.step()  # 自动参数更新

            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs, dim=1).to(device)# 预测类别
            epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0) # 计算准确率
        epoch_loss = epoch_loss / len(tensor_loader.dataset)# 计算平均损失值
        epoch_accuracy = epoch_accuracy / len(tensor_loader) # 计算平均准确率
        print('Epoch:{}, Accuracy:{:.4f},Loss:{:.9f}'.format(epoch + 1, float(epoch_accuracy), float(epoch_loss)))

        training_data.append([epoch + 1,float(epoch_accuracy), float(epoch_loss)])  # 后期增加# 将准确率和损失值添加到训练数据列表中


    return

# def test(model, tensor_loader, criterion, device):
def test(model, tensor_loader, criterion, device,tname,num_classes):
    model.eval()  #将模型设置为评估模式，这会关闭模型中的dropout和batch normalization层。
    test_acc = 0
    test_loss = 0

    for data in tensor_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels=labels.to(device)
        labels = labels.type(torch.LongTensor)
        with torch.no_grad():
            outputs = model(inputs)
            outputs = outputs.type(torch.FloatTensor)
            outputs.to(device)

        loss = criterion(outputs, labels)

        predict_y = torch.argmax(outputs, dim=1).to(device)
        accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
        test_acc += accuracy
        test_loss += loss.item() * inputs.size(0)

    test_acc = test_acc / len(tensor_loader)
    test_loss = test_loss / len(tensor_loader.dataset)
    print("validation accuracy:{:.4f}, loss:{:.5f}".format(float(test_acc), float(test_loss)))
    return


def main():
    root = './'
    parser = argparse.ArgumentParser('WiFi Imaging Benchmark')
    parser.add_argument('--dataset', choices=['UT_HAR_data',  'NTU-Fi_HAR'])
    parser.add_argument('--model',
                        choices=[ 'ResNet18','ResNet18_scSE','ResNet18_dual','ResNet18_sd',  'GRU','GRU_A','GRU_scSE',
                                 'CNN+GRU', 'ViT', 'ViT_scSE', 'GRU_Attention','CNN_BiLSTM'])
    args = parser.parse_args()
    # dataset_name = "UT_HAR_data"
    # model_name="MLP"
    # train_loader, test_loader, model, train_epoch = load_data_n_model(args.dataset, args.model, root)

    train_loader, test_loader, model, train_epoch,num_classes= load_data_n_model(args.dataset, args.model, root)

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    name=args.dataset+"_"+args.model #后期增加
    filename = os.path.join('model', name)
   
    train(
        model=model,
        tensor_loader=train_loader,
        num_epochs=train_epoch,
        learning_rate=1e-3,
        criterion=criterion,
        device=device,
        name=name #后期增加
    )
  
    
    
    test(
        model=model,
        tensor_loader=test_loader,
        criterion=criterion,
        device=device,
        tname=name,      #后期增加
        num_classes=num_classes #后期增加
    )


    return


if __name__ == "__main__":
    main()
