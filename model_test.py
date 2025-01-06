import numpy as np
import torch
import torch.nn as nn
import argparse
import seaborn as sns
import os
from util import load_data_n_model
# from torch.utils.tensorboard import SummaryWriter  # 后期增加
import csv   # 后期增加
import torch.nn.functional as F




def test(model, tensor_loader, criterion, device,tname,num_classes):
    model.eval()  #将模型设置为评估模式，这会关闭模型中的dropout和batch normalization层。
    test_acc = 0
    test_loss = 0

    for data in tensor_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = labels.type(torch.LongTensor)


        with torch.no_grad():

            outputs = model(inputs)
       
            outputs = outputs.type(torch.FloatTensor)
            outputs.to(device)

        # loss = criterion(src, labels)
        # print(loss)
        loss = criterion(outputs, labels)
        # print(loss)
        predict_y = torch.argmax(outputs, dim=1).to(device)
        # predict_src1 = torch.argmax(src, dim=1).to(device)
        # print(predict_y.equal(predict_src1))


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
                        choices=['ResNet18','ResNet18_sd','ResNet18_scSE','ResNet18_dual', 'GRU','GRU_Attention','GRU_scSE', 
                                 'CNN+GRU', 'ViT','ViT_scSE','CNN_BiLSTM'])
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
    model.load_state_dict(torch.load(f'{filename}_Model.pth'))
    # model.eval()


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