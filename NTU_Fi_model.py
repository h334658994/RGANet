import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class DualConv_G(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DualConv_G, self).__init__()

        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=(stride, 1), padding=padding, groups=in_channels, bias=bias)
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DualConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DualConv, self).__init__()

        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x






class cSE(nn.Module):
    def __init__(self, in_channel):
        super(cSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        y = self.relu(y)
        y = nn.functional.interpolate(y, size=(x.size(2), x.size(3)), mode='nearest')
        return x * y.expand_as(x)

class sSE(nn.Module):
    def __init__(self, in_channel):
        super(sSE, self).__init__()
        self.Conv1x1 = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, x):
        y = self.Conv1x1(x)
        y = self.norm(y)
        return x * y

class scSE(nn.Module):
    def __init__(self, in_channel):
        super(scSE, self).__init__()
        self.cSE = cSE(in_channel)
        self.sSE = sSE(in_channel)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return torch.max(U_cse, U_sse)

class NTU_Fi_MLP(nn.Module):
    def __init__(self, num_classes):
        super(NTU_Fi_MLP,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3*114*500,1024),
            nn.ReLU(),
            nn.Linear(1024,128),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(128,num_classes)
    def forward(self,x):
        x = x.view(-1,3*114*500)
        x = self.fc(x)
        x = self.classifier(x)
        return x
    

class NTU_Fi_LeNet(nn.Module):
    def __init__(self, num_classes):
        super(NTU_Fi_LeNet,self).__init__()
        self.encoder = nn.Sequential(
            #input size: (3,114,500)
            nn.Conv2d(3,32,(15,23),stride=9),
            nn.ReLU(True),
            nn.Conv2d(32,64,3,stride=(1,3)),
            nn.ReLU(True),
            nn.Conv2d(64,96,(7,3),stride=(1,3)),
            nn.ReLU(True),
        )
        self.fc = nn.Sequential(
            nn.Linear(96*4*6,128),
            nn.ReLU(),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        x = self.encoder(x)
        x = x.view(-1,96*4*6)
        out = self.fc(x)
        return out



class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)
        return x

class Block_scSE(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block_scSE, self).__init__()
        # 第一个3x3卷积层，用于特征提取
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        # 第二个3x3卷积层，用于进一步特征提取
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        # 添加scSE模块
        self.scSE = scSE(out_channels)
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))
        #print(x.shape) #[64, 128, 4, 4] [64, 64, 8, 8]...
        x = self.scSE(x)  # 应用scSE模块
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)
        return x



class Block_dual(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block_dual, self).__init__()

        # 使用DualConv模块替换原有的卷积层
        self.conv1 = DualConv(in_channels, out_channels, stride=stride)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        # 使用DualConv模块替换原有的卷积层
        self.conv2 = DualConv(out_channels, out_channels, stride=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.batch_norm1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        x += identity
        x = self.relu(x)

        return x
class Block_sd(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block_sd, self).__init__()
        # 第一个3x3卷积层，用于特征提取
        self.conv1 = DualConv(in_channels, out_channels, stride=stride)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        # 第二个3x3卷积层，用于进一步特征提取
        self.conv2 = DualConv(out_channels, out_channels, stride=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        # 添加scSE模块
        self.scSE = scSE(out_channels)
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))
        #print(x.shape) #[64, 128, 4, 4] [64, 64, 8, 8]...
        x = self.scSE(x)  # 应用scSE模块
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)
        return x
class Block_sd_G(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block_sd_G, self).__init__()
        # 第一个3x3卷积层，用于特征提取
        self.conv1 = DualConv_G(in_channels, out_channels, stride=stride)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        # 第二个3x3卷积层，用于进一步特征提取
        self.conv2 = DualConv_G(out_channels, out_channels, stride=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        # 添加scSE模块
        self.scSE = scSE(out_channels)
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))
        #print(x.shape) #[64, 128, 4, 4] [64, 64, 8, 8]...
        x = self.scSE(x)  # 应用scSE模块
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)
        return x
        
class NTU_Fi_ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes):
        super(NTU_Fi_ResNet, self).__init__()
        self.reshape = nn.Sequential(
            nn.Conv2d(3,3,(15,23),stride=(3,9)),
            nn.ReLU(),
            nn.Conv2d(3,3,kernel_size=(3,23),stride=1),
            nn.ReLU()
        )
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*ResBlock.expansion, num_classes)
        
    def forward(self, x):
        x = self.reshape(x)
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)
    
def NTU_Fi_ResNet18(num_classes):
    return NTU_Fi_ResNet(Block, [2,2,2,2], num_classes = num_classes)
def NTU_Fi_ResNet18_scSE(num_classes):
    return NTU_Fi_ResNet(Block_scSE, [2,2,2,2], num_classes = num_classes)
def NTU_Fi_ResNet18_dual(num_classes):
    return NTU_Fi_ResNet(Block_dual, [2,2,2,2], num_classes = num_classes)
def NTU_Fi_ResNet18_sd(num_classes):
    return NTU_Fi_ResNet(Block_sd, [2,2,2,2], num_classes = num_classes)



class NTU_Fi_ResNet_G(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes):
        super(NTU_Fi_ResNet_G, self).__init__()

        # 调整reshape部分，以更好地保留时间信息
        self.reshape = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(15, 1), stride=(3, 1)),  # 只在高度上应用卷积，保持时间维度
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 1), stride=1),  # 同样只在高度上应用卷积
            nn.ReLU()
        )

        self.in_channels = 32

        # 卷积层，这里不使用max_pool，或者使用stride=1的max_pool
        self.conv1 = nn.Conv2d(32, 32, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0), bias=False)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))  # 仅在高度上进行池化

        # 构建ResNet的四个残差层
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=32, stride=1)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=64, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=128, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=256, stride=2)

        # 如果你想在最终分类前保留时间维度，不要在这里使用全局平均池化
        # self.avgpool = nn.AdaptiveAvgPool2d((1, None))  # 适应性池化，仅在高度上进行池化



    def forward(self, x):

        x = self.reshape(x)  # (batch, 64, height, time)

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)  # (batch, 64, reduced_height, time)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)  # (batch, 512*expansion, very_small_height, time)torch.Size([64, 512, 1, 500])



        # 如果你希望在此处仍然保留时间信息，可以省略avgpool
        # x = self.avgpool(x)

        # 将空间维度展平，但保持时间维度
        x = x.permute(0, 3, 1, 2)  # (batch, time, 512*expansion, very_small_height)torch.Size([64, 500, 512, 1])


        x = x.reshape(x.shape[0], x.shape[1], -1)  # (batch, time, 512*expansion*very_small_height)torch.Size([64, 500, 512])



        return x  # 返回形状 (batch, time, feature_dim)

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        # 如果需要下采样，创建下采样的卷积层
        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=(1, 1), stride=(stride, 1)),
                # 只在高度维度上应用 stride
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for i in range(1, blocks):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)



class NTU_Fi_GRU_R(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=64, num_classes=7):
        super(NTU_Fi_GRU_R, self).__init__()
        # GRU 层
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        # 注意力机制
        self.attention = Attention2(hidden_dim)
        # 全连接层
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        gru_output, _ = self.gru(x)

        attention_output = self.attention(gru_output)
        outputs = self.fc(attention_output)
        return outputs

class Attention2(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention2, self).__init__()
        # 定义权重矩阵 W 和偏置 b
        self.W = nn.Parameter(torch.randn(hidden_dim, 1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x):
        # 确保输入形状为 (batch_size, seq_len, hidden_dim)
        assert x.dim() == 3, "Input tensor must have 3 dimensions: (batch_size, seq_len, hidden_dim)"

        # 计算注意力分数 e = tanh(X · W + b)
        e = torch.tanh(torch.matmul(x, self.W) + self.b)  # 形状变为 (batch_size, seq_len, 1)
        e = e.squeeze(2)  # 去掉最后一个维度，形状变为 (batch_size, seq_len)

        # 计算注意力权重 a = softmax(e, axis=1)
        a = F.softmax(e, dim=1)  # 形状保持为 (batch_size, seq_len)

        # 将 a 的形状调整为 (batch_size, seq_len, 1)，以便与 x 进行逐元素乘法
        a = a.unsqueeze(2)  # 形状变为 (batch_size, seq_len, 1)

        # 计算加权和 output = ∑(a · x)
        output = torch.sum(a * x, dim=1)  # 形状变为 (batch_size, hidden_dim)

        return output

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Parameter(torch.randn(hidden_dim))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 确保输入形状为 (batch_size, seq_len, hidden_dim)
        assert x.dim() == 3, "Input tensor must have 3 dimensions: (batch_size, seq_len, hidden_dim)"


        # 计算注意力分数 e = v^T · tanh(W1 * x + W2 * x)
        e = torch.tanh(self.W1(x) + self.W2(x))
        e = torch.matmul(e, self.v)  # 形状变为 (batch_size, seq_len)


        # 计算注意力权重 a = softmax(e, axis=1)
        a = self.softmax(e)  # 形状变为 (batch_size, seq_len)


        # 将 a 的形状调整为 (batch_size, seq_len, 1)，以便与 x 进行逐元素乘法
        a = a.unsqueeze(2)  # 形状变为 (batch_size, seq_len, 1)


        # 计算加权和 output = ∑(a · x)
        output = torch.sum(a * x, dim=1)  # 形状变为 (batch_size, hidden_dim)


        return output

def NTU_Fi_ResNet_G_sd(num_classes):
    return NTU_Fi_ResNet_G(Block_sd_G, [2,2,2,2],num_classes = num_classes)

class NTU_Fi_GRU_Attention(nn.Module):
    def __init__(self,  num_classes):
        super(NTU_Fi_GRU_Attention, self).__init__()
        self.resnet = NTU_Fi_ResNet_G_sd(num_classes)
        self.gru = NTU_Fi_GRU_R(input_dim=256, hidden_dim=64, num_classes=num_classes)

    def forward(self, x):
        x= self.resnet(x)  # (batch_size, 250, 512*90)
        x = self.gru(x)
        return x


class NTU_Fi_GRU(nn.Module):
    def __init__(self,num_classes):
        super(NTU_Fi_GRU,self).__init__()
        self.gru = nn.GRU(342,64,num_layers=1)
        self.fc = nn.Linear(64,num_classes)
    def forward(self,x):
        x = x.view(-1,342,500)
        x = x.permute(2,0,1)
        _, ht = self.gru(x)
        outputs = self.fc(ht[-1])
        return outputs
    



class NTU_Fi_CNN_GRU(nn.Module):
    def __init__(self,num_classes):
        super(NTU_Fi_CNN_GRU,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1,16,12,6),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16,32,7,3),
            nn.ReLU(),
        )
        self.mean = nn.AvgPool1d(32)
        self.gru = nn.GRU(8,128,num_layers=1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128,num_classes),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        batch_size = len(x)
        # batch x 3 x 114 x 500
        x = x.view(batch_size,3*114,500)
        x = x.permute(0,2,1)
        # batch x 500 x 342
        x = x.reshape(batch_size*500,1, 3*114)
        # (batch x 500) x 1 x 342
        x = self.encoder(x)
        # (batch x 500) x 32 x 8
        x = x.permute(0,2,1)
        x = self.mean(x)
        x = x.reshape(batch_size, 500, 8)
        # batch x 500 x 8
        x = x.permute(1,0,2)
        # 500 x batch x 8
        _, ht = self.gru(x)
        outputs = self.classifier(ht[-1])
        return outputs
##*****************************************CNN+BiLSTM
class CNN_Module(nn.Module):
    def __init__(self):
        super(CNN_Module, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 输入通道为3，输出通道减少至32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 空间维度减半
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 空间维度再减半
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 再次减小空间维度
        )

    def forward(self, x):
        return self.cnn(x)
# 双向 LSTM 模块
class BiLSTM_Module(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiLSTM_Module, self).__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)

    def forward(self, x):
        output, _ = self.bilstm(x)
        return output  # 返回 [batch_size, seq_len, hidden_size*2]
# 整体模型
class NTU_Fi_CNN_BiLSTM(nn.Module):
    def __init__(self, num_classes=7):
        super(NTU_Fi_CNN_BiLSTM, self).__init__()
        self.cnn_module = CNN_Module()
        self.bilstm_module1 = BiLSTM_Module(input_size=128*14, hidden_size=128, num_layers=1)
        self.bilstm_module2 = BiLSTM_Module(input_size=128*2, hidden_size=128, num_layers=1)
        self.fc = nn.Linear(128*2, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x=x.permute(0, 1, 3, 2)

        # CNN 提取空间特征
        cnn_features = self.cnn_module(x)  # 输出 [batch_size, 128, 62, 22]
        print(cnn_features.shape)
        cnn_features = cnn_features.view(batch_size, 62, -1)  # 展平为 [batch_size, 62, 128*22]

        # Bi-LSTM 层1
        lstm_out1 = self.bilstm_module1(cnn_features)  # 输出 [batch_size, 62, 256]

        # Bi-LSTM 层2
        lstm_out2 = self.bilstm_module2(lstm_out1)  # 输出 [batch_size, 62, 256]

        # 最后时刻的输出用于分类
        final_output = lstm_out2[:, -1, :]  # 取最后一个时间步的输出 [batch_size, 256]

        # 全连接层输出
        output = self.fc(final_output)  # 输出 [batch_size, num_classes]
        return F.log_softmax(output, dim=1)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels = 1, patch_size_w = 9, patch_size_h = 25, emb_size = 9*25, img_size = 342*500):
        self.patch_size_w = patch_size_w
        self.patch_size_h = patch_size_h
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size = (patch_size_w, patch_size_h), stride = (patch_size_w, patch_size_h)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size))
        self.position = nn.Parameter(torch.randn(int(img_size/emb_size) + 1, emb_size))
    
    def forward(self, x):
        x = x.view(-1,1,342,500)
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.position
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size = 225, num_heads = 5, dropout = 0.0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size*3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
    
    def forward(self, x, mask = None):
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion = 4, drop_p = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
        
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size = 225,
                 drop_p = 0.5,
                 forward_expansion = 4,
                 forward_drop_p = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))
        
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth = 1, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])
        
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, num_classes):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, num_classes))
        
class NTU_Fi_ViT(nn.Sequential):
    def __init__(self,     
                in_channels = 1,
                patch_size_w = 9,
                patch_size_h = 25,
                emb_size = 225,
                img_size = 342*500,
                depth = 1,
                *,
                num_classes,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size_w, patch_size_h, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, num_classes)
        )
