import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


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


class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()

        # 第一个3x3卷积层，用于特征提取
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        # 第二个3x3卷积层，用于进一步特征提取
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


# UT_HAR_ResNet类，整个网络的主类
class UT_HAR_ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes=7):
        super(UT_HAR_ResNet, self).__init__()
        # 数据预处理层
        self.reshape = nn.Sequential(
            nn.Conv2d(1, 3, 7, stride=(3, 1)),  # 输入: (batch_size, 1, H, W), 输出: (batch_size, 3, H/3, W)
            nn.ReLU(),
            nn.MaxPool2d(2),  # 输入: (batch_size, 3, H/3, W), 输出: (batch_size, 3, H/6, W/2)
            nn.Conv2d(3, 3, kernel_size=(10, 11), stride=1),  # 输入: (batch_size, 3, H/6, W/2), 输出: (batch_size, 3, H/6-10+1, W/2-11+1)
            nn.ReLU()
        )
        self.in_channels = 64
        # 初始卷积层，用于初步特征提取
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 输入: (batch_size, 3, H/6-10+1, W/2-11+1), 输出: (batch_size, 64, H/6-10+1/2+3, W/2-11+1/2+3)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 输入: (batch_size, 64, H/6-10+1/2+3, W/2-11+1/2+3), 输出: (batch_size, 64, H/6-10+1/2+3/2+1, W/2-11+1/2+3/2+1)
        # 构建网络的四个残差层
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)  # 输入: (batch_size, 64, H/6-10+1/2+3/2+1, W/2-11+1/2+3/2+1), 输出: (batch_size, 64, H/6-10+1/2+3/2+1, W/2-11+1/2+3/2+1)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)  # 输入: (batch_size, 64, H/6-10+1/2+3/2+1, W/2-11+1/2+3/2+1), 输出: (batch_size, 128, H/6-10+1/2+3/2+1/2, W/2-11+1/2+3/2+1/2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)  # 输入: (batch_size, 128, H/6-10+1/2+3/2+1/2, W/2-11+1/2+3/2+1/2), 输出: (batch_size, 256, H/6-10+1/2+3/2+1/2/2, W/2-11+1/2+3/2+1/2/2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)  # 输入: (batch_size, 256, H/6-10+1/2+3/2+1/2/2, W/2-11+1/2+3/2+1/2/2), 输出: (batch_size, 512, H/6-10+1/2+3/2+1/2/2/2, W/2-11+1/2+3/2+1/2/2/2)
        # 全局平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 输入: (batch_size, 512, H/6-10+1/2+3/2+1/2/2/2, W/2-11+1/2+3/2+1/2/2/2), 输出: (batch_size, 512, 1, 1)
        # 分类全连接层
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)  # 输入: (batch_size, 512), 输出: (batch_size, num_classes)

    def forward(self, x):
        # 数据预处理
        x = self.reshape(x)  # 输入: (batch_size, 1, H, W), 输出: (batch_size, 3, H/6-10+1, W/2-11+1) # 假设输入 x 的初始形状为 (batch_size, 1, 250, 90) # 输出形状: (batch_size, 3, 80, 35)

        # 初始卷积和最大池化
        x = self.relu(self.batch_norm1(self.conv1(x)))  # 输入: (batch_size, 3, H/6-10+1, W/2-11+1), 输出: (batch_size, 64, H/6-10+1/2+3, W/2-11+1/2+3)  # 输出形状: (batch_size, 64, 40, 18)
        x = self.max_pool(x)  # 输入: (batch_size, 64, H/6-10+1/2+3, W/2-11+1/2+3), 输出: (batch_size, 64, H/6-10+1/2+3/2+1, W/2-11+1/2+3/2+1) # 输出形状: (batch_size, 64, 20, 9)

        # 四个残差层
        x = self.layer1(x) # 输入: (batch_size, 64, H/6-10+1/2+3/2+1, W/2-11+1/2+3/2+1), 输出: (batch_size, 64, H/6-10+1/2+3/2+1, W/2-11+1/2+3/2+1)  # 输出形状: (batch_size, 64, 20, 9)
        x = self.layer2(x) # 输出形状: (batch_size, 128, 10, 5)
        x = self.layer3(x) # 输出形状: (batch_size, 256, 5, 3)
        x = self.layer4(x)# 输出形状: (batch_size, 512, 3, 2)
        # 全局平均池化
        x = self.avgpool(x) # 输出形状: (batch_size, 512, 1, 1)
        # 展平
        x = x.reshape(x.shape[0], -1)# 输出形状: (batch_size, 512)
        # 分类
        x = self.fc(x)# 输出形状: (batch_size, num_classes)
        
        return x
        # 构建残差层的方法
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        # 判断是否需要下采样
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            # 构建第一个残差块
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        # 构建剩余的残差块
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            # 返回构建好的残差层
        return nn.Sequential(*layers)
# 创建不同深度的ResNet网络的函数
def UT_HAR_ResNet18():
    return UT_HAR_ResNet(Block, [2,2,2,2])
def UT_HAR_ResNet18_scSE():
    return UT_HAR_ResNet(Block_scSE, [2,2,2,2])
def UT_HAR_ResNet18_dual():
    return UT_HAR_ResNet(Block_dual, [2,2,2,2])
def UT_HAR_ResNet18_sd():
    return UT_HAR_ResNet(Block_sd, [2,2,2,2])



class UT_HAR_GRU_scSE(nn.Module):
    def __init__(self, hidden_dim=64):
        super(UT_HAR_GRU_scSE, self).__init__()
        # GRU 层
        self.gru = nn.GRU(90, hidden_dim, num_layers=1)

        # scSE 模块
        self.scse = scSE(hidden_dim)

        # 全连接层
        self.fc = nn.Linear(hidden_dim, 7)

    def forward(self, x):
        # 输入 x 的形状: (batch_size, 1, 250, 90)
        x = x.view(-1, 250, 90)
        x = x.permute(1, 0, 2)  # 变换为 (250, batch_size, 90)

        # 通过 GRU 层
        _, ht = self.gru(x)

        # 将GRU输出转换为特征图
        # ht[-1] 的形状: (batch_size, hidden_dim)
        # 转换为特征图 (batch_size, hidden_dim, 1, 1)
        feature_map = ht[-1].unsqueeze(dim=2).unsqueeze(dim=3)

        #print(feature_map.shape) #[64, 64, 1, 1]
        # 通过 scSE 模块
        feature_map = self.scse(feature_map)

        # 将scSE输出转换回序列形式
        # feature_map 的形状: (batch_size, hidden_dim, 1, 1)
        # 转换为 (batch_size, hidden_dim)
        feature_map = feature_map.squeeze(dim=2).squeeze(dim=2)

        # 经过 fc 层
        outputs = self.fc(feature_map)

        return outputs

class UT_HAR_ResNet_G(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes=7):
        super(UT_HAR_ResNet_G, self).__init__()
        # 保留时间维度，只对空间维度进行卷积和池化
        self.reshape = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=(7, 1), stride=(3, 1)),  # 只在高度方向上应用卷积，保持时间维度
            nn.ReLU(),
            nn.Conv2d(3, 32, kernel_size=(11, 1), stride=(1, 1)),  # 同样只在高度方向上应用卷积，保持时间维度
            nn.ReLU()
        )

        self.in_channels = 32

        # 初始卷积层，用于初步特征提取，只对空间维度进行卷积
        self.conv1 = nn.Conv2d(32, 32, kernel_size=(7, 1), stride=(2, 1),
                               padding=(3, 0), bias=False)  # 输入: (batch_size, 64, 17, 250)，输出: (batch_size, 64, 9, 250)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1),
                                     padding=(1, 0))  # 输入: (batch_size, 64, 9, 250)，输出: (batch_size, 64, 5, 250)

        # 构建网络的四个残差层，只对空间维度进行卷积
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=32, stride=1)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=64, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=128, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=256, stride=2)

    def forward(self, x):
        # 数据预处理，交换时间维度和特征维度
        x = x.permute(0, 1, 3, 2)  # (batch, 1, 250, 90) -> (batch, 1, 90, 250)

        # 数据预处理，保留时间维度
        x = self.reshape(x)  # 输出: (batch, 64, 17, 250)

        # 初始卷积和最大池化，保留时间维度
        x = self.relu(self.batch_norm1(self.conv1(x)))  # 输出: (batch, 64, 9, 250)
        x = self.max_pool(x)  # 输出: (batch, 64, 5, 250)

        # 四个残差层，保留时间维度
        x = self.layer1(x)  # 输出: (batch, 64, 5, 250)
        x = self.layer2(x)  # 输出: (batch, 128, 3, 250)
        x = self.layer3(x)  # 输出: (batch, 256, 2, 250)
        x = self.layer4(x)  # 输出: (batch, 512, 1, 250)

        # 保留时间维度，展平空间维度

        x = x.permute(0, 3, 1, 2)  # (batch, time_steps, channels, width)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (batch, time_steps, channels * width)


        return x  # 返回形状 (batch, 250, 512)

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        # 如果需要下采样或通道数变化，则添加下采样层
        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=(1, 1), stride=(stride, 1)),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        # 添加第一个残差块
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        # 添加剩余的残差块
        for i in range(1, blocks):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)

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
class UT_HAR_GRU_A(nn.Module):
    def __init__(self,hidden_dim=64):
        super(UT_HAR_GRU_A,self).__init__()
        # GRU 层
        # 输入维度: 90 (假设输入特征的数量)
        # 隐藏层维度: hidden_dim (默认为 64)
        # 层数: 1 (单层 GRU)
        self.gru = nn.GRU(90,hidden_dim,num_layers=1)
        self.attention = Attention2(hidden_dim)
        # 全连接层
        self.fc = nn.Linear(hidden_dim, 7)
        # 全连接层
        # 输入维度: hidden_dim (GRU 输出的隐藏层维度)
        # 输出维度: 7 (假设类别数为 7)
    def forward(self,x):

        # 输入 x 的形状: (batch_size, seq_length, input_size)
        # 假设输入 x 的初始形状为 (batch_size,1, 250, 90)
        # x = x.view(-1, 250, 90)
        # 经过 view 操作后 x 的形状变为 (batch_size, 250, 90)
        # 注意: 这一步在实际执行中是不必要的，因为输入 x 已经具有正确的形状。
        x = x.view(-1,250,90)


        # 通过 GRU 层
        # 输入 x 的形状: (250, batch_size, 90)
        # 返回值: (output, ht)
        # - output: (seq_length, batch_size, hidden_dim)
        # - ht: (num_layers * num_directions, batch_size, hidden_dim)
        output, _ = self.gru(x)

        attention_output = self.attention(output)
        outputs = self.fc(attention_output)
        # ht[-1]: 最后一层的隐藏状态
        # ht[-1] 的形状: (batch_size, hidden_dim)
        # 经过 fc 层
        # 输入 ht[-1] 的形状: (batch_size, hidden_dim)
        # 输出的形状: (batch_size, 7)
        return outputs

class UT_HAR_GRU_R(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=64, num_classes=7):
        super(UT_HAR_GRU_R, self).__init__()
        # GRU 层
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        # 注意力机制
        self.attention = Attention2(hidden_dim)
        # 全连接层
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        gru_output, _ = self.gru(x) #b,250,64

        attention_output = self.attention(gru_output)
        outputs = self.fc(attention_output)
        return outputs


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

def UT_HAR_ResNet_G_sd():
    return UT_HAR_ResNet_G(Block_sd_G, [2,2,2,2])

class UT_HAR_GRU_Attention(nn.Module):
    def __init__(self,  hidden_dim=64, num_classes=7):
        super(UT_HAR_GRU_Attention, self).__init__()
        self.resnet = UT_HAR_ResNet_G_sd()
        self.gru = UT_HAR_GRU_R(input_dim=256, hidden_dim=hidden_dim, num_classes=num_classes)

    def forward(self, x):
        x= self.resnet(x)
        x = self.gru(x)
        return x



class UT_HAR_GRU(nn.Module):
    def __init__(self,hidden_dim=64):
        super(UT_HAR_GRU,self).__init__()
        # GRU 层
        # 输入维度: 90 (假设输入特征的数量)
        # 隐藏层维度: hidden_dim (默认为 64)
        # 层数: 1 (单层 GRU)
        self.gru = nn.GRU(90,hidden_dim,num_layers=1)
        # 全连接层
        # 输入维度: hidden_dim (GRU 输出的隐藏层维度)
        # 输出维度: 7 (假设类别数为 7)
        self.fc = nn.Linear(hidden_dim,7)
    def forward(self,x):

        # 输入 x 的形状: (batch_size, seq_length, input_size)
        # 假设输入 x 的初始形状为 (batch_size,1, 250, 90)
        # x = x.view(-1, 250, 90)
        # 经过 view 操作后 x 的形状变为 (batch_size, 250, 90)
        # 注意: 这一步在实际执行中是不必要的，因为输入 x 已经具有正确的形状。
        x = x.view(-1,250,90)

        # 重排 x 的维度，使其满足 GRU 的输入要求
        # 输入 GRU 的形状: (seq_length, batch_size, input_size)
        # x = x.permute(1, 0, 2)
        # 经过 permute 操作后 x 的形状变为 (250, batch_size, 90)
        x = x.permute(1,0,2)

        # 通过 GRU 层
        # 输入 x 的形状: (250, batch_size, 90)
        # 返回值: (output, ht)
        # - output: (seq_length, batch_size, hidden_dim)
        # - ht: (num_layers * num_directions, batch_size, hidden_dim)
        _, ht = self.gru(x)
        # ht[-1]: 最后一层的隐藏状态
        # ht[-1] 的形状: (batch_size, hidden_dim)
        # 经过 fc 层
        # 输入 ht[-1] 的形状: (batch_size, hidden_dim)
        # 输出的形状: (batch_size, 7)
        outputs = self.fc(ht[-1])
        return outputs



class UT_HAR_CNN_GRU(nn.Module):
    def __init__(self):
        super(UT_HAR_CNN_GRU,self).__init__()
        self.encoder = nn.Sequential(
            #input size: (250,90)
            nn.Conv1d(250,250,12,3),
            nn.ReLU(True),
            nn.Conv1d(250,250,5,2),
            nn.ReLU(True),
            nn.Conv1d(250,250,5,1)
            # 250 x 8
        )
        self.gru = nn.GRU(8,128,num_layers=1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128,7),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        # batch x 1 x 250 x 90
        x = x.view(-1,250,90)
        x = self.encoder(x)
        # batch x 250 x 8
        x = x.permute(1,0,2)
        # 250 x batch x 8
        _, ht = self.gru(x)
        outputs = self.classifier(ht[-1])
        return outputs


# PatchEmbedding类，用于将输入图像分割成多个固定大小的patch，并嵌入到高维空间中
class PatchEmbedding_scSE(nn.Module):
    def __init__(self, in_channels=1, patch_size_w = 40, patch_size_h = 16, emb_size =128 , img_size = 240*16):
        super().__init__()
        self.patch_size_w = patch_size_w
        self.patch_size_h = patch_size_h
        # 卷积层用于将输入图像分割成patch，并映射到高维空间
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=(patch_size_w, patch_size_h), stride=(patch_size_w, patch_size_h)),
            Rearrange('b e (h) (w) -> b (h w) e'),  # 重排数据，以便适应Transformer的输入格式
        )
        # CLS token，用于汇总整个序列的信息
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # 位置编码，用于区分不同patch的位置信息
        self.position = nn.Parameter(torch.randn(int(img_size/emb_size) + 1, emb_size))
        # 添加scSE模块
        self.scSE = scSE(emb_size)

    def forward(self, x):
        b, _, _, _ = x.shape
        # 将输入图像分割成patch，并映射到高维空间
        x = self.projection(x)
        # print(x.shape)
        # 使用 permute() 方法改变维度顺序
        x = x.permute(0, 2, 1).unsqueeze(-1)
        # print(x.shape)
        # 应用scSE模块
        x = self.scSE(x)
        x = x.squeeze(-1).permute(0, 2, 1)

        # 复制CLS token，使其与batch size匹配
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # 将CLS token添加到patch序列的开始
        x = torch.cat([cls_tokens, x], dim=1)
        # 添加位置编码
        x += self.position #[64, 26, 900]

        return x




class PatchEmbedding(nn.Module):
    def __init__(self, in_channels = 1, patch_size_w = 40, patch_size_h = 16, emb_size =128 , img_size = 240*16):
        self.patch_size_w = patch_size_w
        self.patch_size_h = patch_size_h
        # 卷积层用于将输入图像分割成patch，并映射到高维空间
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size = (patch_size_w, patch_size_h), stride = (patch_size_w, patch_size_h)),
            Rearrange('b e (h) (w) -> b (h w) e'),  # 重排数据，以便适应Transformer的输入格式
        )
        # CLS token，用于汇总整个序列的信息
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size))
        # 位置编码，用于区分不同patch的位置信息
        self.position = nn.Parameter(torch.randn(int(img_size/emb_size) + 1, emb_size))

    def forward(self, x):
        b, _, _, _ = x.shape
        # 将输入图像分割成patch，并映射到高维空间
        x = self.projection(x)
        # 复制CLS token，使其与batch size匹配
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # 将CLS token添加到patch序列的开始
        x = torch.cat([cls_tokens, x], dim=1)
        # print(x.shape)
        # print(self.position.shape)
        # 添加位置编码
        x += self.position
        # print(x.shape)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size=640, num_heads=4, dropout=0.0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
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
    def __init__(self, emb_size, expansion=2, drop_p=0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size=640, drop_p=0., forward_expansion=2, forward_drop_p=0., **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth=2, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=256, n_classes=7):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )


class UT_HAR_ViT(nn.Sequential):
    def __init__(self, in_channels=1, patch_size_w=40, patch_size_h=16, emb_size=128, img_size=240*16, depth=2,
                 n_classes=7, **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size_w, patch_size_h, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )

class UT_HAR_ViT_scSE(nn.Sequential):
    def __init__(self,
                in_channels = 1,
                patch_size_w = 40,
                patch_size_h = 16,
                emb_size = 128,
                img_size = 240*16,
                depth = 2,
                n_classes = 7,
                **kwargs):
        super().__init__(
            PatchEmbedding_scSE(in_channels, patch_size_w, patch_size_h, emb_size, img_size),# 假设输入为 (batch_size, 1, 250, 90)  (batch_size, num_patches + 1, emb_size) 输出为 (batch_size, 5 + 1, 900)，其中 5 是 (250*90) / (50*18) 的整数部分
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),#假设输入为 (batch_size, 6, 900) 输出为 (batch_size, 6, 900)
            ClassificationHead(emb_size, n_classes) # 输出为 (batch_size, 7)
        )



##*****************************************CNN+BiLSTM
class CNN_Module(nn.Module):
    def __init__(self):
        super(CNN_Module, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 输入通道为1，输出通道为64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 空间维度减半
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 空间维度再减半
        )

    def forward(self, x):
        return self.cnn(x)  # 输出 [batch_size, 128, 62, 22] 的特征
# 双向 LSTM 模块
class BiLSTM_Module(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiLSTM_Module, self).__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)

    def forward(self, x):
        output, _ = self.bilstm(x)
        return output  # 返回 [batch_size, seq_len, hidden_size*2]
# 整体模型
class UT_HAR_CNN_BiLSTM(nn.Module):
    def __init__(self, num_classes=7):
        super(UT_HAR_CNN_BiLSTM, self).__init__()
        self.cnn_module = CNN_Module()
        self.bilstm_module1 = BiLSTM_Module(input_size=128*22, hidden_size=128, num_layers=1)  # 注意 input_size 变为 128*22
        self.bilstm_module2 = BiLSTM_Module(input_size=128*2, hidden_size=128, num_layers=1)
        self.fc = nn.Linear(128*2, num_classes)  # 双向LSTM输出为 hidden_size*2

    def forward(self, x):
        batch_size = x.size(0)

        # CNN 提取空间特征
        cnn_features = self.cnn_module(x)  # 输出 [batch_size, 128, 62, 22]
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