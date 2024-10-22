import torch
import torch.nn as nn
import torch.nn.functional as F


class Hamburger(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(Hamburger, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, None))
        self.pool_w = nn.AdaptiveAvgPool3d((None, 1, None))
        self.pool_d = nn.AdaptiveAvgPool3d((None, None, 1))

        mip = max(8, inp // reduction)
        # print(mip)
        self.conv1 = nn.Conv3d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv3d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv3d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.gn1 = nn.GroupNorm(8, mip)
        self.gn2 = nn.GroupNorm(8, mip)
        self.gn3 = nn.GroupNorm(8, mip)
        # self.gn1 = nn.BatchNorm2d(64)
        self.act = nn.LeakyReLU(0.2)
        self.conv_h = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_d = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w, d = x.size()
        x_h = self.pool_h(x)
        # print(x_h.shape)
        x_w = self.pool_w(x).permute(0, 1, 3, 2, 4)
        # print(x_w.shape)
        x_d = self.pool_d(x).permute(0, 1, 4, 2, 3)
        # print(x_d.shape)
        y_hwd = torch.cat([x_h, x_w, x_d], dim=2)
        # y_hd = torch.cat([x_h, x_d], dim=2)
        # y_dw = torch.cat([x_d, x_w], dim=2)
        y_hwd = self.conv1(y_hwd)
        # y_hd = self.conv2(y_hd)
        # y_dw = self.conv3(y_dw)
        y_hwd = self.gn1(y_hwd)
        # y_hd = self.gn2(y_hd)
        # y_dw = self.gn3(y_dw)
        y_hwd = self.act(y_hwd)
        # y_hd = self.act(y_hd)
        # y_dw = self.act(y_dw)
        # print(y_hwd.shape)
        x_h, x_w, x_d = torch.split(y_hwd, [1, 1, 1], dim=2)
        # print(x_h.shape)
        # print(x_w.shape)
        # print(x_d.shape)

        x_w = x_w
        x_h = x_h.permute(0, 1, 3, 2, 4)
        x_d = x_d.permute(0, 1, 3, 4, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        a_d = self.conv_d(x_d).sigmoid()
        a_hw = a_w * a_h
        out = a_hw * a_d
        # print(out.shape)
        return out + x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(channels * self.expansion)
            )

        self.stride = stride

    def forward(self, x):
        shortcut = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += shortcut
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

        self.conv3 = nn.Conv3d(channels, channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(channels * self.expansion)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(channels * self.expansion)
            )

    def forward(self, x):
        shortcut = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += shortcut
        out = self.relu(out)

        return out


class ResNet3D_OLD(nn.Module):
    def __init__(self, block, layers, num_classes):
        super().__init__()
        self.in_channels = 64
        # self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)  # 第一个原来是3
        # self.bn1 = nn.BatchNorm3d(64)
        # self.relu = nn.ReLU()
        # self.max_pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)  # 第一个原来是3
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.hamburger = Hamburger(2048, 2048)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        # TODO sequence part
        # self.rnn = nn.GRU(  # if use nn.RNN(), it hardly learns
        #     input_size=22,
        #     hidden_size=64,  # rnn hidden unit
        #     num_layers=1,  # number of rnn layer
        #     batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        #     bidirectional=True
        # )
        # self.out = nn.Sequential(
        #     nn.Linear(128, 256),
        #     nn.ReLU()
        # )

        self.seq_fc = nn.Sequential(nn.Linear(50, 1024), nn.ReLU(), nn.Linear(1024, 256))
        self.fc = nn.Linear(512 * block.expansion + 256, num_classes)

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, channels, n_blocks, stride=1):
        assert n_blocks > 0, "number of blocks should be greater than zero"
        layers = []
        layers.append(block(self.in_channels, channels, stride))
        self.in_channels = channels * block.expansion
        for i in range(1, n_blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x, debug=False):
        voxel, seq = x
        out = self.conv1(voxel)
        out = self.bn1(out)
        out = self.relu(out)
        if debug:
            print("shape1:", out.shape)
        out = self.max_pool(out)
        if debug:
            print("shape2:", out.shape)

        out = self.layer1(out)
        if debug:
            print("shape3:", out.shape)
        out = self.layer2(out)
        if debug:
            print("shape4:", out.shape)
        out = self.layer3(out)
        if debug:
            print("shape5:", out.shape)
        out = self.layer4(out)
        # out = self.hamburger(out)
        if debug:
            print("shape6:", out.shape)
        out = self.avg_pool(out)
        if debug:
            print("shape7:", out.shape)

        out = out.view(out.size(0), -1)
        feature = out
        # gru
        # one_hot_list = []
        # for i in range(seq.shape[0]):
        #     one_hot_list.append(F.one_hot(seq[i, :].to(torch.int64), num_classes=22).unsqueeze(0))
        # one_hot_seq = torch.cat(one_hot_list, dim=0).float()
        #
        # r_out, _ = self.rnn(one_hot_seq.squeeze(1), None)  # None represents zero initial hidden state
        # seq = self.out(r_out[:, -1, :])

        # print(out.shape)
        seq = self.seq_fc(seq).squeeze(1)

        fusion = torch.cat((seq, feature), dim=1)
        out = self.fc(fusion)
        return out


class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes):
        super().__init__()
        self.in_channels = 64
        # self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)  # 第一个原来是3
        # self.bn1 = nn.BatchNorm3d(64)
        # self.relu = nn.ReLU()
        # self.max_pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)  # 第一个原来是3
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.hamburger = Hamburger(2048, 2048)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        # TODO sequence part
        # self.rnn = nn.GRU(  # if use nn.RNN(), it hardly learns
        #     input_size=22,
        #     hidden_size=64,  # rnn hidden unit
        #     num_layers=1,  # number of rnn layer
        #     batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        #     bidirectional=True
        # )
        # self.out = nn.Sequential(
        #     nn.Linear(128, 256),
        #     nn.ReLU()
        # )
        self.seq_fc = nn.Sequential(nn.Linear(50, 1024), nn.ReLU(), nn.Linear(1024, 256))

        # self.seq_fc = nn.Sequential(nn.Linear(50, 50, bias=False), nn.ReLU(), nn.Linear(50, 1024), nn.ReLU(), nn.Linear(1024, 256))
        self.fc = nn.Linear(512 * block.expansion + 256, num_classes)

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, channels, n_blocks, stride=1):
        assert n_blocks > 0, "number of blocks should be greater than zero"
        layers = []
        layers.append(block(self.in_channels, channels, stride))
        self.in_channels = channels * block.expansion
        for i in range(1, n_blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x, debug=False):
        voxel, seq = x
        out = self.conv1(voxel)
        out = self.bn1(out)
        out = self.relu(out)
        if debug:
            print("shape1:", out.shape)
        out = self.max_pool(out)
        if debug:
            print("shape2:", out.shape)

        out = self.layer1(out)
        if debug:
            print("shape3:", out.shape)
        out = self.layer2(out)
        if debug:
            print("shape4:", out.shape)
        out = self.layer3(out)
        if debug:
            print("shape5:", out.shape)
        out = self.layer4(out)
        # out = self.hamburger(out)
        if debug:
            print("shape6:", out.shape)
        out = self.avg_pool(out)
        if debug:
            print("shape7:", out.shape)

        out = out.view(out.size(0), -1)
        feature = out
        # gru
        # one_hot_list = []
        # for i in range(seq.shape[0]):
        #     one_hot_list.append(F.one_hot(seq[i, :].to(torch.int64), num_classes=22).unsqueeze(0))
        # one_hot_seq = torch.cat(one_hot_list, dim=0).float()
        #
        # r_out, _ = self.rnn(one_hot_seq.squeeze(1), None)  # None represents zero initial hidden state
        # seq = self.out(r_out[:, -1, :])

        # print(out.shape)
        seq = self.seq_fc(seq).squeeze(1)
        # print(seq.shape)
        # print(feature.shape)
        fusion = torch.cat((seq, feature), dim=1)
        out = self.fc(fusion)
        return out
        # out2 = nn.functional.softmax(out, dim=1)  # 后加的
        # return out2, feature


def resnet26(num_classes):
    return ResNet3D(Bottleneck, [1, 2, 4, 1], num_classes=num_classes)


def resnet50(num_classes):
    return ResNet3D(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def resnet101(num_classes):
    return ResNet3D(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def resnet152(num_classes):
    return ResNet3D(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)


class GRU(nn.Module):
    def __init__(self, input_dim, classes):
        super(GRU, self).__init__()

        self.rnn = nn.GRU(  # if use nn.RNN(), it hardly learns
            input_size=input_dim,
            hidden_size=64,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            bidirectional=True
        )

        self.out = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, classes)
        )

    def forward(self, x):
        seq = x

        one_hot_list = []
        for i in range(seq.shape[0]):
            one_hot_list.append(F.one_hot(seq[i, 0, :].to(torch.int64), num_classes=22).unsqueeze(0))
        one_hot_seq = torch.cat((one_hot_list), dim=0).float()
        r_out = self.rnn(one_hot_seq, None)[0]  # None represents zero initial hidden state
        out = self.out(r_out[:, -1, :])
        return out
    # def forward(self, x, seq_lengths):
    #     xx = torch.nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True)
    #     r_out, _ = self.rnn(xx, None)   # None represents zero initial hidden state
    #     u_out, lens = torch.nn.utils.rnn.pad_packed_sequence(r_out, batch_first=True)
    #     l_out = []
    #     for length, op in zip(seq_lengths,u_out) :
    #         l_out.append(op[length-1])
    #     out = torch.stack(l_out)
    #     out = self.out(out)
    #     return out


def gru(input_dim, classes):
    return GRU(input_dim, classes)


if __name__ == "__main__":
    model = resnet26(1)
    voxel = torch.zeros((4, 3, 64, 64, 64))
    h_in = torch.zeros((2, 2048, 2, 2, 2))
    h = Hamburger(2048, 2048)
    h(h_in)
    seq = torch.ones((4, 50))
    res = model.forward((voxel, seq), debug=True)
