import torch
import torch.nn as nn
from torch.nn import functional as F
from models import common, swt_pytorch
from torch.autograd import Variable
from torch.nn import init

def make_model(args, parent=False):
    return Rainnet(args)

def get_residue(tensor, r_dim=1):  ## MAX-MIN  &&  MAX-AVG (gray) ##

    max_channel = torch.max(tensor, dim=r_dim, keepdim=True)
    min_channel = torch.min(tensor, dim=r_dim, keepdim=True)
    mean_channel = torch.mean(tensor, dim=r_dim, keepdim=True)

    res_channel1 = max_channel[0] - min_channel[0]
    res_channel2 = max_channel[0] - mean_channel[0]

    return res_channel1, res_channel2

def cate_residue(rcp1, rcp2):  ## MAX-MIN  &&  MAX-AVG (rgb) ##

    res_x1 = torch.cat((rcp1, rcp1, rcp1), dim=1)
    res_x2 = torch.cat((rcp2, rcp2, rcp2), dim=1)

    return res_x1, res_x2

def wave(t1, t2, t3):  ## wavelet transform ##

    t1_0, t1_1, t1_2, t1_3 = torch.chunk(t1[0], 4, dim=1)
    t2_0, t2_1, t2_2, t2_3 = torch.chunk(t2[0], 4, dim=1)
    t3_0, t3_1, t3_2, t3_3 = torch.chunk(t3[0], 4, dim=1)
    ll = torch.cat((t1_0, t2_0, t3_0), dim=1)
    lh = torch.cat((t1_1, t2_1, t3_1), dim=1)
    hl = torch.cat((t1_2, t2_2, t3_2), dim=1)
    hh = torch.cat((t1_3, t2_3, t3_3), dim=1)

    return ll, lh, hl, hh

def get_list(out):

    SWT_list = [out]

    return SWT_list

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.prelu(x)
        return x

class Inception(nn.Module):
    def __init__(self, in_channels, n3x3_reduce, n3x3, n5x5_reduce, n5x5, n7x7_reduce, n7x7):
        super(Inception, self).__init__()

        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, n3x3_reduce, kernel_size=1, stride=1, padding=0),
            ConvBlock(n3x3_reduce, n3x3, kernel_size=3, stride=1, padding=1))

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, n5x5_reduce, kernel_size=1, stride=1, padding=0),
            ConvBlock(n5x5_reduce, n5x5, kernel_size=5, stride=1, padding=2))

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, n7x7_reduce, kernel_size=1, stride=1, padding=0),
            ConvBlock(n7x7_reduce, n7x7, kernel_size=7, stride=1, padding=3))

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        return torch.cat([x1, x2, x3], dim=1)

class RMA(nn.Module):  ### ResMA Block ###
    def __init__(self, n_feats, nm='in', use_GPU=True):
        super(RMA, self).__init__()

        self.sep_ch = Inception(32, 16, 32, 16, 32, 16, 32)
        self.conv_change = nn.Sequential(
            nn.Conv2d(96, n_feats, 3, 1, 1),
            nn.PReLU()
        )
        self.se = common.SELayer(n_feats, 1)

    def forward(self, x):
        res = self.sep_ch(x)
        res = self.conv_change(res)
        res = self.se(res)
        res += x
        return res

class MARR(nn.Module):  ### MARR Block ###
    def __init__(self, n_feats, recurrent_iter=3, use_GPU=True):

        super(MARR, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv_rma = RMA(n_feats)  # ResMA Block

        self.conv_i = nn.Sequential(
            nn.Conv2d(n_feats + n_feats, n_feats, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            nn.Conv2d(n_feats + n_feats, n_feats, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(n_feats + n_feats, n_feats, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(n_feats + n_feats, n_feats, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True))

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        x_list = []

        for i in range(self.iteration):
            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h

            x = self.conv_rma(x)
            x = self.conv(x)
            x = x + input
            x_list.append(x)

        return x, h

class MARR_H(nn.Module):
    def __init__(self, n_feats, recurrent_iter=3, use_GPU=True):

        super(MARR_H, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv_rma = RMA(n_feats)

        self.conv_i = nn.Sequential(
            nn.Conv2d(n_feats + n_feats, n_feats, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            nn.Conv2d(n_feats + n_feats, n_feats, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(n_feats + n_feats, n_feats, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(n_feats + n_feats, n_feats, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True))

    def forward(self, input, state):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = state
        c = Variable(torch.zeros(batch_size, 32, row, col))


        if self.use_GPU:
            c = c.cuda()

        x_list = []

        for i in range(self.iteration):
            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h

            x = self.conv_rma(x)
            x = self.conv(x)
            x = x + input
            x_list.append(x)

        return x, h


class RCG_Fuse(nn.Module):  ### RCG -> Fusion Block ###    #  LL -> max-min  &&  max-avg

    def __init__(self, in_dim=3):
        super(RCG_Fuse, self).__init__()

        self.p_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.q_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.p1_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.q1_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.se = common.SELayer(channel=3, reduction=1)
        self.image_conv = nn.Sequential(
            nn.Conv2d(6, in_dim, 3, 1, 1, bias=True))
        self.conv_change = nn.Sequential(
            nn.Conv2d(in_dim, 32, 3, 1, 1),
            nn.PReLU()
        )

    def forward(self, rc1, rc2):
        a = self.p_conv(rc1)
        b = self.q_conv(rc2)
        energy = a * b
        se = self.se(energy)

        p_gamma = self.p1_conv(se)
        p_out = rc1 + p_gamma

        q_gamma = self.q1_conv(se)
        q_out = rc2 + q_gamma
        fuse_cat = torch.cat((p_out, q_out), dim=1)
        fuse_conv = self.image_conv(fuse_cat)  # 6->3
        fuse = self.conv_change(fuse_conv)  # 3->32

        return fuse

class CWFA_Fuse(nn.Module):  ### CWFA -> Fusion Block ###    #  guide  &&  wavelet subband

    def __init__(self, in_dim=32):
        super(CWFA_Fuse, self).__init__()

        self.conv_change = nn.Sequential(  # 3->32 밑 입력
            nn.Conv2d(3, in_dim, 3, 1, 1),
            nn.PReLU()
        )
        self.rcg_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.wave_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.rcg1_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.wave1_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.se = common.SELayer(channel=32, reduction=1)
        self.image_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1, bias=True))  # 64->32

    def forward(self, x, sub):
        sub = self.conv_change(sub)
        a = self.rcg_conv(x)
        b = self.wave_conv(sub)
        energy = a * b
        se = self.se(energy)

        rcg_gamma = self.rcg1_conv(se)
        rcg_out = x + rcg_gamma

        wave_gamma = self.wave1_conv(se)
        wave_out = sub + wave_gamma
        fuse_cat = torch.cat((rcg_out, wave_out), dim=1)
        fuse = self.image_conv(fuse_cat)

        return fuse

class CWFA_Fuse2(nn.Module):

    def __init__(self, in_dim=32):
        super(CWFA_Fuse2, self).__init__()

        self.rcg_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.wave_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.rcg1_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.wave1_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.se = common.SELayer(channel=32, reduction=1)
        self.image_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1, bias=True))  # 64->32

    def forward(self, x, sub):
        a = self.rcg_conv(x)
        b = self.wave_conv(sub)
        energy = a * b
        se = self.se(energy)

        rcg_gamma = self.rcg1_conv(se)
        rcg_out = x + rcg_gamma

        wave_gamma = self.wave1_conv(se)
        wave_out = sub + wave_gamma
        fuse_cat = torch.cat((rcg_out, wave_out), dim=1)
        fuse = self.image_conv(fuse_cat)

        return fuse

class RCG(nn.Module):  ### RCG Block ###
    def __init__(self, in_dim=32):

        super(RCG, self).__init__()

        self.fuse = RCG_Fuse()
        self.LSTM = MARR(in_dim)

    def forward(self, rc1, rc2):
        L_Fuse = self.fuse(rc1, rc2)
        L_FE, _ = self.LSTM(L_Fuse)  # L_Fuse Feature extraction

        return L_FE

class DFA(nn.Module):  ### Downward Feature Aggregation phase -> stage1 ###
    def __init__(self, n_feats):

        super(DFA, self).__init__()

        self.fuse_LL = CWFA_Fuse()
        self.fuse_LH = CWFA_Fuse()
        self.fuse_HL = CWFA_Fuse()
        self.fuse_HH = CWFA_Fuse()

        self.LSTM_LL = MARR(n_feats)
        self.LSTM_LH = MARR_H(n_feats)
        self.LSTM_HL = MARR_H(n_feats)
        self.LSTM_HH = MARR_H(n_feats)

    def forward(self, L1_FE, ll, lh, hl, hh):
        LL_Fuse = self.fuse_LL(L1_FE, ll)
        LH_Fuse = self.fuse_LH(L1_FE, lh)
        HL_Fuse = self.fuse_HL(L1_FE, hl)
        HH_Fuse = self.fuse_HH(L1_FE, hh)

        LL_, state = self.LSTM_LL(LL_Fuse)
        LH_, state_lh = self.LSTM_LH(LH_Fuse, state)
        HL_, state_hl = self.LSTM_HL(HL_Fuse, state_lh)
        HH_, _ = self.LSTM_HH(HH_Fuse, state_hl)


        return LL_, LH_, HL_, HH_

class DFA2(nn.Module):  ### Downward Feature Aggregation phase -> stage2 ###
    def __init__(self, n_feats):

        super(DFA2, self).__init__()

        self.fuse_LL2 = CWFA_Fuse()
        self.fuse_LH2 = CWFA_Fuse2()
        self.fuse_HL2 = CWFA_Fuse2()
        self.fuse_HH2 = CWFA_Fuse2()

        self.LSTM_LL = MARR(n_feats)
        self.LSTM_LH = MARR_H(n_feats)
        self.LSTM_HL = MARR_H(n_feats)
        self.LSTM_HH = MARR_H(n_feats)

    def forward(self, L2_FE, ll, lh, hl, hh):
        LL_Fuse2 = self.fuse_LL2(L2_FE, ll)
        LH_Fuse2 = self.fuse_LH2(L2_FE, lh)
        HL_Fuse2 = self.fuse_HL2(L2_FE, hl)
        HH_Fuse2 = self.fuse_HH2(L2_FE, hh)

        LL_, state2 = self.LSTM_LL(LL_Fuse2)  
        LH_, state_lh2 = self.LSTM_LH(LH_Fuse2, state2)
        HL_, state_hl2 = self.LSTM_HL(HL_Fuse2, state_lh2)
        HH_, _ = self.LSTM_HH(HH_Fuse2, state_hl2)


        return LL_, LH_, HL_, HH_

class UFA(nn.Module):  ### Upward Feature Aggregation phase -> stage1 ###
    def __init__(self, n_feats):

        super(UFA, self).__init__()

        self.conv_image_LL_initial = nn.Sequential(  # 64 -> 32
            nn.Conv2d(64, n_feats, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU()
        )
        self.conv_image_LH_initial = nn.Sequential(  # 64 -> 32
            nn.Conv2d(64, n_feats, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU()
        )
        self.conv_image_HL_initial = nn.Sequential(  # 64 -> 32
            nn.Conv2d(64, n_feats, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU()
        )
        self.conv_image_LL = nn.Sequential(  # 32 -> 3
            nn.Conv2d(n_feats, 3, kernel_size=3, stride=1, padding=1, bias=True))

        self.LSTM1 = MARR(n_feats)
        self.LSTM2 = MARR(n_feats)
        self.LSTM3 = MARR(n_feats)


    def forward(self, LL, LH, HL, HH):

        HL_cat = torch.cat((HL, HH), dim=1)
        HL_change = self.conv_image_HL_initial(HL_cat) # 64->32
        HL_final_stage1, _ = self.LSTM1(HL_change)
        LH_cat = torch.cat((LH, HL_final_stage1), dim=1)
        LH_change = self.conv_image_LH_initial(LH_cat)  # 64->32
        LH_final_stage1, _ = self.LSTM2(LH_change)
        LL_cat = torch.cat((LL, LH_final_stage1), dim=1)
        LL_change = self.conv_image_LL_initial(LL_cat)  # 64->32
        LL_final_stage1, _ = self.LSTM3(LL_change)

        LL_image = self.conv_image_LL(LL_final_stage1)  # 32->3


        return LL_image, LH_final_stage1, HL_final_stage1, HH

class UFA2(nn.Module):  ### Upward Feature Aggregation phase -> stage2 ###
    def __init__(self, n_feats):

        super(UFA2, self).__init__()

        self.conv_image_LL_initial = nn.Sequential(  # 64 -> 32
            nn.Conv2d(64, n_feats, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU()
        )
        self.conv_image_LH_initial = nn.Sequential(  # 64 -> 32
            nn.Conv2d(64, n_feats, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU()
        )
        self.conv_image_HL_initial = nn.Sequential(  # 64 -> 32
            nn.Conv2d(64, n_feats, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU()
        )

        self.LSTM2_1 = MARR(n_feats)
        self.LSTM2_2 = MARR(n_feats)
        self.LSTM2_3 = MARR(n_feats)

        self.conv_image_LL = nn.Sequential( # 32 -> 3
            nn.Conv2d(n_feats, 3, kernel_size=3, stride=1, padding=1, bias=True))
        self.conv_image_LH = nn.Sequential(  # 32 -> 3
            nn.Conv2d(n_feats, 3, kernel_size=3, stride=1, padding=1, bias=True))
        self.conv_image_HL = nn.Sequential(  # 32 -> 3
            nn.Conv2d(n_feats, 3, kernel_size=3, stride=1, padding=1, bias=True))
        self.conv_image_HH = nn.Sequential(  # 32 -> 3
            nn.Conv2d(n_feats, 3, kernel_size=3, stride=1, padding=1, bias=True))


    def forward(self, LL, LH, HL, HH):

        HL_cat = torch.cat((HL, HH), dim=1)
        HL_change = self.conv_image_HL_initial(HL_cat)  # 64->32
        HL_final_stage2, _ = self.LSTM2_1(HL_change)
        LH_cat = torch.cat((LH, HL_final_stage2), dim=1)
        LH_change = self.conv_image_LH_initial(LH_cat)  # 64->32
        LH_final_stage2, _ = self.LSTM2_2(LH_change)
        LL_cat = torch.cat((LL, LH_final_stage2), dim=1)
        LL_change = self.conv_image_LL_initial(LL_cat)  # 64->32
        LL_final_stage2, _ = self.LSTM2_3(LL_change)

        LL_image = self.conv_image_LL(LL_final_stage2)  # 32->3
        LH_image = self.conv_image_LH(LH_final_stage2)  # 32->3
        HL_image = self.conv_image_HL(HL_final_stage2)  # 32->3
        HH_image = self.conv_image_HH(HH)  # 32->3
        wave_cat = torch.cat((LL_image, LH_image, HL_image, HH_image), dim=1) #12 channel

        return wave_cat


class Rainnet(nn.Module):
    def __init__(self, args, use_GPU=True):
        super(Rainnet, self).__init__()
        n_feats = args.n_feats

        self.SWT = SWTForward(J=1, wave='haar').cuda()
        self.ISWT = SWTInverse(wave='haar').cuda()
        self.use_GPU = use_GPU

        self.RCG = RCG()
        self.RCG2 = RCG()
        self.DFA = DFA(n_feats)
        self.DFA2 = DFA2(n_feats)
        self.UFA = UFA(n_feats)
        self.UFA2 = UFA2(n_feats)

    def forward(self, x):

        r, g, b = torch.split(x, 1, dim=1)
        red = self.SWT(r)
        green = self.SWT(g)
        blue = self.SWT(b)

        x1_ll, x1_lh, x1_hl, x1_hh = wave(red, green, blue)  # wavelet transform

        #####stage 1#####

        res_1l, res_2l = get_residue(x1_ll)
        res_1lt, res_2lt = cate_residue(res_1l, res_2l)

        L1_FE = self.RCG(res_1lt, res_2lt)
        LL_, LH_, HL_, HH_ = self.DFA(L1_FE, x1_ll, x1_lh, x1_hl, x1_hh)
        LL_1, LH_1, HL_1, HH_1 = self.UFA(LL_, LH_, HL_, HH_)  # LL만 3channel

        #####stage 2#####

        res_1l_2, res_2l_2 = get_residue(LL_1)  # LL MAX - MIN
        res_1lt_2, res_2lt_2 = cate_residue(res_1l_2, res_2l_2)

        L2_FE = self.RCG2(res_1lt_2, res_2lt_2)
        LL_2, LH_2, HL_2, HH_2 = self.DFA2(L2_FE, LL_1, LH_1, HL_1, HH_1)
        x1_ = self.UFA2(LL_2, LH_2, HL_2, HH_2)

        x1 = get_list(x1_)

        out = self.ISWT(x1)

        return out


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
