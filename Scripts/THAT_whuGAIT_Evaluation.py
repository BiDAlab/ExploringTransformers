import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from GaitDataset import *
import argparse
from Models_Transformers import Transformer_THAT, PositionalEncoding, AutoformerAutoformer, HAR_BRTransformer, InformerAutoformer, Gaussian_Position, HARTransformerAutoformer, HARInformerAutoformer, HARAutoformerAutoformer, HARTransformer, TransformerAutoformer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

############  TRAINING SETTINGS  #################
configs = argparse.ArgumentParser()
# configs.model = Transformer_model #from CNNModel, LSTMModel, CNN_LSTMModel, Transformer, c, Autoformer
configs.output_attention = False #'whether to output attention in encoder'
configs.enc_in = 80 #Encoder input
configs.d_model = 80 #Dim of model 512
configs.embed = 'timeF' #time features encoding, options:[timeF, fixed, learned]
configs.freq = 'm' #d_model?  freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
configs.factor = 0.5 #attention factor - inside self attention
configs.n_heads = 10 #vheads en prob
configs.d_ff = 2048 #dimension of fcn
configs.activation ='gelu'
configs.e_layers = 5 #number of encoder layers
configs.distil = False #whether to use distilling in encoder, using this argument means not using distilling
configs.dropout = 0.5
configs.moving_avg = 3 #Autoformer window size of moving average from height
configs.layer_dim = 3  # LSTM ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
configs.K = 20 #number of Gaussian distributions
configs.output_dim = 118 #number of users 
configs.data_length = 80 
configs.dimension = 6 
configs.lr = 0.001
configs.hlayers = 5
configs.hheads = 8 #(input/10) 80, 64, 32, 16, 8, 4, 2
configs.vlayers = 1
configs.vheads = 2 #(input/                                                                #^#

##################  DATASET  ######################
testing_dataset = torch.load('testing_dataset_whuGAIT.pt')
testing_dataloader = DataLoader(testing_dataset, batch_size=512, shuffle=False)

###########  NETWORK PARAMETERS  #################
data_length = 80       # number of signals per each gait cycle
dimension = 6           # number of channnels
user_number = 118


class THAT(torch.nn.Module):
    def __init__(self, args):
        super(THAT, self).__init__()
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.transformer = HARTransformer(80, args.hlayers, args.hheads, 6)
        self.args = args
        self.kernel_num = 128
        self.kernel_num_v = 16
        self.filter_sizes = [6, 6] #Reduced because of the input (6,80)
        self.filter_sizes_v = [2, 2] #[2,2]
        self.pos_encoding_h = Gaussian_Position(80, 6, args.K)
        self.pos_encoding_v = Gaussian_Position(6, 80, args.K)
        if args.vlayers == 0:
            self.v_transformer = None
            self.dense = torch.nn.Linear(80, 118)
        else:
            self.v_transformer = HARTransformer(6, args.vlayers, args.vheads, 2)
            self.dense = torch.nn.Linear(self.kernel_num * len(self.filter_sizes) + self.kernel_num_v * len(self.filter_sizes_v), 118)

        self.dense2 = torch.nn.Linear(self.kernel_num * len(self.filter_sizes), 118)
        self.dropout_rate = 0.5
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.encoders = []
        self.encoder_v = []
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                             torch.nn.Conv1d(in_channels=80,
                                       out_channels=self.kernel_num,
                                       kernel_size=filter_size).to('cuda')
                             )
            self.encoders.append(self.__getattr__(enc_attr_name))
        for i, filter_size in enumerate(self.filter_sizes_v):
            enc_attr_name_v = "encoder_v_%d" % i
            self.__setattr__(enc_attr_name_v,
                             torch.nn.Conv1d(in_channels=6,
                                       out_channels=self.kernel_num_v,
                                       kernel_size=filter_size).to('cuda')
                             )
            self.encoder_v.append(self.__getattr__(enc_attr_name_v))

    def _aggregate(self, o, v=None):
        enc_outs = []
        enc_outs_v = []
        for encoder in self.encoders:
            f_map = encoder(o.transpose(-1, -2))
            enc_ = F.relu(f_map)
            k_h = enc_.size()[-1]
            enc_ = F.max_pool1d(enc_, kernel_size=k_h)
            enc_ = enc_.squeeze(dim=-1)
            enc_outs.append(enc_)
        encoding = self.dropout(torch.cat(enc_outs, 1))
        q_re = F.relu(encoding)
        if self.v_transformer is not None:
            for encoder in self.encoder_v:
                f_map = encoder(v.transpose(-1, -2))
                enc_ = F.relu(f_map)
                k_h = enc_.size()[-1]
                enc_ = F.max_pool1d(enc_, kernel_size=k_h)
                enc_ = enc_.squeeze(dim=-1)
                enc_outs_v.append(enc_)
            encoding_v = self.dropout(torch.cat(enc_outs_v, 1))
            v_re = F.relu(encoding_v)
            q_re = torch.cat((q_re, v_re), dim=1)
        return q_re

    def forward(self, data):
        x = data.view(data.shape[0], 6, -1)
        x = self.pos_encoding_h(x)
        x = self.transformer(x)

        if self.v_transformer is not None:
            y = data.view(data.shape[0], 6, -1)
            y = y.transpose(-1, -2) 
            y = self.pos_encoding_v(y)
            y = self.v_transformer(y)
            re = self._aggregate(x, y)
            predict = self.softmax(self.dense(re))
        else:
            re = self._aggregate(x)
            predict = self.softmax(self.dense2(re))

        return predict    
 
    
TransformerModel = THAT(configs) 
TransformerModel = TransformerModel.to(device)
############################################################################
# TESTING
TransformerModel.load_state_dict(torch.load('THAT_whuGAIT'))
correct_Transformer = 0
TransformerModel.eval()
for batch_idx, data in enumerate(testing_dataloader, 0):
    data, target = Variable(data["data"]).float(), Variable(data["label"]).long()
    data = data.to(device)
    target = target.squeeze()
    target = target.to(device)
    # run model
    if target.dim() != 0:
        with torch.no_grad():
            output_Transformer = TransformerModel(data)
            pred_Transformer = output_Transformer.data.max(1, keepdim=True)[1]
            correct_Transformer += pred_Transformer.eq(target.data.view_as(pred_Transformer)).cpu().sum()
            

print('Accuracy THAT Transformer: {}/{} ({:.3f}%)'.format(
    correct_Transformer, len(testing_dataloader.dataset),
    float(100. * correct_Transformer) / len(testing_dataloader.dataset)))

