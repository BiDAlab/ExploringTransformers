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
configs.enc_in = 128 #Encoder input
configs.d_model = 128 #Dim of model 512
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
configs.output_dim = 745 #number of users 
configs.data_length = 128 
configs.dimension = 6 
configs.lr = 0.001
configs.hlayers = 9
configs.hheads = 10 #(input/10) 100, 64, 32, 16, 8, 4, 2
configs.vlayers = 1
configs.vheads = 6 #(input/
configs.hlayers_rec = 1
configs.hlayers_pos = 2                                                                         #^#

##################  DATASET  ######################
training_dataset = torch.load('data/training_dataset_OUISIR.pt')
validation_dataset = torch.load('data/validation_dataset_OUISIR.pt')
testing_dataset = torch.load('data/testing_dataset_OUISIR.pt')
train_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
testing_dataloader = DataLoader(testing_dataset, batch_size=32, shuffle=False)


###########  NETWORK PARAMETERS  #################
data_length = 128       # number of signals per each gait cycle
dimension = 6           # number of channnels
user_number = 745

class ProposedTransformer(torch.nn.Module):
    def __init__(self, args):
        super(ProposedTransformer, self).__init__()
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.transformer_pre = HARAutoformerAutoformer(128, args.hlayers, args.hheads, 6)
        self.transformer_rec = HAR_BRTransformer(128, args.hlayers_rec, args.hheads, 6)
        self.transformer_post = HARAutoformerAutoformer(128, args.hlayers_pos, args.hheads, 6)
        self.args = args
        self.kernel_num = 512
        self.kernel_num_v = 256
        self.filter_sizes = [6, 6] #Reduced because of the input (6,128)
        self.filter_sizes_v = [10, 10] #[2,2]
        self.pos_encoding_h = Gaussian_Position(128, 6, args.K)
        self.pos_encoding_v = Gaussian_Position(6, 128, args.K)
        if args.vlayers == 0:
            self.v_transformer = None
            self.dense = torch.nn.Linear(128, 745)
        else:
            self.v_transformer = HARAutoformerAutoformer(6, args.vlayers, args.vheads, 2)
            self.dense = torch.nn.Linear(self.kernel_num * len(self.filter_sizes) + self.kernel_num_v * len(self.filter_sizes_v), 745)

        self.dense2 = torch.nn.Linear(self.kernel_num * len(self.filter_sizes), 745)
        self.dropout_rate = 0.5
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.encoders = []
        self.encoder_v = []
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                             torch.nn.Conv1d(in_channels=128,
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
            enc_ = F.elu(f_map)
            k_h = enc_.size()[-1]
            enc_ = F.max_pool1d(enc_, kernel_size=k_h)
            enc_ = enc_.squeeze(dim=-1)
            enc_outs.append(enc_)
        encoding = self.dropout(torch.cat(enc_outs, 1))
        q_re = F.elu(encoding)
        if self.v_transformer is not None:
            for encoder in self.encoder_v:
                f_map = encoder(v.transpose(-1, -2))
                enc_ = F.elu(f_map)
                k_h = enc_.size()[-1]
                enc_ = F.max_pool1d(enc_, kernel_size=k_h)
                enc_ = enc_.squeeze(dim=-1)
                enc_outs_v.append(enc_)
            encoding_v = self.dropout(torch.cat(enc_outs_v, 1))
            v_re = F.elu(encoding_v)
            q_re = torch.cat((q_re, v_re), dim=1)
        return q_re

    def forward(self, data):
        x = data.view(data.shape[0], 6, -1)
        x = self.pos_encoding_h(x)
        x = self.transformer_pre(x)
        x = self.transformer_rec(x)
        x = self.transformer_post(x)
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
    
TransformerModel = ProposedTransformer(configs) 
TransformerModel = TransformerModel.to(device)
############################################################################
# TRAINING
optimizer = torch.optim.Adam(TransformerModel.parameters(), lr=0.001)
loss_fn= torch.nn.CrossEntropyLoss()
batch_size=64

def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.
    for i, data in enumerate(train_dataloader):
        # Every data instance is an input + label pair
        inputs, labels = Variable(data["data"]).float().requires_grad_(), Variable(data["label"]).long()
        labels = labels-1
        labels = labels.data[:,0]
        labels = torch.squeeze(F.one_hot(labels, num_classes=user_number), 1)
        inputs, labels = inputs.to(device), labels.to(device)        
        # Zero your gradients for every batch!
        optimizer.zero_grad()        
        # Make predictions for this batch
        outputs = TransformerModel(inputs.float())
        loss = loss_fn(outputs.float(), labels.float())
        loss.backward()        
        # Adjust learning weights
        torch.nn.utils.clip_grad_norm_(TransformerModel.parameters(), 1.0)
        optimizer.step()        
        # Gather data and report
        running_loss += loss.item()
        labels = labels.cpu()
        outputs = outputs.cpu()
        pred = torch.max(outputs, 1)[1]
        labels_max = torch.max(labels, 1)[1]
        running_correct = (pred == labels_max).sum()
        tr_acc = running_correct.item()
        total_num  = len(labels_max)
        acc = tr_acc/total_num
        if i % batch_size == batch_size-1:
            last_loss = running_loss / batch_size # loss per batch
            running_loss = 0.
            
    return last_loss, acc

epoch_number = 0
EPOCHS = 50
best_vloss = 1_000_000.
best_vacc = 0.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))
    
    # Make sure gradient tracking is on, and do a pass over the data
    TransformerModel.train()
    avg_loss, acc = train_one_epoch(epoch_number)
    print("\nAccuracy in training data is", acc)

    TransformerModel.eval()
    vtr_acc = 0.
    vtotal_num = 0
    running_correct = 0
    running_vloss = 0.0
    running_vAUC = 0.0
    for i, vdata in enumerate(validation_dataloader):
        vinputs, vlabels = Variable(vdata["data"]).float().requires_grad_(), Variable(vdata["label"]).long()
        vlabels = vlabels-1
        vlabels = vlabels.data[:,0]
        vlabels = torch.squeeze(F.one_hot(vlabels, num_classes=user_number), 1)
        vinputs, vlabels = vinputs.to(device),  vlabels.to(device)
        voutputs = TransformerModel(vinputs.float())
        vloss = loss_fn(voutputs.float(), vlabels.float())
        running_vloss += vloss
        vlabels = vlabels.cpu()
        voutputs = voutputs.cpu()
        running_vloss += vloss
        pred = torch.max(voutputs, 1)[1]
        vlabels_max = torch.max(vlabels, 1)[1]
        running_correct = (pred == vlabels_max).sum()
        vtr_acc = running_correct.item()
        vtotal_num  = len(vlabels_max)
        vacc = vtr_acc/vtotal_num
    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    print("\nAccuracy in validation data is", vacc)
    if avg_vloss < best_vloss or epoch == EPOCHS-1:
        best_vloss = avg_vloss
    model_path = 'ProposedTransformer_OUISIR_{}'.format(epoch_number)
    torch.save(TransformerModel.state_dict(), model_path)  
    epoch_number += 1

