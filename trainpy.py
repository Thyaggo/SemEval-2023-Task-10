# %%
import os
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import lightning as L
import torchmetrics
from torchmetrics.functional.classification import binary_accuracy
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from collections import Counter
import pickle
import re
import copy
import time
import math

# Scoring
from sklearn.metrics import classification_report, f1_score
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device => ",device, ' torch ', torch.__version__)
torch.device(device)

# hyper parameters
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

#@title Hyper Parameters { display-mode: "both" }

EPOCHS             = 20
MAX_NO_OF_SPEAKERS = 8
MAX_DIALOGUE_LEN   = 33
MAX_SEQUENCE_LEN   = 24
original_labels    = ['abuse', 'adoration', 'annoyance', 'awkwardness', 'benefit', 'boredom', 'calmness', 'challenge', 'cheer', 'confusion', 'curiosity', 'desire', 'excitement', 'guilt', 'horror', 'humour', 'impressed', 'loss', 'nervousness', 'nostalgia', 'pain', 'relief', 'satisfaction', 'scold', 'shock', 'sympathy', 'threat']
train_count        = [31, 190, 1051, 880, 220, 78, 752, 214, 534, 486, 545, 180, 867, 216, 280, 153, 257, 351, 398, 65, 36, 173, 136, 94, 372, 209, 263]

EMOTIONS           = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
ALPHA_TENSOR       = torch.Tensor([0.1110, 0.0271, 0.0268, 0.1745, 0.4715, 0.0684, 0.1206]).to(device)

# DataLoader Hyperparamaters
BATCH_SIZE = 32

# Module 1 hyperparamaters(speaker_specific_emotion_sequence) : GRU n-n
input_size_1  = 7
hidden_size_1 = 10 
num_layers_1  = 2 
output_size_1 = 10


# Module 2 hyperparamaters(utterance_context) : Transformer Enc
input_size_2 = 768
n_head_2     = 4
dm_ff_2      = 2048
dp_2         = 0.2
num_layers_2 = 4 
act_fn_2     = 'relu'

# Module 3 hyperparamaters(speaker_context) : Transformer Enc
input_size_3 = 8
n_head_3     = 4
dm_ff_3      = 2048
dp_3         = 0.2
num_layers_3 = 4 
act_fn_3     = 'relu'

# Module 4 hyperparamaters(global_emotion_sequence) : GRU
input_size_4  = 1
hidden_size_4 = 10 
num_layers_4  = 2 
output_size_4 = 10

# Module 5 hyperparamaters(valence) : Transformer Enc
input_size_5 = 3
n_head_5     = 1
dm_ff_5      = 2048
dp_5         = 0.2
num_layers_5 = 4 
act_fn_5     = 'relu'

# Module 5 hyperparamaters(valence) : GRU
#input_size_5  = 3
#hidden_size_5 = 10
#num_layers_5  = 2
#output_size_5 = 10


# Final Model Hyperparamerters:
fc1_out = 800
fc2_out = 650
fc3_out = 500
fc4_out = 350
fc5_out = 200
fc6_out = 50
fc7_out = len(EMOTIONS)

# LSTM
input_size_6  = fc1_out + input_size_5 + 1
hidden_size_6 = 7
num_layers_6  = 3
output_size_6 = 7


#LSTM Parameters
#input_size_lstm = output_size_1 + fc1_out + output_size_4 + output_size_6   # Tamaño de entrada

learning_rate = 0.001

# %%
#Pandas Train Dataset after pre-processing
with open('train_df.pkl', 'rb') as f:
    train_df = pickle.load(f)

# %%
#Pandas Test Dataset after pre-processing
with open('test_df.pkl', 'rb') as f:
    test_df = pickle.load(f)

# %%
class SemEvalDataset(Dataset):
    """
        Class Dataset: Data
    """
    def __init__(self, data):
        self.data = data
        self.len = len(self.data)
        print(list(train_df.columns))
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        dict_x = {}
        dict_x['speaker'] = torch.tensor(self.data['speakers'][index], dtype=torch.float32)
        dict_x['triggers'] =  torch.tensor(self.data['triggers'][index], dtype=torch.float32)
        dict_x['sentence_embeddings'] = torch.tensor(self.data['sentence_embeddings'][index], dtype=torch.float32)
        dict_x['VAD'] = torch.tensor(self.data['VAD'][index], dtype=torch.float32)
        dict_x['VAD_mask'] = torch.tensor(self.data['VAD_mask'][index], dtype=torch.float32)

        dict_y = {}
        dict_y['emotion'] = torch.tensor(self.data['emotions'][index], dtype=torch.float32)

        return dict_x, dict_y

# %%
class SemEvalDatasetTest(Dataset):
    def __init__(self, data):
        self.data = data
        self.len = len(self.data)
        print(list(train_df.columns))
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        dict_x = {}
        dict_x['speaker'] = torch.tensor(self.data['speakers'][index], dtype=torch.float32)
        dict_x['triggers'] =  torch.tensor(self.data['triggers'][index], dtype=torch.float32)
        dict_x['sentence_embeddings'] = torch.tensor(self.data['sentence_embeddings'][index], dtype=torch.float32)
        dict_x['VAD'] = torch.tensor(self.data['VAD'][index], dtype=torch.float32)
        dict_x['VAD_mask'] = torch.tensor(self.data['VAD_mask'][index], dtype=torch.float32)

        return dict_x

# %%
dataset = SemEvalDataset(train_df)
test = SemEvalDatasetTest(test_df)

# %%
from torch.utils.data import random_split
total_size = len(dataset)
train_ratio = 0.8
val_ratio = 0.2

train_size = int(total_size * train_ratio)
val_size = int(total_size * val_ratio)

# Dividir el conjunto de datos
train_data, val_data = random_split(dataset, [train_size, val_size])

# %%
train_loader  = DataLoader(dataset = train_data, batch_size = BATCH_SIZE, num_workers=3)#, shuffle=True, collate_fn= MELDCollate())
val_loader    = DataLoader(dataset = val_data, batch_size = BATCH_SIZE, num_workers=3)#, shuffle=True, collate_fn= MELDCollate())

# %%
#desired_batch_index = 4
#for i, batch in enumerate(train_loader):
# if i == desired_batch_index:
#     # 'batch' contendrá el batch en el índice especificado
#     print(f"Batch {i}:")
#     bat = batch
#     break

# %%
#idx = torch.nonzero(bat[0]['VAD_mask'].logical_not())[-1]
#bat[0]['VAD_mask'].T
#print(bat[0]['VAD_mask'].unsqueeze(-1).shape, '\n\n', bat[1]['emotion'].shape)
#test = torch.cat((bat[0]['VAD_mask'].unsqueeze(-1), bat[1]['emotion']), -1)
#test.argmax(-1)
#torch.hstack((test2[:,:idx+1]+1,test2[:,idx+1:]))

# %%
class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)  # 1000^(2i/dim_model)
        
        # Reshape division_term to have the same number of rows as pos_encoding
        #division_term = division_term.view(1, -1)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding", pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:, :token_embedding.size(1), :]).requires_grad_(False)


# %%
class Module6GRU(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, output_size):
        super(Module6GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Since there are maximum of 8 speakers in a dialogue, so we decided to make 8 GRUs one for each speaker.
        self.gru_list= []
        for id in range(MAX_NO_OF_SPEAKERS):
            self.gru_list.append(nn.GRU(input_size, hidden_size, num_layers, batch_first = True))
        self.gru_modules = nn.ModuleList(self.gru_list)

    def valence_specific(self, valence, speaker):
        speaker = speaker.unique(dim=0, return_inverse=True)[1]
        # Asegúrate de que el tensor de padding esté en el mismo dispositivo que 'speaker' y 'valence'.
        padding_tensor = torch.zeros(valence.size(1), device=speaker.device)
        # Ahora utiliza el tensor de padding que está en el dispositivo correcto.
        return [torch.where(speaker.unsqueeze(1) == i, valence[:speaker.size(0)], padding_tensor) for i in speaker.unique()], speaker.unique()

    def applyGRU(self, speaker_valence, sp_idx , seq_len):
        speaker_output = torch.zeros(seq_len, self.output_size, device = device)
        for sp_idx, valence in zip(sp_idx, speaker_valence):
            # Verificar si hay alguna entrada para este hablante
            if valence.nonzero().size(0) == 0:
                continue

            if sp_idx >= 8:
                # Manejar el error o ajustar sp_idx aquí
                sp_idx = 7
            # Asegúrate de que valence tenga al menos dos dimensiones

            # Inicializar h0 como un tensor 2D
            h0 = torch.zeros(self.num_layers, self.hidden_size, device = device)  # Ahora h0 es 2D
            out, _ = self.gru_modules[sp_idx](valence, h0)

            # Rellenar speaker_output con la salida correspondiente
            for uid, output in enumerate(out.squeeze(0)):
                speaker_output[uid] = output

        return speaker_output


    def forward(self, x, speakers):
        batch_size = x.size(0)
        seq_len    = speakers.size(1)
        outputs = []
        for i in range(batch_size):
            speaker_specific, sp_idx = self.valence_specific(x[i], speakers[i])
            out = self.applyGRU(speaker_specific, sp_idx ,seq_len)
            outputs.append(out)
        
        final_output = torch.cat([outputs[i].unsqueeze(2) for i in range(len(outputs))], 2).permute(2,0,1)
        
        return final_output

# %%
class Module5TransformerEnc(nn.Module):
    # S, N, E : (seq_len, batch_size, input/embedding_size)
    def __init__(self, input_size, n_head, dim_ff, dp, num_layers, act_fn = 'relu'):
        super(Module5TransformerEnc, self).__init__()
        self.input_size = input_size
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = input_size, nhead = n_head, dim_feedforward = dim_ff, dropout=dp, activation=act_fn)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        
        self.positional_encoder = PositionalEncoding(
            dim_model=input_size, dropout_p=dp, max_len=MAX_SEQUENCE_LEN
        )
        

    def forward(self, x, x_mask):
        # x shape: seq_len, batch_size, input_size
        x = self.positional_encoder(x)
        x = x.permute(1,0,2)
        # Since batch_first is not a parameter in trasformer so the input must be S, N, E
        
        out = self.encoder(x, src_key_padding_mask=x_mask)
        # out shape : (S, N, E)
        return out

# %%
class Module4GRU(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, output_size):
        super(Module4GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first = True)
        self.fc  = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device = device)
        out, _ = self.gru(x.unsqueeze(-1), h0)
        
        # shape of out :  (N, seq_len, hidden_size)     (torch.Size([10, 33, 8])) 
        # shape of hn  :  (num_layers, N, hidden_size)     (torch.Size([2, 10, 8]))
        # shape of hn  :  (N, num_layers, hidden_size) and then flatten it to (N, num_layers*hiddem_size) 3D to 2D
        output = self.fc(out)
        # shape of output : [N, output_size]

        return output

# %%
class Module3TransformerEnc(nn.Module):
    # S, N, E : (seq_len, batch_size, input/embedding_size)
    def __init__(self, input_size, n_head, dim_ff, dp, num_layers, act_fn = 'relu'):
        super(Module3TransformerEnc, self).__init__()
        self.input_size = input_size
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = input_size, nhead = n_head, dim_feedforward = dim_ff, dropout=dp, activation=act_fn)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.positional_encoder = PositionalEncoding(
            dim_model=input_size, dropout_p=dp, max_len=MAX_SEQUENCE_LEN
        )
        

    def forward(self, x, x_mask):
        # x shape: seq_len, batch_size, input_size
        x = self.positional_encoder(x)
        x = x.permute(1,0,2)
        # Since batch_first is not a parameter in trasformer so the input must be S, N, E
        
        out = self.encoder(x, src_key_padding_mask=x_mask)
        # out shape : (S, N, E)
        return out

# %%
class Module2TransformerEnc(nn.Module):
    # S, N, E : (seq_len, batch_size, input/embedding_size)
    def __init__(self, input_size, n_head, dim_ff, dp, num_layers, act_fn = 'relu'):
        super(Module2TransformerEnc, self).__init__()
        self.input_size = input_size
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = input_size, nhead = n_head, dim_feedforward = dim_ff, dropout=dp, activation=act_fn)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.positional_encoder = PositionalEncoding(
            dim_model=input_size, dropout_p=dp, max_len=MAX_SEQUENCE_LEN
        )

    def forward(self, x, x_mask):
        # x shape: seq_len, batch_size, input_size
        x = self.positional_encoder(x)
        x = x.permute(1,0,2)
        # Since batch_first is not a parameter in trasformer so the input must be S, N, E
        
        out = self.encoder(x, src_key_padding_mask=x_mask)
        # out shape : (S, N, E)
        return out

# %%
class Module1GRU(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, output_size):
        super(Module1GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Since there are maximum of 8 speakers in a dialogue, so we decided to make 8 GRUs one for each speaker.
        self.gru_list= []
        for id in range(MAX_NO_OF_SPEAKERS):
            self.gru_list.append(nn.GRU(input_size, hidden_size, num_layers, batch_first = True))
        self.gru_modules = nn.ModuleList(self.gru_list)
        # self.fc  = nn.Linear(num_layers*hidden_size, output_size)
            
    
    def segregateEmotions(self, emotions, speakers):
        speaker_specific = []
        utt_id = []
        for i in range(MAX_NO_OF_SPEAKERS):
            speaker_tensor = torch.zeros(MAX_NO_OF_SPEAKERS, device = device)
            speaker_tensor[i] = 1
            emo = emotions[torch.nonzero((speakers == speaker_tensor).sum(dim=1) == speakers.size(1))].permute(1,0,2)
            if(emo.size(1) == 0):
                continue
            utt_id.append(torch.nonzero((speakers == speaker_tensor).sum(dim=1) == speakers.size(1))[0])
            speaker_specific.append(emo)
#             print('\n emo size : ',emo.size())
#         print('\n emo concat size : ',speaker_specific, utt_id)
        return speaker_specific, utt_id
    
    def applyGRU(self, speaker_specific, utt_id, seq_len):
        speaker_output = torch.zeros(seq_len, self.output_size, device = device)  
        for sp_idx in range(len(utt_id)):
            h0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(device)
            out, hn = self.gru_list[sp_idx](speaker_specific[sp_idx], h0)
            for uid in range(utt_id[sp_idx].size(0)):
                speaker_output[utt_id[sp_idx][uid]] = out[0][uid].clone()
        return speaker_output

    def forward(self, x, speakers):
        batch_size = x.size(0)
        seq_len    = x.size(1)
        outputs = []
        for i in range(batch_size):
            speaker_specific, utt_id = self.segregateEmotions(x[i], speakers[i])
            out = self.applyGRU(speaker_specific, utt_id, seq_len)
            outputs.append(out)
        
        final_output = torch.cat([outputs[i].unsqueeze(2) for i in range(len(outputs))], 2).permute(2,0,1)
        
        return final_output

# %%
class FinalModel(L.LightningModule):
        def __init__(self, 
                input_size_1, hidden_size_1, num_layers_1, output_size_1,      # module 1    
                input_size_2, n_head_2, dm_ff_2, dp_2, num_layers_2, act_fn_2, # module 2
                input_size_3, n_head_3, dm_ff_3, dp_3, num_layers_3, act_fn_3, # module 3
                input_size_4, hidden_size_4, num_layers_4, output_size_4,      # module 4
                input_size_5, n_head_5, dm_ff_5, dp_5, num_layers_5, act_fn_5, # module 5
                input_size_6, hidden_size_6, num_layers_6, output_size_6,      # module 6
                fc1_out, fc2_out, fc3_out, fc4_out, fc5_out, fc6_out, fc7_out,dp, criterion #masking = False            # final Model parameters
                ):
                super(FinalModel, self).__init__()

                #self.masking = masking
                self.criterion = criterion
                #self.module1 = Module1GRU(input_size = input_size_1, num_layers = num_layers_1, hidden_size = hidden_size_1, output_size = output_size_1)
                self.module2 = Module2TransformerEnc(input_size = input_size_2, n_head = n_head_2, dim_ff = dm_ff_2, dp = dp_2, num_layers = num_layers_2, act_fn = act_fn_2)
                self.module3 = Module3TransformerEnc(input_size = input_size_3, n_head = n_head_3, dim_ff = dm_ff_3, dp = dp_3, num_layers = num_layers_3, act_fn = act_fn_3)
                #self.module4 = Module4GRU(input_size = input_size_4, num_layers = num_layers_4, hidden_size = hidden_size_4, output_size = output_size_4)
                #self.module5 = Module5TransformerEnc(input_size = input_size_5, n_head = n_head_5, dim_ff = dm_ff_5, dp = dp_5, num_layers = num_layers_5, act_fn = act_fn_5)
                #self.module6 = Module6GRU(input_size = input_size_6, num_layers = num_layers_6, hidden_size = hidden_size_6, output_size = output_size_6)

                
                self.fc1 = nn.Linear(input_size_2+input_size_3, fc1_out)
                self.classification = nn.Sequential(
                        nn.Linear((fc1_out ), fc2_out),#+ input_size_5
                        nn.ReLU(),
                        nn.Dropout(dp), 
                        nn.Linear(fc2_out, fc3_out),
                        nn.ReLU(),
                        nn.Dropout(dp),
                        nn.Linear(fc3_out, fc4_out),
                        nn.ReLU(),
                        nn.Dropout(dp),
                        nn.Linear(fc4_out, fc5_out),
                        nn.ReLU(),
                        nn.Dropout(dp),
                        nn.Linear(fc5_out, fc6_out),
                        nn.ReLU(),
                        nn.Dropout(dp),
                        nn.Linear(fc6_out, fc7_out),
                        nn.Softmax(dim=-1)
                )
                self.lstm = nn.LSTM(input_size_6, hidden_size_6, num_layers_6, dropout=dp)
                
        def forward(self, x):
                speaker = x['speaker']
                triggers = x['triggers'].T
                sentence_embeddings = x['sentence_embeddings']
                VAD = x['VAD'].permute(1, 0, 2)
                VAD_mask = x['VAD_mask']
                
                #out1 = self.module1(emotion, speaker)
                #print('Hay NaN Out1') if out1.isnan().any() == True else None
                out2 = self.module2(sentence_embeddings, VAD_mask)
                print('Hay NaN Out2') if out2.isnan().any() == True else None
                out3 = self.module3(speaker, VAD_mask)
                print('Hay NaN Out3') if out3.isnan().any() == True else None
                #out4 = self.module4(triggers)
                #print('Hay NaN Out4') if out4.isnan().any() == True else None
                #out5 = self.module5(VAD, VAD_mask).permute(1,0,2)
                #print('Hay NaN Out5') if out5.isnan().any() == True else None
                #out6 = self.module6(VAD, speaker)
                #print('Hay NaN Out6') if out6.isnan().any() == True else None

                #out46 = torch.cat((out4, out6), 2)
                out235 = F.relu(self.fc1(out235))
                #out23456 = torch.cat((out46, out235), 2),  triggers.unsqueeze(-1)
                #out23456_vad = torch.cat((out235), 2)#VAD,
                final_seq = self.classification(out235)
                return final_seq
            

        def training_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self(x)
                predics = y_hat
                labels = y['emotion'].permute(1, 0, 2)
                mask = x['VAD_mask'].T
                #labels_i = y['emotion'][i][torch.nonzero(x['VAD_mask'][i].logical_not()).squeeze()]
                #predics_i = y_hat[i][torch.nonzero(x['VAD_mask'][i].logical_not()).squeeze()]
                #y_hat = y_hat.view(-1, y_hat.size(2))
                #labels = y['emotion'].view(-1, y['emotion'].size(2))
                #padding_idxs = torch.any(labels != 0, dim=1)
                #labels = labels[padding_idxs]
                #predics = y_hat[padding_idxs]
                #print(predics, '\n', labels, '\n\n')
                F1score = torchmetrics.classification.MulticlassF1Score(num_classes=7).to(device)
                    #print(predics_i, '\n', labels_i, '\n\n')
                y_hat = y_hat.view(-1, y_hat.size(2))
                labels = y['emotion'].view(-1, y['emotion'].size(2))
                padding_idxs = torch.any(labels != 0, dim=1)
                labels = labels[padding_idxs]
                predics = y_hat[padding_idxs]
                print(predics, '\n', labels, '\n\n')
                loss_mean = criterion(predics, labels)  # Acumula la pérdida en cada iteración
                score_mean = F1score(predics, labels)
                
                self.log('Train_Loss', loss_mean, on_epoch=True, prog_bar=True, logger=True)
                self.log('F1Score-Train', score_mean, on_epoch=True, prog_bar=True, logger=True)

                return loss_mean  # Devuelve el promedio de la pérdida

        
        def validation_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self(x)
                predics = y_hat
                labels = y['emotion'].permute(1, 0, 2)
                mask = x['VAD_mask'].T
                #labels_i = y['emotion'][i][torch.nonzero(x['VAD_mask'][i].logical_not()).squeeze()]
                #predics_i = y_hat[i][torch.nonzero(x['VAD_mask'][i].logical_not()).squeeze()]
                #y_hat = y_hat.view(-1, y_hat.size(2))
                #labels = y['emotion'].view(-1, y['emotion'].size(2))
                #padding_idxs = torch.any(labels != 0, dim=1)
                #labels = labels[padding_idxs]
                #predics = y_hat[padding_idxs]
                #print(predics, '\n', labels, '\n\n')
                F1score = torchmetrics.classification.MulticlassF1Score(num_classes=7).to(device)
                    #print(predics_i, '\n', labels_i, '\n\n')
                y_hat = y_hat.view(-1, y_hat.size(2))
                labels = y['emotion'].view(-1, y['emotion'].size(2))
                padding_idxs = torch.any(labels != 0, dim=1)
                labels = labels[padding_idxs]
                predics = y_hat[padding_idxs]
                loss_mean = criterion(predics, labels)  # Acumula la pérdida en cada iteración
                score_mean = F1score(predics, labels)
                
                self.log('Val_Loss', loss_mean, on_epoch=True, prog_bar=True, logger=True)
                self.log('F1Score-Val',score_mean, on_epoch=True, prog_bar=True, logger=True)
    

                return loss_mean  # Devuelve el promedio de la pérdida
        
        def configure_optimizers(self):
                optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
                #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.05)
                return optimizer#, [lr_scheduler]
    
        def test_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self(x)
                predics = y_hat
                labels = y['emotion'].permute(1, 0, 2)
                mask = x['VAD_mask'].T
                #labels_i = y['emotion'][i][torch.nonzero(x['VAD_mask'][i].logical_not()).squeeze()]
                #predics_i = y_hat[i][torch.nonzero(x['VAD_mask'][i].logical_not()).squeeze()]
                #y_hat = y_hat.view(-1, y_hat.size(2))
                #labels = y['emotion'].view(-1, y['emotion'].size(2))
                #padding_idxs = torch.any(labels != 0, dim=1)
                #labels = labels[padding_idxs]
                #predics = y_hat[padding_idxs]
                #print(predics, '\n', labels, '\n\n')
                F1score = torchmetrics.classification.MulticlassF1Score(num_classes=7).to(device)
                    #print(predics_i, '\n', labels_i, '\n\n')
                y_hat = y_hat.view(-1, y_hat.size(2))
                labels = y['emotion'].view(-1, y['emotion'].size(2))
                padding_idxs = torch.any(labels != 0, dim=1)
                labels = labels[padding_idxs]
                predics = y_hat[padding_idxs]
                loss_mean = criterion(predics, labels)  # Acumula la pérdida en cada iteración
                score_mean = F1score(predics, labels)
                self.log('Test_Loss', loss_mean, on_epoch=True, prog_bar=True, logger=True)
                self.log('F1Score-Test', score_mean, on_epoch=True, prog_bar=True, logger=True)
                return loss_mean

# %%
criterion = nn.CrossEntropyLoss()
model = FinalModel(input_size_1, hidden_size_1, num_layers_1, output_size_1,input_size_2, n_head_2, dm_ff_2, dp_2, num_layers_2, act_fn_2,input_size_3, n_head_3, dm_ff_3, dp_3, num_layers_3, act_fn_3, input_size_4, hidden_size_4, num_layers_4, output_size_4,  input_size_5, n_head_5, dm_ff_5, dp_5, num_layers_5, act_fn_5, input_size_6, hidden_size_6, num_layers_6, output_size_6, fc1_out, fc2_out, fc3_out, fc4_out, fc5_out, fc6_out, fc7_out, dp=0.2 ,criterion=criterion)
        

# %%
trainer = L.Trainer()
trainer.fit(model, train_loader, val_loader)

# %%
trainer.test(model, dataloaders=train_loader)


