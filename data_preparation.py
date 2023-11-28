# %%
import os
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, RobertaModel, RobertaTokenizer
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import tqdm
import matplotlib.pyplot as plt
from collections import Counter
import pickle
import re
import copy
import pprint
import time

MAX_NO_OF_SPEAKERS = 8
MAX_VALENCE_LEN    = 69
MAX_DIALOGUE_LEN   = 33
original_labels    = ['abuse', 'adoration', 'annoyance', 'awkwardness', 'benefit', 'boredom', 'calmness', 'challenge', 'cheer', 'confusion', 'curiosity', 'desire', 'excitement', 'guilt', 'horror', 'humour', 'impressed', 'loss', 'nervousness', 'nostalgia', 'pain', 'relief', 'satisfaction', 'scold', 'shock', 'sympathy', 'threat']
train_count        = [31, 190, 1051, 880, 220, 78, 752, 214, 534, 486, 545, 180, 867, 216, 280, 153, 257, 351, 398, 65, 36, 173, 136, 94, 372, 209, 263]

EMOTIONS           = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

sent_model = 'roberta-base-nli-stsb-mean-tokens'

print('tr version', transformers.__version__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device => ",device, ' torch ', torch.__version__)

# %%
class EmotionClassifier(nn.Module):
    def __init__(self, n_classes):
        super(EmotionClassifier, self).__init__()
        self.bert = RobertaModel.from_pretrained('roberta-base')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    def forward(self, input_ids, attention_mask):
        op = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        output = self.drop(op[1])
        return self.out(output), op[1]

# load finetuned roberta model
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_finetuned = EmotionClassifier(7).to(device)
#roberta_tf_checkpoint = torch.load('dump_files/finetuned/best_model_state_roberta.bin', map_location=torch.device(device))
#roberta_finetuned.load_state_dict(roberta_tf_checkpoint)
print('model loaded')


def get_emotion(train_df):
    """
    Convierte una lista de emociones en un DataFrame de variables dummy y luego crea una lista de estas
    variables dummy para cada conjunto de emociones en train_df['emotions'].

    :param train_df: DataFrame de pandas que contiene la columna 'emotions'.
    :param EMOTIONS: Lista de todas las emociones posibles.
    :return: Lista de listas de variables dummy para cada conjunto de emociones.
    """
    # Crear un DataFrame de variables dummy para EMOTIONS
    dummies = pd.get_dummies(EMOTIONS)

    # Crear la lista de listas de variables dummy para cada fila en train_df['emotions']
    listaEmo = [[dummies[emotion] for emotion in emotion_list] for emotion_list in train_df['emotions']]

    return listaEmo

# Uso de la función
# listaEmo = get_emotion(train_df, EMOTIONS)

def process_speakers(df, specific_speakers_len = False):
    """
    Procesa una columna de 'speakers' en un DataFrame para crear una lista de variables dummy y
    una lista de listas de estas variables dummy para cada fila.

    :param df: DataFrame de pandas.
    :return: Una tupla que contiene (speaker_specific, listasp).
    """
    # Obtener una lista única de speakers
    if specific_speakers_len:
        listasp = []
        for conversation in df['speakers']:
            # Obtener una lista única de speakers para la conversación actual
            listSpk = sorted(set(conversation))

            # Crear un DataFrame de variables dummy para los speakers de esta conversación
            speaker_specific = pd.get_dummies(listSpk)

            # Almacenar el DataFrame en la lista
            listasp.append(speaker_specific)
    else:
        listSpk = sorted(set(speaker for sublist in df['speakers'] for speaker in sublist))
        # Crear un DataFrame de variables dummy
        speaker_specific = pd.get_dummies(listSpk)
        # Crear la lista de listas de variables dummy para cada fila en df['speakers']
        listasp = [[speaker_specific[speaker] for speaker in sublist] for sublist in df['speakers']]

    return listasp

def process_dialogues(train_df):
    """
    Procesa una columna de 'dialogues' en un DataFrame para crear una lista de variables dummy y
    una lista de listas de estas variables dummy para cada fila.

    :param df: DataFrame de pandas.
    :return: Una tupla que contiene (dialogue_specific, listad).
    """
    # Obtener una lista única de dialogues
    sentence_embeddings = []
    for i in range(len(train_df)):
        utterances = train_df['utterances'][i]
        # Asumiendo que 'utterances' es una lista de oraciones
        embeddings = []
        for utt in utterances:
            encodings = roberta_tokenizer.encode_plus(utt, max_length=100, padding='max_length', add_special_tokens=True, return_token_type_ids=True, return_attention_mask=True, truncation=True, return_tensors='pt').to(device)
            utt_emb = roberta_finetuned(encodings['input_ids'], encodings['attention_mask'])[1].detach().tolist()[0]
            utt_emb = np.round(utt_emb, decimals=10)
            embeddings.append(utt_emb)
        sentence_embeddings.append(embeddings)
    return sentence_embeddings

def process_VAD(train_df, csvread):
    def clean_and_get_values(sentence):
        cleaned_words = [re.sub(r'[^a-zA-Z]', '', word).lower() for word in sentence.split()]
        return [[float(csvread.at[word, prop]) if word in csvread.index else 0 for prop in ['Valence', 'Arousal', 'Dominance']] for word in cleaned_words]

    # Vectorizando la limpieza y obtención de valores
    utterance_values = train_df['utterances'].apply(lambda x: [clean_and_get_values(sentence) for sentence in x])

    # Separar en listas de Valence, Arousal y Dominance
    valen = utterance_values.apply(lambda x: [[val[0] for val in sentence] for sentence in x])
    aros = utterance_values.apply(lambda x: [[val[1] for val in sentence] for sentence in x])
    domi = utterance_values.apply(lambda x: [[val[2] for val in sentence] for sentence in x])

    return valen, aros, domi

def pad_utterances(utterance_values, max_length):
    """
    Aplica padding a las listas de valores de utterances para que todas tengan la misma longitud.

    :param utterance_values: Lista de listas de listas con los valores de Valence, Arousal y Dominance.
    :param max_length: La longitud máxima para el padding.
    :return: Lista de listas de listas con los valores paddeados.
    """
    padded_values = utterance_values.apply(lambda x: [np.pad(sublist, (0, max_length - len(sublist)), 'constant') for sublist in x])
    return padded_values

# %%
def clean_triggers(train_df):
    return train_df['triggers'].apply(lambda x: [0 if pd.isna(item) else item for item in x])

# %%
train_csv = pd.read_json("EDiReF-Train-Data/Task 3/MELD_train_efr.json")
csvread = pd.read_csv("./EDiReF-Train-Data/Task 3/out.csv",names=["Valence", "Arousal", "Dominance"])
train_df = pd.DataFrame(train_csv)


