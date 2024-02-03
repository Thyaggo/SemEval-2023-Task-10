import os
import copy
import pickle
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

# Global variables
MAX_NO_OF_SPEAKERS = 9
EMOTIONS = ['anger','joy','surprise','neutral','sadness','disgust','fear','contempt']

model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
train_csv = pd.read_json("EDiReF-Train-Data/Task 1/MaSaC_train_erc.json")

def one_hot_emotions(dts: pd.DataFrame) -> list:
    emotions_dummies = pd.get_dummies(EMOTIONS)
    return [[emotions_dummies[x_1] for x_1 in x_0] for x_0 in dts]

def one_hot_speakers(dts: pd.DataFrame) -> list:
    tmp = []
    for conversation in dts:
        speakers, idxs = np.unique(conversation, return_inverse=True)
        tmp.append(np.eye(len(speakers), MAX_NO_OF_SPEAKERS)[idxs])
    return tmp

# %%
def generate_sentence_embeddings(dts: pd.DataFrame, model: SentenceTransformer) -> list:
    if not os.path.exists('df_sent_task1.pkl'):
        sentence_embeddings = [copy.deepcopy(model.encode(utt)) for utt in dts]
        with open('df_sent_task1.pkl', 'wb') as f:
            pickle.dump(sentence_embeddings, f)
    else:
        with open('df_sent_task1.pkl', 'rb') as f:
            sentence_embeddings = pickle.load(f)
    
    return sentence_embeddings

generate_sentence_embeddings(train_csv["utterances"], model)


