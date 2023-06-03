import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import sys
from Classification.audio_features_whole import wav2vlad

import numpy as np
import pandas as pd
import wave

import jieba

from elmoformanylangs import Embedder
elmo = Embedder('./eng.model')


prefix = os.getcwd()
train_split_df = pd.read_csv(os.path.join(prefix, 'DAIC/train_split_Depression_AVEC2017.csv'))
test_split_df = pd.read_csv(os.path.join(prefix, 'DAIC/dev_split_Depression_AVEC2017.csv'))
train_split_num = train_split_df[['Participant_ID']]['Participant_ID'].tolist()
test_split_num = test_split_df[['Participant_ID']]['Participant_ID'].tolist()
train_split_clabel = train_split_df[['PHQ8_Binary']]['PHQ8_Binary'].tolist()
test_split_clabel = test_split_df[['PHQ8_Binary']]['PHQ8_Binary'].tolist()
train_split_rlabel = train_split_df[['PHQ8_Score']]['PHQ8_Score'].tolist()
test_split_rlabel = test_split_df[['PHQ8_Score']]['PHQ8_Score'].tolist()

with open('./queries.txt') as f:
    queries = f.readlines()

def identify_topics(sentence):
    for query in queries:
        query = query.strip('\n')
        sentence = sentence.strip('\n')
        if query == sentence:
            return True
    return False

def extract_features(number):
    transcript = pd.read_csv(os.path.join('./Transcript/{0}_TRANSCRIPT.csv'.format(number)), sep='\t').fillna('')
    
    wavefile = wave.open(os.path.join('./audio_data/{0}_AUDIO.wav'.format(number, 'r')))
    sr = wavefile.getframerate()
    nframes = wavefile.getnframes()
    wave_data = np.frombuffer(wavefile.readframes(nframes), dtype=np.short)
    
    response = ''
    start_time = 0
    stop_time = 0
    feats = []
    signal = []
    text = []
    seg_text_iter = []

    for t in transcript.itertuples():
        if getattr(t,'speaker') == 'Ellie' and (identify_topics(getattr(t,'value')) or 'i think i have asked everything' in getattr(t,'value')):
            # Initialization
            response = ''
            if len(signal) == 0:
                continue
            feats.append(wav2vlad(signal, sr).tolist())
            text.append([item for items in seg_text_iter for item in items])
            signal, seg_text_iter = [], []
        elif getattr(t,'speaker') == 'Participant':
            if 'scrubbed_entry' in getattr(t,'value'):
                continue
            start_time = int(getattr(t,'start_time')*sr)
            stop_time = int(getattr(t,'stop_time')*sr)
            response += (' ' + getattr(t,'value'))
            seg_text_iter.append(jieba.cut(getattr(t,'value'), cut_all=False))
            signal = np.hstack((signal, wave_data[start_time:stop_time].astype(np.float64)))

    print(np.shape(feats))
    print('{}_P feature done'.format(number))
    return feats,text


    
# training set
audio_features_train = []
text_features_train = []
audio_ctargets_train = []
audio_rtargets_train = []

# test set
audio_features_test = []
text_features_test = []
audio_ctargets_test = []
audio_rtargets_test = []
# negative = 0
# neg_sam = np.random.choice(30, 17, replace=False)

# training set
for index in range(len(train_split_num)):
    feat,text_feats = extract_features(train_split_num[index])
    text_feats = [np.array(item).mean(axis=0) for item in elmo.sents2elmo(text_feats)]
    audio_features_train.append(np.squeeze(np.array(feat)))
    text_features_train.append(np.array(text_feats))
    audio_ctargets_train.append(train_split_clabel[index])
    audio_rtargets_train.append(train_split_rlabel[index])

print("Saving npz file locally...")
np.savez(os.path.join(prefix, 'DAICCode/Features_E/train_samples_clf.npz'), audio_features_train)
np.savez(os.path.join(prefix, 'DAICCode/Features_E/train_samples_reg.npz'), audio_features_train)
np.savez(os.path.join(prefix, 'DAICCode/Features_E/train_text_clf.npz'), text_features_train)
np.savez(os.path.join(prefix, 'DAICCode/Features_E/train_labels_clf.npz'), audio_ctargets_train)
np.savez(os.path.join(prefix, 'DAICCode/Features_E/train_labels_reg.npz'), audio_rtargets_train)

# test set
for index in range(len(test_split_num)):
    if test_split_num[index] == 451 or test_split_num[index] == 458:
        continue
    feat,text_feats = extract_features(test_split_num[index])
    text_feats = [np.array(item).mean(axis=0) for item in elmo.sents2elmo(text_feats)]
    audio_features_test.append(np.squeeze(np.array(feat)))
    text_features_test.append(np.array(text_feats))
    audio_ctargets_test.append(test_split_clabel[index])
    audio_rtargets_test.append(test_split_rlabel[index])

print("Saving npz file locally...")
np.savez(os.path.join(prefix, 'DAICCode/Features_E/test_samples_clf.npz'), audio_features_test)
np.savez(os.path.join(prefix, 'DAICCode/Features_E/test_samples_reg.npz'), audio_features_test)
np.savez(os.path.join(prefix, 'DAICCode/Features_E/test_text_clf.npz'), text_features_test)
np.savez(os.path.join(prefix, 'DAICCode/Features_E/test_labels_clf.npz'), audio_ctargets_test)
np.savez(os.path.join(prefix, 'DAICCode/Features_E/test_labels_reg.npz'), audio_rtargets_test)
