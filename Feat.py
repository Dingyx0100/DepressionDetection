import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import librosa
import sys
# sys.path.append('/Users/linlin/Desktop/DepressionCollected')
from Classification.audio_features_whole import wav2vlad

import numpy as np
import pandas as pd
import wave

import jieba

from elmoformanylangs import Embedder
# elmo = Embedder('../autodl-tmp/eng.model')


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
    transcript = pd.read_csv(os.path.join('../autodl-tmp/Transcript/{0}_TRANSCRIPT.csv'.format(number)), sep='\t').fillna('')
    
    wavefile = wave.open(os.path.join('../autodl-tmp/audio_data/{0}_AUDIO.wav'.format(number, 'r')))
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
    stopt, lastt = 0, 0

    for t in transcript.itertuples():
        # 问题开始
        if getattr(t,'speaker') == 'Ellie' and (identify_topics(getattr(t,'value')) or 'i think i have asked everything' in getattr(t,'value')):
            # 初始化
            response = ''
            if len(signal) == 0:
                continue
            amp = max(signal)-min(signal)
            cent = librosa.feature.spectral_centroid(y=signal, sr=sr)
            cent_times = librosa.frames_to_time(np.arange(len(cent[0])), sr=sr, hop_length=512)
            feats.append([amp, np.mean(cent_times), stopt])
            signal, seg_text_iter = [], []
            lastt, stopt = int(getattr(t,'start_time')*sr), 0
        elif getattr(t,'speaker') == 'Participant':
            if 'scrubbed_entry' in getattr(t,'value'):
                continue
            start_time = int(getattr(t,'start_time')*sr)
            stop_time = int(getattr(t,'stop_time')*sr)
            stopt += start_time-lastt
            response += (' ' + getattr(t,'value'))
            signal = np.hstack((signal, wave_data[start_time:stop_time].astype(np.float64)))
    
#     text_feats = [np.array(item).mean(axis=0) for item in elmo.sents2elmo(text)]
    print(np.shape(feats))
    print('{}_P feature done'.format(number))
    return feats


    
# training set
features_train = []

# test set
features_test = []


# training set
for index in range(len(train_split_num)):
    feat= extract_features(train_split_num[index])
    features_train.append(feat)

print("Saving npz file locally...")
np.savez(os.path.join(prefix, 'DAICCode/Features_E/train_feat.npz'), features_train)

# test set
for index in range(len(test_split_num)):
    if test_split_num[index] == 451 or test_split_num[index] == 458:
        continue
    feat = extract_features(test_split_num[index])
    features_test.append(feat)

print("Saving npz file locally...")
np.savez(os.path.join(prefix, 'DAICCode/Features_E/test_feat.npz'), features_test)