import os
import numpy as np

prefix = os.path.abspath(os.path.join(os.getcwd(), "./"))
prefix = os.path.abspath(os.path.join(os.getcwd(), "./"))
atrain_features = np.load(os.path.join(prefix, 'DAICCode/Features/train_samples_clf.npz'), allow_pickle=True)['arr_0']
atest_features = np.load(os.path.join(prefix, 'DAICCode/Features/test_samples_clf.npz'), allow_pickle=True)['arr_0']
ttrain_features = np.load(os.path.join(prefix, 'DAICCode/Features/train_text_clf.npz'), allow_pickle=True)['arr_0']
ttest_features = np.load(os.path.join(prefix, 'DAICCode/Features/test_text_clf.npz'), allow_pickle=True)['arr_0']

ctrain_targets = np.load(os.path.join(prefix, 'DAICCode/Features/train_labels_clf.npz'), allow_pickle=True)['arr_0']
ctest_targets = np.load(os.path.join(prefix, 'DAICCode/Features/test_labels_clf.npz'), allow_pickle=True)['arr_0']
rtrain_targets = np.load(os.path.join(prefix, 'DAICCode/Features/train_labels_reg.npz'), allow_pickle=True)['arr_0']
rtest_targets = np.load(os.path.join(prefix, 'DAICCode/Features/test_labels_reg.npz'), allow_pickle=True)['arr_0']

# training set
audio_features_train = []
text_features_train = []
audio_ctargets_train = []
audio_rtargets_train = []

for i in range(atrain_features.shape[0]):
    for j in range(atrain_features[i].shape[0]//5):
        audio_features_train.append(atrain_features[i][j*5:(j+1)*5])
        text_features_train.append(ttrain_features[i][j*5:(j+1)*5])
        audio_ctargets_train.append(ctrain_targets[i])
        audio_rtargets_train.append(rtrain_targets[i])

        
X_train, Y_train = audio_features_train, audio_ctargets_train
# rf
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=50)
clf.fit([f.flatten() for f in X_train], Y_train)
pred = clf.predict([f.flatten() for f in X_train])
index = (pred-Y_train).astype(int)
print(1-np.sum(index)/len(index))
