import os
import numpy as np

"""
The basic idea of resampling and grouping:
"""
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

# test set
audio_features_test = []
text_features_test = []
audio_ctargets_test = []
audio_rtargets_test = []

# squeezing out all possible samples
sample_non = np.random.choice(np.where(ctrain_targets == 0)[0], 29, replace=False)
for i in range(atrain_features.shape[0]):
    if ctrain_targets[i] == 0:
        snum = 3 if i not in sample_non else 4
        sample = np.random.choice(atrain_features[i].shape[0]//5, snum, replace=False)
        for j in sample:
            audio_features_train.append(atrain_features[i][j*5:(j+1)*5])
            text_features_train.append(ttrain_features[i][j*5:(j+1)*5])
            audio_ctargets_train.append(ctrain_targets[i])
            audio_rtargets_train.append(rtrain_targets[i])
    else:
        for j in range(atrain_features[i].shape[0]//5):
            audio_features_train.append(atrain_features[i][j*5:(j+1)*5])
            text_features_train.append(ttrain_features[i][j*5:(j+1)*5])
            audio_ctargets_train.append(ctrain_targets[i])
            audio_rtargets_train.append(rtrain_targets[i])

print("Saving npz file locally...")
np.savez(os.path.join(prefix, 'DAICCode/Features3/train_samples_clf.npz'), audio_features_train)
np.savez(os.path.join(prefix, 'DAICCode/Features3/train_samples_reg.npz'), audio_features_train)
np.savez(os.path.join(prefix, 'DAICCode/Features3/train_text_clf.npz'), text_features_train)
np.savez(os.path.join(prefix, 'DAICCode/Features3/train_labels_clf.npz'), audio_ctargets_train)
np.savez(os.path.join(prefix, 'DAICCode/Features3/train_labels_reg.npz'), audio_rtargets_train)

for i in range(atest_features.shape[0]):
    sample = np.random.choice(atest_features[i].shape[0]//5, 2, replace=False)
    for j in sample: #range(atest_features[i].shape[0]//5)
        audio_features_test.append(atest_features[i][j*5:(j+1)*5])
        text_features_test.append(ttest_features[i][j*5:(j+1)*5])
        audio_ctargets_test.append(ctest_targets[i])
        audio_rtargets_test.append(rtest_targets[i])

print("Saving npz file locally...")
np.savez(os.path.join(prefix, 'DAICCode/Features3/test_samples_clf.npz'), audio_features_test)
np.savez(os.path.join(prefix, 'DAICCode/Features3/test_samples_reg.npz'), audio_features_test)
np.savez(os.path.join(prefix, 'DAICCode/Features3/test_text_clf.npz'), text_features_test)
np.savez(os.path.join(prefix, 'DAICCode/Features3/test_labels_clf.npz'), audio_ctargets_test)
np.savez(os.path.join(prefix, 'DAICCode/Features3/test_labels_reg.npz'), audio_rtargets_test)
