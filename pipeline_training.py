"""pipeline_training.py: purpose of this script is to train the generated pipeline model through genetic approach"""

__author__ = "Dhaval Thakkar"

from sklearn.ensemble import ExtraTreesClassifier
import glob
import librosa
import librosa.display
import numpy as np
import pandas as pd
import _pickle as pickle


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                              sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz


def parse_audio_files(path):
    features, labels = np.empty((0, 193)), np.empty(0)
    labels = []
    for fn in glob.glob(path):
        try:
            mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
        except Exception as e:
            print("Error encountered while parsing file: ", fn)
            continue
        ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        features = np.vstack([features, ext_features])
        labels = np.append(labels, fn.split("_")[3].split(".")[0])

    return np.array(features), np.array(labels)


tr_features, tr_labels = parse_audio_files('./training_sounds/*.wav')
ts_features, ts_labels = parse_audio_files('./test_sounds/*.wav')

tr_features = np.array(tr_features, dtype=pd.Series)
tr_labels = np.array(tr_labels, dtype=pd.Series)

ts_features = np.array(ts_features, dtype=pd.Series)
ts_labels = np.array(ts_labels, dtype=pd.Series)

# Score on the training set was:0.9964285714285716
exported_pipeline = ExtraTreesClassifier(bootstrap=True, criterion="entropy", max_features=0.25, min_samples_leaf=1,
                                         min_samples_split=8, n_estimators=100)

exported_pipeline.fit(tr_features, tr_labels)

filename = 'genetic_generated_pipeline.sav'

pickle.dump(exported_pipeline, open(filename, 'wb'), protocol=2)

print('Model Saved..')
