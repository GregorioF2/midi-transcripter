import librosa
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy import signal
from scipy.io import wavfile as wav
from scipy.signal import butter, lfilter, freqz

from IPython.display import Image
from IPython.core.display import HTML 

from mido import MidiFile, Message

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

from joblib import dump, load

sns.set_style('whitegrid')

SOUNDFONT_PATH = './data/live_hq_natural_soundfont_gm.sf2'
DATASET_PATH = './data/mididrums'

classifiers = {}

def midi2audio(midi_path, out_path, out_name=None, southfont_path=SOUNDFONT_PATH):
    out = out_name or midi_path.split('/')[-1].replace('midi', 'wav')
    cmd = f'timidity {midi_path} -Ow -o {out_path}/{out_name}'
    print('Commmnad')
    print(cmd)
    subprocess.call(cmd.split(' '))
    return f'{out_path}/{out_name}'

def bandwise_spectrum(x, sample_rate, frame_length=512, hop_length=256):
    # TODO: this should be a paramter
    bands = [
        [10, 70],
        [70, 130],
        [130, 300],
        [300, 800],
        [800, 1500],
        [1500, 5000],
    ]
    
    # hop_size = nperseg - noverlap => noverlap = nperseg - hop_size
    noverlap = frame_length - frame_length
    f, t, zxx = signal.stft(x, sample_rate, nperseg=frame_length, noverlap=noverlap)
    
    energies = []
    for low, high in bands:
        idx = (low < f) & (f < high)
        energy = np.sum(np.abs(zxx[idx, :]), axis=0)
        energies.append(energy)
    return np.array(energies)

def zero_crossing_rate(x, sample_rate, frame_length=2048, hop_length=2048):
    zcr = librosa.feature.zero_crossing_rate(x, frame_length=frame_length, hop_length=hop_length)
    return [np.mean(zcr), np.var(zcr)]


def energy(x, sample_rate, frame_length=2048, hop_length=2048):
    rmse = librosa.feature.rms(x, frame_length=frame_length, hop_length=hop_length)
    return [np.mean(rmse), np.var(rmse)]


def spectral_centroid(x, sample_rate, frame_length=2048, hop_length=2048):
    sc = librosa.feature.spectral_centroid(x, sr=sample_rate, n_fft=frame_length, hop_length=hop_length)
    return [np.mean(sc), np.var(sc)]


def spectral_rolloff(x, sample_rate, frame_length=2048, hop_length=2048):
    srf = librosa.feature.spectral_rolloff(x, sr=sample_rate, n_fft=frame_length, hop_length=hop_length)
    return [np.mean(srf), np.std(srf)]


def mfcc(x, sample_rate, frame_length=2048, hop_length=2048):
    mfccs = librosa.feature.mfcc(x, sr=sample_rate, n_mfcc=40, n_fft=frame_length, hop_length=hop_length)
    return list(np.concatenate([np.mean(mfccs, axis=1), np.std(mfccs, axis=1)]))

def bandwise_spectrum_features(x, sample_rate, frame_length=2048, hop_length=2048):
    bsf = bandwise_spectrum(x, sample_rate, frame_length=frame_length, hop_length=hop_length)
    return list(np.mean(bsf, axis=1))

FEATURES = {
    'zcr': zero_crossing_rate,
    'energy': energy,
    'spectral_centroid': spectral_centroid,
    'spectral_rolloff': spectral_rolloff,
    'mfcc': mfcc,
    'bands': bandwise_spectrum_features,
}


def compute_features(x, sr, window_size=0.05, hop_length=0.01):
    feats = []
    for name, ffun in FEATURES.items():
        fval = ffun(x, sr, frame_length=int(window_size * sr), hop_length=int(hop_length * sr))
        if isinstance(fval, list):
            feats += fval
        else:
            feats.append(fval)
    return np.array(feats)


INSTRUMENT_KEY_MAPPING = {
    0: 36,
    1: 38,
    2: 50,
    3: 47,
    4: 43,
    5: 42,
    6: 46,
    7: 51,
    8: 49,
}

INSTRUMENT_NAME_MAPPING = {
    0: 'Bass Drum',
    1: 'Snare Drum',
    2: 'High-Tom',
    3: 'Mid-Tom',
    4: 'Floor-Tom',
    5: 'Closed Hi-hat',
    6: 'Open Hi-hat',
    7: 'Ride',
    8: 'Crash',
}

PITCH_INSTRUMENT_MAPPING = {
    22: 5,
    26: 6,
    36: 0,
    37: 1,
    38: 1,
    40: 1,
    42: 5,
    43: 4,
    44: 5,
    45: 3,
    46: 6,
    47: 3,
    48: 2,
    49: 8,
    50: 2,
    51: 7,
    52: 8,
    53: 7,
    55: 8,
    57: 8,
    58: 4,
    59: 7,
}


class MidiGrid:
    
    def __init__(self, midi, midi_instrument_mapping=PITCH_INSTRUMENT_MAPPING):
        self._midi_file = midi
        # Extract tempo and clock information
        self.bpm = (60 * 1000000) / self._midi_file.tracks[0][0].tempo
        self.clocks_per_click = self._midi_file.tracks[0][1].clocks_per_click * 2
        self.time_resolution = (self._midi_file.tracks[0][0].tempo / (1 * self.clocks_per_click)) / 10000000
        # Sequence of midi events
        notes, times = [], []
        time = 0
        for event in self._midi_file.tracks[1]:
            time += event.time
            if event.type == 'note_on' and event.velocity > 0:
                notes.append(event.note)
                times.append(time)
        
        # MIDI-note map
        self.midi_2_instrument = midi_instrument_mapping
        ninstruments = len(np.unique(list(self.midi_2_instrument.values())))
        # Build grid
        self.grid = np.zeros((np.max(times) + 1, ninstruments)).astype(bool)
        for note, time in zip(notes, times):
            if note in self.midi_2_instrument:
                # en la columna del instrumento en el tiempo "time" pone un 1
                self.grid[time, self.midi_2_instrument[note]] = 1
            
    def query(self, start, end):
        st = int(start / self.time_resolution)
        et = int(end / self.time_resolution)
        # array con cuantas veces cada instrumento fue tocado en este intervalo
        return np.sum(self.grid[st:et, :], axis=0) > 0


def process_midi_file(midi_path, outdir='./data/mididrums/synth',
                      event_window=(0.01, 0.01), clf_window=(0.01, 0.05),
                      feature_window_size=0.01, hop_length=0.01):
    # 1 Generate audio track
    # 1.1 Load original midi file
    midi_file = MidiFile(midi_path, clip=True)
    # 1.2 Apply custom instrument mapping and save
    for msg in midi_file.tracks[1]:
        if msg.type == 'note_on' and msg.velocity > 0:
            instrument = PITCH_INSTRUMENT_MAPPING.get(msg.note)
            msg.note = INSTRUMENT_KEY_MAPPING.get(instrument)
    
    track_name = midi_path.split('/')[-1].replace('midi', 'wav')
    fname = f"{outdir}/{track_name}.midi"
    midi_file.save(fname)
    # 1.4 Synthetize with fluidsynth
    audio_synth_path = midi2audio(fname, outdir, track_name)
    
    # 2. Load audio file
    sample_rate, data = wav.read(audio_synth_path)
    amplitude = np.mean(data, axis=1)
    # 3. Initialize midi grid
    midi_grid = MidiGrid(midi_file)
    # 4. Detect onsets
    onsets = librosa.onset.onset_detect(y=amplitude, sr=sample_rate, units='time')
    # 5. Segment audio signal and query ground truth labels
    xs, ys = [], []
    for t in onsets:
        samples = amplitude[int((t - clf_window[0]) * sample_rate): int((t + clf_window[1]) * sample_rate)]
        x = compute_features(samples.astype(float),
                             sample_rate, 
                             window_size=feature_window_size,
                             hop_length=hop_length)
        y = midi_grid.query(t - event_window[0], t + event_window[1])
        xs.append(x); ys.append(y)
    # XS: arreglo de arreglos de features ordeanadas segun el Dict FEATURES
    # YS: arreglo de arreglos que dicen que instrumento toco en la ventana (tn : tn+1)
    # ordeandos por INSTRUMENT_NAME_MAPPING
    return np.array(xs), np.array(ys)



def process_audio_file(path, event_window=(0.01, 0.01), clf_window=(0.01, 0.05),
                      feature_window_size=0.01, hop_length=0.01):
    # 2. Load audio file
    sample_rate, data = wav.read(audio_synth_path)
    amplitude = np.mean(data, axis=1)
    # 3. Initialize midi grid
    onsets = librosa.onset.onset_detect(y=amplitude, sr=sample_rate, units='time')
    # 5. Segment audio signal and query ground truth labels
    xs = []
    for t in onsets:
        samples = amplitude[int((t - clf_window[0]) * sample_rate): int((t + clf_window[1]) * sample_rate)]
        x = compute_features(samples.astype(float),
                             sample_rate, 
                             window_size=feature_window_size,
                             hop_length=hop_length)
        xs.append(x); ys.append(y)
    # XS: arreglo de arreglos de features ordeanadas segun el Dict FEATURES
    # ordeandos por INSTRUMENT_NAME_MAPPING
    return np.array(xs)

def extract_metrics_for_dataset(rows):

    # Sample  tracks
    sample = rows.sample(10)

    event_window = (0.05, 0.075)
    clf_window = (0.01, 0.20)
    feature_window_size = 0.05
    hop_length = 0.01


    xs, ys = [], []
    BASE_PATH = DATASET_PATH
    for i, fname in enumerate(sample['midi_filename'].values):
        print(f'Processing file {i} / {len(sample)}')
        fname = f'{BASE_PATH}/{fname}'
        x, y = process_midi_file(fname, event_window=event_window,
                                clf_window=clf_window,
                                feature_window_size=feature_window_size,
                                hop_length=hop_length)
        xs.append(x); ys.append(y)


    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y

def train_classifier(X_train, y_train):
    classifiers = {}

    for label_idx, instrument_name in INSTRUMENT_NAME_MAPPING.items():
        try:
            clf = GradientBoostingClassifier()
            # y_train[:, label_idx] es un nuevo arreglo, con las apariciones del
            # instrmento i en el intervalo (tn, tn + 1)
            clf.fit(X_train, y_train[:, label_idx])
            classifiers[instrument_name] = clf
        except:
            pass
    return classifiers

def test_classifier(X_test, y_test, classifiers):
    METRICS = {
        'ROC_AUC': lambda y_true, y_pred, y_pred_proba: metrics.roc_auc_score(y_true, y_pred_proba),
        'Precision': lambda y_true, y_pred, y_pred_proba: metrics.precision_score(y_true, y_pred),
        'Recall': lambda y_true, y_pred, y_pred_proba: metrics.recall_score(y_true, y_pred),
        'Accuracy': lambda y_true, y_pred, y_pred_proba: metrics.accuracy_score(y_true, y_pred),
        'F-score': lambda y_true, y_pred, y_pred_proba: metrics.f1_score(y_true, y_pred),
    }

    scores = []
    for idx, instrument in INSTRUMENT_NAME_MAPPING.items():
        try:
            y_pred = classifiers[instrument].predict(X_test)
            y_pred_proba = classifiers[instrument].predict_proba(X_test)[:, 1]
            row = {name: metric(y_test[:, idx], y_pred, y_pred_proba)
                for name, metric in METRICS.items()}
            row['instrument'] = instrument
            row['Positive'] = np.sum(y_test[:, idx])
            row['Negative'] = np.sum(~y_test[:, idx])
            tn, fp, fn, tp = metrics.confusion_matrix(y_test[:, idx], y_pred).ravel()
            row['TP'] = tp
            row['TN'] = tn
            row['FP'] = fp
            row['FN'] = fn
            scores.append(row)
        except Exception as e:
            # raise e
            pass

def train():
    global classifiers
    dataset_index = pd.read_csv(f'{DATASET_PATH}/e-gmd-v1.0.0.csv')
    rows = dataset_index.query('10 < duration < 250 and kit_name=="Acoustic Kit"')

    X_train, y_train = extract_metrics_for_dataset(rows)

    classifiers = train_classifier(X_train, y_train)

def save(filename='classifiers.joblib'):
    dump(classifiers, filename)

def load(filename='classifiers.joblib'):
    laod(filename)
