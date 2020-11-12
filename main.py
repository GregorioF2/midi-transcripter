import classifier
import numpy as np
from scipy.io import wavfile as wav
from midiutil import MIDIFile
import librosa


# ONLY IF IS THE FIRST TIME OR YOU WANT TO RE TRAIN
#classifier.train()
#classifier.save_clf()

print('entrenando classfier: ')
classifier.load_clf()
print('done...\n\n')


event_window = classifier.event_window
clf_window = classifier.clf_window
feature_window_size = classifier.feature_window_size
hop_length = classifier.hop_length

duration = 1    # In beats
volume   = 100

wav_path = './balada_bateria.wav'
print('Leyendo wav')
sample_rate, data = wav.read(wav_path)
amplitude = np.mean(data, axis=1)
onsets = onsets = librosa.onset.onset_detect(y=amplitude, sr=sample_rate, units='time')

duration = onsets[-1] + 2
note_frecuency = duration / len(onsets)
print('note frecuency: ', note_frecuency)

#belive me
tempo = (1/note_frecuency) * 120
print('tempo: ', tempo)

print('Computing features...')
xs = []
for t in onsets:
    samples = amplitude[int((t - clf_window[0]) * sample_rate): int((t + clf_window[1]) * sample_rate)]
    x = classifier.compute_features(samples.astype(float),
                          sample_rate, 
                          window_size=feature_window_size,
                          hop_length=hop_length)
    xs.append(x)

print('\n\nPredicting instruments...')
predictions = classifier.predict_intruments(xs)

mf = MIDIFile(1)     # only 1 track
track = 0 
time     = 0    # In beats
channel = 9
volume = 100
mf.addTrackName(track, time, "Sample Track")
mf.addTempo(track, time, tempo)

print('Generating midi...')
for time in range(len(onsets)):
  for idx, instrument in classifier.INSTRUMENT_NAME_MAPPING.items():
    if predictions[instrument][time]:
      pitch = classifier.INSTRUMENT_KEY_MAPPING[idx]
      duration = 1         # 1 beat long
      mf.addNote(track, channel, pitch, time, duration, volume)

with open("first_midi_try.mid", 'wb') as outf:
      mf.writeFile(outf)

print('Done.')
