import os
import librosa
import pandas as pd
from pydub import AudioSegment
import tempfile

# Function to convert MP3 to WAV
def convert_mp3_to_wav(mp3_file):
    try:
        sound = AudioSegment.from_mp3(mp3_file)
        wav_file = tempfile.mktemp(suffix='.wav')
        sound.export(wav_file, format="wav")
        return wav_file
    except Exception as e:
        print(f"Error converting {mp3_file} to WAV: {e}")
        return None

# Function to extract features from an audio file
def extract_features(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=None)
        
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        harmony, perceptr = librosa.effects.hpss(y)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        
        features = {
            'filename': os.path.basename(audio_file),
            'length': len(y) / sr,
            'chroma_stft_mean': chroma_stft.mean() if chroma_stft.size else 0,
            'chroma_stft_var': chroma_stft.var() if chroma_stft.size else 0,
            'rms_mean': rms.mean() if rms.size else 0,
            'rms_var': rms.var() if rms.size else 0,
            'spectral_centroid_mean': spectral_centroid.mean() if spectral_centroid.size else 0,
            'spectral_centroid_var': spectral_centroid.var() if spectral_centroid.size else 0,
            'spectral_bandwidth_mean': spectral_bandwidth.mean() if spectral_bandwidth.size else 0,
            'spectral_bandwidth_var': spectral_bandwidth.var() if spectral_bandwidth.size else 0,
            'rolloff_mean': rolloff.mean() if rolloff.size else 0,
            'rolloff_var': rolloff.var() if rolloff.size else 0,
            'zero_crossing_rate_mean': zero_crossing_rate.mean() if zero_crossing_rate.size else 0,
            'zero_crossing_rate_var': zero_crossing_rate.var() if zero_crossing_rate.size else 0,
            'harmony_mean': harmony.mean() if harmony.size else 0,
            'harmony_var': harmony.var() if harmony.size else 0,
            'perceptr_mean': perceptr.mean() if perceptr.size else 0,
            'perceptr_var': perceptr.var() if perceptr.size else 0,
            'tempo': tempo,
        }
        
        for i in range(1, 21):
            features[f'mfcc{i}_mean'] = mfcc[i-1].mean() if mfcc.shape[0] >= i else 0
            features[f'mfcc{i}_var'] = mfcc[i-1].var() if mfcc.shape[0] >= i else 0
        
        return features
    except Exception as e:
        print(f"Error extracting features from {audio_file}: {e}")
        return None

# Main function to process all audio files in a folder
def process_audio_folder(folder_path):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav') or filename.endswith('.mp3'):
            file_path = os.path.join(folder_path, filename)
            try:
                if filename.endswith('.mp3'):
                    file_path = convert_mp3_to_wav(file_path)
                features = extract_features(file_path)
                results.append(features)
            except Exception as e:
                print(f"Error extracting features from {file_path}: {e}")
    return results


# List of genres
genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

# Base folder containing genre subfolders
base_folder_path = '/Users/simonrisk/Desktop/music-genre-detector-main/genres_original'

# Process each genre
for genre in genres:
    folder_path = os.path.join(base_folder_path, genre)
    if not os.path.exists(folder_path):
        print(f"Folder for genre {genre} does not exist: {folder_path}")
        continue
    print(f"Processing genre: {genre}")
    results = process_audio_folder(folder_path)
    if results:
        csv_file_path = f'/Users/simonrisk/Desktop/music-genre-detector-main/{genre}_audio_features.csv'
        df = pd.DataFrame(results)
        df.to_csv(csv_file_path, index=False)
        print(f"CSV file created for genre {genre}: {csv_file_path}")
    else:
        print(f"No audio files processed for genre {genre}")

print("Feature extraction and CSV generation completed for all genres.")
