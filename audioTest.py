import os
import librosa
import pandas as pd

# Function to extract features from an audio file
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    chroma_stft_mean = librosa.feature.chroma_stft(y=y, sr=sr).mean()
    chroma_stft_var = librosa.feature.chroma_stft(y=y, sr=sr).var()
    rms = librosa.feature.rms(y=y)
    return chroma_stft_mean, chroma_stft_var, rms[0][0]

# Main function to process all audio files in a folder
def process_audio_folder(folder_path):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            chroma_stft_mean, chroma_stft_var, rms = extract_features(file_path)
            results.append([filename, chroma_stft_mean, chroma_stft_var, rms])
    return results

# Folder containing audio files
folder_path = '/Users/lchilly/Desktop/Astra/genres_original/blues'

# Process audio files
results = process_audio_folder(folder_path)

# Write results to a CSV file
df = pd.DataFrame(results, columns=['Filename', 'Chroma STFT Mean', 'Chroma STFT Variation', 'RMS'])
df.to_csv('/Users/lchilly/Desktop/Astra/audio_features.csv', index=False)