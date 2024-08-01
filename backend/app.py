from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sounddevice as sd
import numpy as np
import threading
import pandas as pd
import librosa
import soundfile as sf
from pydub import AudioSegment
import os
import joblib

app = Flask(__name__)
CORS(app)

# Variable to hold the current status
current_status = "Not Listening"
recording = False
audio_thread = None
recording_number = 1
predicted_genres = []  # Add this line

# Load models and other necessary files
rf_best_model = joblib.load('/Users/simonrisk/Desktop/audioInsightFlaskApp/outputs/rf_best_model.pkl')
svm_best_model = joblib.load('/Users/simonrisk/Desktop/audioInsightFlaskApp/outputs/svm_best_model.pkl')
gb_best_model = joblib.load('/Users/simonrisk/Desktop/audioInsightFlaskApp/outputs/gb_best_model.pkl')
knn_best_model = joblib.load('/Users/simonrisk/Desktop/audioInsightFlaskApp/outputs/knn_best_model.pkl')
logreg_best_model = joblib.load('/Users/simonrisk/Desktop/audioInsightFlaskApp/outputs/logreg_best_model.pkl')
scaler = joblib.load('/Users/simonrisk/Desktop/audioInsightFlaskApp/outputs/scaler.pkl')
encoder = joblib.load('/Users/simonrisk/Desktop/audioInsightFlaskApp/outputs/encoder.pkl')

def extract_features_from_segment(y, sr, start_time, end_time):
    segment = y[start_time:end_time]

    chroma_stft = librosa.feature.chroma_stft(y=segment, sr=sr)
    rms = librosa.feature.rms(y=segment)
    spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=segment)
    harmony, perceptr = librosa.effects.hpss(segment)
    tempo, _ = librosa.beat.beat_track(y=segment, sr=sr)
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20)

    features = {
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

def load_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
    except sf.LibsndfileError:
        print(f"LibsndfileError: {file_path}")
        return None, None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None
    return y, sr

def extract_features(audio_file, segment_duration=30):
    try:
        y, sr = load_audio(audio_file)
        if y is None:
            return []

        total_duration = len(y) / sr
        segment_length = int(sr * segment_duration)

        features_list = []

        for start in range(0, len(y), segment_length):
            end = start + segment_length
            if end <= len(y):
                segment_features = extract_features_from_segment(y, sr, start, end)
                all_features = segment_features
                all_features['filename'] = os.path.basename(audio_file)
                all_features['start'] = start / sr
                all_features['end'] = end / sr
                features_list.append(all_features)

        return features_list

    except Exception as e:
        print(f"Error extracting features from {audio_file}: {e}")
        return []

def process_audio_file(file_path):
    results = []
    if file_path.endswith('.wav'):
        features_list = extract_features(file_path)
        results.extend(features_list)
    return results

def create_csv_from_audio(file_path, output_csv_path):
    all_results = process_audio_file(file_path)
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv_path, index=False)
    print(f"CSV file saved to {output_csv_path}")

def evaluate_model_on_external_data(model, X_ext):
    y_pred = model.predict(X_ext)
    return y_pred

def predict_genre(features):
    # Ensure all columns except 'filename', 'start', and 'end' are converted to numeric
    numeric_features = features.drop(columns=['filename', 'start', 'end']).apply(pd.to_numeric, errors='coerce')
    
    # Replace any NaNs that might have resulted from coercion to numeric
    numeric_features = numeric_features.fillna(0)
    
    # Scale the features
    X_ext = scaler.transform(numeric_features)
    
    # Evaluate each model and gather predictions
    rf_predictions = evaluate_model_on_external_data(rf_best_model, X_ext)
    svm_predictions = evaluate_model_on_external_data(svm_best_model, X_ext)
    gb_predictions = evaluate_model_on_external_data(gb_best_model, X_ext)
    knn_predictions = evaluate_model_on_external_data(knn_best_model, X_ext)
    logreg_predictions = evaluate_model_on_external_data(logreg_best_model, X_ext)

    # Aggregate predictions
    all_predictions = np.vstack([rf_predictions, svm_predictions, gb_predictions, knn_predictions, logreg_predictions])
    final_predictions = [np.bincount(row).argmax() for row in all_predictions.T]

    # Map predicted numbers to genre names
    final_predictions_genre = [encoder.classes_[pred] for pred in final_predictions]
    
    return final_predictions_genre

def record_audio():
    global recording, recording_number, predicted_genres  # Add predicted_genres here
    filename = f"recorded_audio_{recording_number}.wav"
    sample_rate = 44100  # 44.1kHz sample rate
    channels = 1  # Mono
    dtype = 'int16'  # 16-bit audio

    audio_data = []

    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio_data.append(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=channels, dtype=dtype, callback=callback):
        while recording:
            sd.sleep(100)

    audio_array = np.concatenate(audio_data, axis=0)
    
    audio_segment = AudioSegment(
        audio_array.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio_array.dtype.itemsize,  # Automatically detects sample width
        channels=channels
    )

    # Normalize the audio to reduce background noise
    audio_segment = audio_segment.normalize()

    audio_segment.export(filename, format="wav")
    print(f"Saved recording to {filename}")
    
    # Process the recorded audio and create a CSV
    csv_filename = f"audio_features_{recording_number}.csv"
    create_csv_from_audio(filename, csv_filename)

    # Predict the genre
    features_df = pd.read_csv(csv_filename)
    genres = predict_genre(features_df)
    print("Predicted Genres: ", genres)
    
    # Update the global variable
    predicted_genres = genres  # Add this line

    recording_number += 1

@app.route('/')
def home():
    return render_template('index.html', status=current_status)

@app.route('/api/action', methods=['POST'])
def handle_action():
    global current_status, recording, audio_thread
    data = request.json
    # Update the current status based on the received data
    if data.get("isListening"):
        current_status = "Listening"
        if not recording:
            recording = True
            audio_thread = threading.Thread(target=record_audio)
            audio_thread.start()
    else:
        current_status = "Not Listening"
        if recording:
            recording = False
            if audio_thread:
                audio_thread.join()
                audio_thread = None
    print(current_status)
    # Send a response back to the client
    return jsonify({"message": "Action received!", "data": data}), 200

@app.route('/api/status', methods=['GET'])
def get_status():
    global current_status
    return jsonify({"status": current_status}), 200

# Add this endpoint to fetch the predicted genres
@app.route('/api/genres', methods=['GET'])
def get_genres():
    global predicted_genres
    return jsonify({"genres": predicted_genres}), 200

if __name__ == '__main__':
    app.run(debug=True)
