from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import librosa
import os
import joblib
import tempfile

app = Flask(__name__)
CORS(app)

current_status = "Not Listening"
genres = []

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    # Remove the check for _MEIPASS since you're not using PyInstaller
    base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

rf_best_model_path = resource_path('outputs/rf_best_model.pkl')
svm_best_model_path = resource_path('outputs/svm_best_model.pkl')
gb_best_model_path = resource_path('outputs/gb_best_model.pkl')
knn_best_model_path = resource_path('outputs/knn_best_model.pkl')
logreg_best_model_path = resource_path('outputs/logreg_best_model.pkl')
scaler_path = resource_path('outputs/scaler.pkl')
encoder_path = resource_path('outputs/encoder.pkl')

rf_best_model = joblib.load(rf_best_model_path)
svm_best_model = joblib.load(svm_best_model_path)
gb_best_model = joblib.load(gb_best_model_path)
knn_best_model = joblib.load(knn_best_model_path)
logreg_best_model = joblib.load(logreg_best_model_path)
scaler = joblib.load(scaler_path)
encoder = joblib.load(encoder_path)

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
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None
    return y, sr

def extract_features(audio_file, segment_duration=10):
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

# def process_audio_file(file_path):
#     results = []
#     if file_path.endswith('.wav'):
#         features_list = extract_features(file_path)
#         results.extend(features_list)
#     return results

def process_audio_file(file_path):
    results = []
    if file_path.endswith('.wav') or file_path.endswith('.m4a'):
        # Convert .m4a to .wav if necessary
        if file_path.endswith('.m4a'):
            wav_file_path = file_path.replace('.m4a', '.wav')
            audio = AudioSegment.from_file(file_path, format='m4a')
            audio.export(wav_file_path, format='wav')
            file_path = wav_file_path
        
        features_list = extract_features(file_path)
        results.extend(features_list)
    return results

def evaluate_model_on_external_data(model, X_ext):
    y_pred = model.predict(X_ext)
    return y_pred

def predict_genre(features):
    numeric_features = pd.DataFrame(features).drop(columns=['filename', 'start', 'end']).apply(pd.to_numeric, errors='coerce')
    numeric_features = numeric_features.fillna(0)

    X_ext = scaler.transform(numeric_features)
    
    rf_predictions = evaluate_model_on_external_data(rf_best_model, X_ext)
    svm_predictions = evaluate_model_on_external_data(svm_best_model, X_ext)
    gb_predictions = evaluate_model_on_external_data(gb_best_model, X_ext)
    knn_predictions = evaluate_model_on_external_data(knn_best_model, X_ext)
    logreg_predictions = evaluate_model_on_external_data(logreg_best_model, X_ext)

    all_predictions = np.vstack([rf_predictions, svm_predictions, gb_predictions, knn_predictions, logreg_predictions])
    final_predictions = [np.bincount(row).argmax() for row in all_predictions.T]

    final_predictions_genre = [encoder.classes_[pred] for pred in final_predictions]
    
    return final_predictions_genre

def record_audio():
    global recording, recording_number, predicted_genres
    filename = f"recorded_audio_{recording_number}.wav"
    sample_rate = 44100
    channels = 1
    dtype = 'int16'

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
        sample_width=audio_array.dtype.itemsize,
        channels=channels
    )

    audio_segment = audio_segment.normalize()

    audio_segment.export(filename, format="wav")
    print(f"Saved recording to {filename}")
    
    csv_filename = f"audio_features_{recording_number}.csv"
    create_csv_from_audio(filename, csv_filename)

    features_df = pd.read_csv(csv_filename)
    genres = predict_genre(features_df)
    print("Predicted Genres: ", genres)
    
    predicted_genres = genres
    
    recording_number += 1
    
    # New: Send the genre prediction back to Swift
    with open('predicted_genre.txt', 'w') as f:
        f.write(genres[0])  # Save the first predicted genre

@app.route('/api/action', methods=['POST'])
def handle_action():
    global current_status, recording, audio_thread
    data = request.json
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
    return jsonify({"message": "Action received!", "data": data}), 200

@app.route('/api/status', methods=['GET'])
def get_status():
    global current_status
    return jsonify({"status": current_status}), 200

@app.route('/api/genres', methods=['GET'])
def get_genres():
    global genres
    return jsonify({"genres": genres}), 200

@app.route('/', methods=['GET'])
def say_hello():
    return "Hello from the music genre classification backend!", 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)