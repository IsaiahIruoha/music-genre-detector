from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Variable to hold the current status
current_status = "Not Listening"

@app.route('/')
def home():
    return render_template('index.html', status=current_status)

@app.route('/api/action', methods=['POST'])
def handle_action():
    global current_status
    data = request.json
    # Update the current status based on the received data
    if data.get("isListening"):
        current_status = "Listening"
    else:
        current_status = "Not Listening"
    print(current_status)
    # Send a response back to the client
    return jsonify({"message": "Action received!", "data": data}), 200

@app.route('/api/status', methods=['GET'])
def get_status():
    global current_status
    return jsonify({"status": current_status}), 200

if __name__ == '__main__':
    app.run(debug=True)
