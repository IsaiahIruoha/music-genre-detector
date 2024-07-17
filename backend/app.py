from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Welcome to the Flask server! Navigate to /api/action to see the endpoint."

@app.route('/api/action', methods=['POST'])
def handle_action():
    data = request.json
    # Process the data received from the client
    print(data)
    # Send a response back to the client
    return jsonify({"message": "Action received!", "data": data}), 200

if __name__ == '__main__':
    app.run(debug=True)
