from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/api/data')
def get_data():
    return jsonify({'message': 'Hello from Python backend!'})

if __name__ == '__main__':
    # Run the app on port 5000
    app.run(debug=True, port=5000)
