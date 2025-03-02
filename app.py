from flask import Flask, jsonify, render_template
import pickle

with open('models/relationships.pkl', 'rb') as f:
    relationships = pickle.load(f)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def default_index():
    return render_template('index.html')

@app.route('/get_relationships', methods=['GET'])
def get_relationships_data():    
    relationship_data = {}
    for feature, (values, predictions) in relationships.items():
        relationship_data[feature] = {
            "values": values.tolist(),
            "predictions": predictions
        }
    
    return jsonify(relationship_data)

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0")
