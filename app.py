"""
Flask backend API to query precomputed bank prediction data from SQLite databases.
"""

from flask import Flask, request, jsonify
import config
from db_manager import get_all_predictions, query_predictions

app = Flask(__name__)

@app.route('/')
def index() -> str:
    """
    Health-check endpoint.
    """
    return "Bank Predictions API is running."

@app.route('/predictions', methods=['GET'])
def predictions() -> 'json':
    """
    Retrieves prediction records based on query parameters.
    
    Query parameters should match state features (wealth, risk_tolerance, loyalty, inflation, gdp_growth, euribor)
    and a bank_id (to select the corresponding database). For example:
    
        /predictions?bank_id=1&wealth=500
    
    :return: JSON list of prediction records.
    """
    bank_id = request.args.get('bank_id', default=1, type=int)
    db_path = config.PREDICTIONS_DB_PATH_TEMPLATE.format(bank_id=bank_id)
    
    # Build filters from query parameters
    filters = {}
    for feature in ['wealth', 'risk_tolerance', 'loyalty', 'inflation', 'gdp_growth', 'euribor']:
        value = request.args.get(feature, default=None, type=float)
        if value is not None:
            filters[feature] = value
    
    if filters:
        results = query_predictions(db_path, filters)
    else:
        results = get_all_predictions(db_path)
    
    # Convert results into a list of dictionaries
    predictions_list = []
    for row in results:
        predictions_list.append({
            "wealth": row[0],
            "risk_tolerance": row[1],
            "loyalty": row[2],
            "inflation": row[3],
            "gdp_growth": row[4],
            "euribor": row[5],
            "predicted_rate": row[6]
        })
    
    return jsonify(predictions_list)

if __name__ == '__main__':
    # Run the Flask API in debug mode (set debug=False in production)
    app.run(debug=True)
