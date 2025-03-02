import io
import os
import sqlite3
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import config
from db_manager import get_all_predictions, query_predictions

from flask_cors import CORS
app = Flask(__name__)
CORS(app)


@app.route("/")
def index() -> str:
    """Health-check endpoint."""
    return "Bank Predictions API (Flask) is running."

@app.route("/graph-image", methods=["GET"])
def graph_image():
    """
    Endpoint that expects query parameters for x-axis and y-axis.
    For example: /graph-image?x_axis=wealth&y_axis=predicted_rate
    It queries the SQLite database and returns a PNG image of the plot.
    """
    # Retrieve query parameters
    x_axis = request.args.get("x_axis", default="wealth", type=str)
    y_axis = request.args.get("y_axis", default="predicted_rate", type=str)
    bank_id = request.args.get("bank_id", default=1, type=int)
    db_path = config.PREDICTIONS_DB_PATH_TEMPLATE.format(bank_id=bank_id)
    
    # Open connection and query the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Allowed columns (ensure these match your DB schema)
    allowed_customer_cols = {"wealth", "risk_tolerance", "loyalty", "inflation", "gdp_growth", "euribor"}
    allowed_bank_cols = {"predicted_rate"}
    
    # Query x_data
    if x_axis in allowed_customer_cols:
        query_x = f"SELECT {x_axis} FROM predictions"
        cursor.execute(query_x)
        x_data = [row[0] for row in cursor.fetchall()]
    else:
        x_data = []
    
    # Query y_data
    y_data = []
    if y_axis in allowed_bank_cols:
        query_y = f"SELECT {y_axis} FROM predictions"
        cursor.execute(query_y)
        y_data = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    # Fallback dummy data if necessary
    if not x_data or not y_data or len(x_data) != len(y_data):
        x_data = list(range(50))
        y_data = [i * 0.1 for i in range(50)]
    
    # Generate the plot
    plt.figure(figsize=(6, 4))
    plt.plot(x_data, y_data, marker="o", color="b")
    plt.title("Dynamic Graph")
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.grid(True)
    
    # Save the plot to a BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    
    return send_file(buf, mimetype="image/png")

@app.route("/predictions", methods=["GET"])
def predictions():
    """
    Endpoint that returns prediction records in JSON format.
    For example: /predictions?bank_id=1&wealth=500
    # """
    # bank_id = request.args.get("bank_id", default=1, type=int)
    db_path = config.PREDICTIONS_DB_PATH_TEMPLATE.format(bank_id=3)
    
    filters = {}
    for feature in ['wealth', 'risk_tolerance', 'loyalty', 'inflation', 'gdp_growth', 'euribor']:
        value = request.args.get(feature, default=None, type=float)
        if value is not None:
            filters[feature] = value
    
    if filters:
        results = query_predictions(db_path, filters)
    else:
        results = get_all_predictions(db_path)
    
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

if __name__ == "__main__":
    # Run on port 8000 (to match your React app's URL) in debug mode
    app.run(debug=True, port=8000)
