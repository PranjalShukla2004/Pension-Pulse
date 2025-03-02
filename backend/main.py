import io
import os
import sqlite3
import matplotlib.pyplot as plt
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS (adjust allowed_origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use an absolute path for the database file
db_path = os.path.join(os.path.dirname(__file__), "users.db")

@app.get("/api/graph-image")
async def get_graph_image(x_axis: str = Query(...), y_axis: str = Query(...)):
    """
    GET endpoint that expects query parameters:
      /api/graph-image?x_axis=Age&y_axis=Beta

    It queries the SQLite database and returns a PNG image of the plot.
    If no y_data is found, a temporary fake y-axis is generated.
    """
    # Connect to the SQLite database using the absolute path
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Validate/sanitize input to prevent SQL injection
    allowed_customer_cols = {"Age", "Income", "Capital", "Market_Knowledge"}
    allowed_bank_cols = {"Alpha", "Beta"}

    # Fetch x_data from USERTABLE
    if x_axis in allowed_customer_cols:
        query_x = f"SELECT {x_axis} FROM USERTABLE ORDER BY UserID"
        c.execute(query_x)
        x_data = [row[0] for row in c.fetchall()]
    else:
        x_data = []

    # Fetch y_data from BANKTABLE
    y_data = []
    if y_axis in allowed_bank_cols:
        query_y = f"SELECT {y_axis} FROM BANKTABLE LIMIT 1"
        c.execute(query_y)
        row = c.fetchone()
        if row:
            y_value = row[0]
            # Create a constant y_data array for each x_data entry
            y_data = [y_value] * len(x_data)

    conn.close()

    # If no y_data is available from the database, create a temporary fake y-axis list.
    if not y_data:
        # For testing: create a fake y-axis that is a linear series (e.g., 0, 2, 4, ...)
        y_data = [i * 2 for i in range(len(x_data))]

    # Check that x_data and y_data have matching lengths
    if not x_data or len(x_data) != len(y_data):
        raise HTTPException(
            status_code=400,
            detail=f"Data error: x_data length is {len(x_data)} and y_data length is {len(y_data)}."
        )

    # Plot with matplotlib
    plt.figure(figsize=(6, 4))
    plt.plot(x_data, y_data, marker="o", linestyle="-")
    plt.title("Dynamic Graph")
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.grid(True)

    # Save the plot to a BytesIO buffer and return as PNG
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
