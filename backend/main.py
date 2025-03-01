from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime
import random

app = FastAPI()

# Allow React's development server to access this API
origins = [
    "http://localhost",
    "http://localhost:3000",
    # Add more origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/graph-data")
async def get_graph_data():
    """
    This endpoint simulates dynamic data for the graph.
    Replace the dummy data with real data from your simulation or database.
    """
    data = {
        "timestamp": datetime.now().isoformat(),
        "values": [random.randint(10, 100) for _ in range(10)]  # Example data points
    }
    return data

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
