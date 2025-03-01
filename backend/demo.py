# API 
# Get : Data from Agents / Database 
# Post : Graphs to Frontend

# List of parameters to assigned to each customer
# Query the database for the parameters
# Return the parameters to the frontend
app = FastAPI()

@app.get("/name")
def read_name():
    return {"name": "Arindam"}