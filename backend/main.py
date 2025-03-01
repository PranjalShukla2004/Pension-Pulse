import io
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import matplotlib.pyplot as plt

app = FastAPI()

@app.get("/api/graph-image")
async def get_graph_image():
    # Create a sample graph using matplotlib
    plt.figure(figsize=(6, 4))
    x = [1, 2, 3, 4, 5]
    y = [10, 20, 15, 30, 25]
    plt.plot(x, y, marker="o")
    plt.title("Sample Graph")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()  # Close the figure to free up memory
    buf.seek(0)
    
    # Return the image as a StreamingResponse with content type image/png
    return StreamingResponse(buf, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
