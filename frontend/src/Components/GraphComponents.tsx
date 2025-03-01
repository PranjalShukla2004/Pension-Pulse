import React from 'react';

function GraphComponent() {
  // Directly use the URL of your backend endpoint.
  // Ensure that the backend (FastAPI) is running on port 8000.
  const imageUrl = "http://localhost:8000/api/graph-image";

  return (
    <div style={{ textAlign: 'center' }}>
      <h2>Graph Rendered from Backend</h2>
      <img src={imageUrl} alt="Graph from backend" style={{ maxWidth: '100%' }} />
    </div>
  );
}

export default GraphComponent;
