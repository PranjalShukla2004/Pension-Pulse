import React from 'react';

interface GraphComponentProps {
  bankParam: string;
  customerParam: string;
}

const GraphComponent: React.FC<GraphComponentProps> = ({ bankParam, customerParam }) => {
  // Construct URL using GET query parameters
  const url = `http://localhost:8000/api/graph-image?x_axis=${customerParam}&y_axis=${bankParam}`;

  return (
    <div style={styles.container}>
      <h2>Graph Rendered</h2>
      <p>
        Y-axis: {bankParam || 'None selected'} <br />
        X-axis: {customerParam || 'None selected'}
      </p>
      {bankParam && customerParam ? (
        <img src={url} alt="Graph from backend" style={styles.image} />
      ) : (
        <p>Please select parameters.</p>
      )}
    </div>
  );
};

const styles: { [key: string]: React.CSSProperties } = {
  container: {
    textAlign: 'center',
  },
  image: {
    maxWidth: '100%',
    marginTop: '20px',
  },
};

export default GraphComponent;
