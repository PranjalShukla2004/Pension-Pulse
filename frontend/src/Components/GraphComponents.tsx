import React from 'react';

interface GraphComponentProps {
  bankParam: string;
  customerParam: string;
}

const GraphComponent: React.FC<GraphComponentProps> = ({ bankParam, customerParam }) => {
  return (
    <div style={styles.container}>
      <h2>Graph Rendered from Backend</h2>
      <p>
        Y-axis: {bankParam ? bankParam : 'None selected'} <br />
        X-axis: {customerParam ? customerParam : 'None selected'}
      </p>
      <img
        src="http://localhost:8000/api/graph-image"
        alt="Graph from backend"
        style={styles.image}
      />
    </div>
  );
};

const styles: { [key: string]: React.CSSProperties } = {
  container: {
    textAlign: 'center',
  },
  image: {
    maxWidth: '100%',
  },
};

export default GraphComponent;
