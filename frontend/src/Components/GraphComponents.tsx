import React from 'react';

interface GraphComponentProps {
  plotType: string;
}

const GraphComponent: React.FC<GraphComponentProps> = ({ plotType }) => {
  // Map plot types to query parameters.
  let x_axis = '';
  let y_axis = 'predicted_rate'; // interest rate is always y-axis
  
  switch (plotType) {
    case 'ir_vs_wealth':
      x_axis = 'wealth';
      break;
    case 'ir_vs_risk':
      x_axis = 'risk_tolerance';
      break;
    case 'loyalty_vs_ir':
      x_axis = 'loyalty';
      break;
    case 'gdp_vs_ir':
      x_axis = 'gdp_growth';
      break;
    case 'ir_vs_inflation':
      x_axis = 'inflation';
      break;
    default:
      x_axis = '';
  }

  const url = `http://localhost:8000/graph-image?x_axis=${x_axis}&y_axis=${y_axis}`;


  return (
    <div style={styles.container}>
      <h2>Graph Rendered</h2>
      {x_axis ? (
        <img src={url} alt="Graph from backend" style={styles.image} />
      ) : (
        <p>Please select a plot type.</p>
      )}
    </div>
  );
};

const styles: { [key: string]: React.CSSProperties } = {
  container: { textAlign: 'center' },
  image: { maxWidth: '100%', marginTop: '20px' },
};

export default GraphComponent;
