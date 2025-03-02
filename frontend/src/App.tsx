import React, { useState } from 'react';
import TopBar from './Components/TopBar';
import ParameterSelector from './Components/Parameters';
import GraphComponent from './Components/GraphComponents';

function App() {
  // Use one state for plot type
  const [plotType, setPlotType] = useState('');

  const [submittedPlotType, setSubmittedPlotType] = useState('');

  const handleSubmit = () => {
    setSubmittedPlotType(plotType);
  };

  return (
    <div style={styles.app}>
      <TopBar />
      <div style={styles.mainContainer}>
        <div style={styles.leftBox}>
          <h2 style={styles.leftBoxTitle}>Filtered Search</h2>
          <div style={styles.boxContent}>
            <ParameterSelector
              onSubmit={handleSubmit}
              plotType={plotType}
              setPlotType={setPlotType}
            />
          </div>
        </div>
        <div style={styles.rightBox}>
          <GraphComponent plotType={submittedPlotType} />
        </div>
      </div>
    </div>
  );
}

const styles: { [key: string]: React.CSSProperties } = {
  // same as before...
};

export default App;
