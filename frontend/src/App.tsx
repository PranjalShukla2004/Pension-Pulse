import React, { useState } from 'react';
import TopBar from './Components/TopBar';
import AgeSlider from './Components/SliderAge';
import IncomeInput from './Components/TextInput';
import ParameterSelector from './Components/Parameters';
import GraphComponent from './Components/GraphComponents';

function App() {
  // "Live" states that track the current radio selections
  const [bankParam, setBankParam] = useState('');
  const [customerParam, setCustomerParam] = useState('');

  // "Submitted" states that define what's actually passed to the Graph
  const [submittedBankParam, setSubmittedBankParam] = useState('');
  const [submittedCustomerParam, setSubmittedCustomerParam] = useState('');

  // Handle the user clicking Submit in ParameterSelector
  const handleSubmit = () => {
    setSubmittedBankParam(bankParam);
    setSubmittedCustomerParam(customerParam);
  };

  return (
    <div style={styles.app}>
      <TopBar />
      <div style={styles.mainContainer}>
        {/* Left box: Filtered Search */}
        <div style={styles.leftBox}>
          <h2 style={styles.leftBoxTitle}>Filtered Search</h2>
          <div style={styles.boxContent}>
            {/* Restored slider and text input */}
            <AgeSlider />
            <IncomeInput />

            {/* Combined parameter selector + submit button */}
            <ParameterSelector
              bankParam={bankParam}
              setBankParam={setBankParam}
              customerParam={customerParam}
              setCustomerParam={setCustomerParam}
              onSubmit={handleSubmit}
            />
          </div>
        </div>

        {/* Right box: Graph (uses submitted parameters only) */}
        <div style={styles.rightBox}>
          <GraphComponent
            bankParam={submittedBankParam}
            customerParam={submittedCustomerParam}
          />
        </div>
      </div>
    </div>
  );
}

const styles: { [key: string]: React.CSSProperties } = {
  app: {
    display: 'flex',
    flexDirection: 'column',
    width: '100vw',
    height: '100vh',
    margin: 0,
    padding: 0,
    boxSizing: 'border-box',
  },
  mainContainer: {
    display: 'flex',
    flex: 1,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    padding: '20px',
    boxSizing: 'border-box',
  },
  leftBox: {
    display: 'flex',
    flexDirection: 'column',
    width: '30%',
    backgroundColor: '#f9f9f9',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
    padding: '20px',
  },
  leftBoxTitle: {
    marginBottom: '10px',
    fontSize: '1.5rem',
    fontFamily: 'Arial, sans-serif',
  },
  boxContent: {
    display: 'flex',
    flexDirection: 'column',
    gap: '20px',
  },
  rightBox: {
    display: 'flex',
    flexDirection: 'column',
    width: '65%',
    backgroundColor: '#f9f9f9',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
    padding: '20px',
  },
};

export default App;
