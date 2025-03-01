import React, { useState } from 'react';
import TopBar from './Components/TopBar';
import AgeSlider from './Components/SliderAge';
import IncomeInput from './Components/TextInput';
import GraphComponent from './Components/GraphComponents';
import BankParams from './Components/BankParams';
import CustomerParams from './Components/CustomerParams';

function App() {
  const [bankParam, setBankParam] = useState<string>('');
  const [customerParam, setCustomerParam] = useState<string>('');

  return (
    <div style={styles.app}>
      <TopBar />
      <div style={styles.mainContainer}>
        {/* Left box: Filtered Search */}
        <div style={styles.leftBox}>
          <h2 style={styles.leftBoxTitle}>Filtered Search</h2>
          <div style={styles.boxContent}>
            <AgeSlider />
            <IncomeInput />
            <BankParams selected={bankParam} onChange={setBankParam} />
            <CustomerParams selected={customerParam} onChange={setCustomerParam} />
          </div>
        </div>
        {/* Right box: Graph */}
        <div style={styles.rightBox}>
          <GraphComponent bankParam={bankParam} customerParam={customerParam} />
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
