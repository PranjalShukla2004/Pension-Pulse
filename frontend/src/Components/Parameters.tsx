import React from 'react';

interface ParameterSelectorProps {
  bankParam: string;
  setBankParam: (val: string) => void;
  customerParam: string;
  setCustomerParam: (val: string) => void;
  onSubmit: () => void; // Callback to finalize changes
}

const ParameterSelector: React.FC<ParameterSelectorProps> = ({
  bankParam,
  setBankParam,
  customerParam,
  setCustomerParam,
  onSubmit,
}) => {
  const handleBankChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setBankParam(e.target.value);
  };

  const handleCustomerChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setCustomerParam(e.target.value);
  };

  return (
    <div style={styles.container}>
      <div style={styles.paramGroup}>
        <h3 style={styles.heading}>Bank Parameters</h3>
        <label style={styles.radioLabel}>
          <input
            type="radio"
            name="bankParam"
            value="alpha"
            checked={bankParam === 'alpha'}
            onChange={handleBankChange}
          />
          alpha (fixed rate)
        </label>
        <label style={styles.radioLabel}>
          <input
            type="radio"
            name="bankParam"
            value="beta"
            checked={bankParam === 'beta'}
            onChange={handleBankChange}
          />
          beta (Libor multiplier)
        </label>
        <label style={styles.radioLabel}>
          <input
            type="radio"
            name="bankParam"
            value="netProfit"
            checked={bankParam === 'netProfit'}
            onChange={handleBankChange}
          />
          net profit at some time stamp
        </label>
      </div>

      <div style={styles.paramGroup}>
        <h3 style={styles.heading}>Customer Parameters</h3>
        <label style={styles.radioLabel}>
          <input
            type="radio"
            name="customerParam"
            value="Age"
            checked={customerParam === 'Age'}
            onChange={handleCustomerChange}
          />
          Age
        </label>
        <label style={styles.radioLabel}>
          <input
            type="radio"
            name="customerParam"
            value="Income"
            checked={customerParam === 'Income'}
            onChange={handleCustomerChange}
          />
          Income
        </label>
        <label style={styles.radioLabel}>
          <input
            type="radio"
            name="customerParam"
            value="Knowledge"
            checked={customerParam === 'Knowledge'}
            onChange={handleCustomerChange}
          />
          Knowledge
        </label>
        <label style={styles.radioLabel}>
          <input
            type="radio"
            name="customerParam"
            value="Capital"
            checked={customerParam === 'Capital'}
            onChange={handleCustomerChange}
          />
          Capital
        </label>
      </div>

      <button style={styles.submitButton} onClick={onSubmit}>
        Submit
      </button>
    </div>
  );
};

const styles: { [key: string]: React.CSSProperties } = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1rem',
    backgroundColor: '#fff',
    padding: '1rem',
    borderRadius: '0.5rem',
    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
  },
  paramGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.3rem',
  },
  heading: {
    fontSize: '1.1rem',
    fontWeight: 'bold',
    marginBottom: '0.5rem',
  },
  radioLabel: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.4rem',
    fontFamily: 'Arial, sans-serif',
  },
  submitButton: {
    padding: '10px 16px',
    fontSize: '1rem',
    cursor: 'pointer',
    alignSelf: 'flex-start',
    marginTop: '0.5rem',
  },
};

export default ParameterSelector;
