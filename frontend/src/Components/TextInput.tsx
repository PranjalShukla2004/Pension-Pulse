import { Input, Button } from 'antd';
import React, { useState } from 'react';

function InputComponent() {
  const [income, setIncome] = useState('');

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setIncome(e.target.value);
  };

  const handleSubmit = () => {
    console.log("Income entered:", income); // Handle the income value here
  };

  return (
    <div style={{ width: '100%', textAlign: 'center' }}>
      <h3 style={{ marginBottom: '10px', color: '#333', fontWeight: 'bold' }}>Income</h3> {/* Updated heading */}
      <div style={{ marginBottom: '20px' }}>
        <Input
          type="number" // Allow only numeric input
          value={income}
          onChange={handleInputChange}
          placeholder="Enter your income"
          style={styles.input}
        />
      </div>
      <Button onClick={handleSubmit} style={styles.button}>
        Submit
      </Button>
    </div>
  );
}

// Styling for input box and button
const styles = {
  input: {
    width: '100%',
    maxWidth: '300px',
    marginBottom: '10px',
    padding: '10px',
    fontSize: '16px',
  },
  button: {
    backgroundColor: '#007a33', // Green button
    color: '#fff',
    borderRadius: '1px',
    padding: '10px 20px',
    border: 'none',
    fontWeight: 'bold',
    fontSize: '14px',
    cursor: 'pointer',
  },
};
export default InputComponent;