import React from 'react';

interface CustomerParamsProps {
  selected: string;
  onChange: (value: string) => void;
}

const CustomerParams: React.FC<CustomerParamsProps> = ({ selected, onChange }) => {
  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    onChange(event.target.value);
  };

  return (
    <div style={styles.customerParameters}>
      <h3 style={styles.title}>Customer Parameters</h3>
      <label style={styles.radioLabel}>
        <input
          type="radio"
          name="customerParam"
          value="option1"
          checked={selected === 'option1'}
          onChange={handleChange}
        />
        <span>Option 1</span>
      </label>
      <label style={styles.radioLabel}>
        <input
          type="radio"
          name="customerParam"
          value="option2"
          checked={selected === 'option2'}
          onChange={handleChange}
        />
        <span>Option 2</span>
      </label>
      <label style={styles.radioLabel}>
        <input
          type="radio"
          name="customerParam"
          value="option3"
          checked={selected === 'option3'}
          onChange={handleChange}
        />
        <span>Option 3</span>
      </label>
    </div>
  );
};

const styles: { [key: string]: React.CSSProperties } = {
  customerParameters: {
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
    marginTop: '20px',
  },
  title: {
    fontSize: '1.2rem',
    marginBottom: '5px',
    fontFamily: 'Arial, sans-serif',
  },
  radioLabel: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontFamily: 'Arial, sans-serif',
  },
};

export default CustomerParams;
