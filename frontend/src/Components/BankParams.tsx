import React from 'react';

interface BankParamsProps {
  selected: string;
  onChange: (value: string) => void;
}

const BankParams: React.FC<BankParamsProps> = ({ selected, onChange }) => {
  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    onChange(event.target.value);
  };

  return (
    <div style={styles.bankParameters}>
      <h3 style={styles.title}>Bank Parameters</h3>
      <label style={styles.radioLabel}>
        <input
          type="radio"
          name="bankParam"
          value="alpha"
          checked={selected === 'alpha'}
          onChange={handleChange}
        />
        <span>alpha (fixed rate)</span>
      </label>
      <label style={styles.radioLabel}>
        <input
          type="radio"
          name="bankParam"
          value="beta"
          checked={selected === 'beta'}
          onChange={handleChange}
        />
        <span>beta (Libor multiplier)</span>
      </label>
      <label style={styles.radioLabel}>
        <input
          type="radio"
          name="bankParam"
          value="netProfit"
          checked={selected === 'netProfit'}
          onChange={handleChange}
        />
        <span>net profit at some time stamp</span>
      </label>
    </div>
  );
};

const styles: { [key: string]: React.CSSProperties } = {
  bankParameters: {
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
    marginTop: '10px',
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

export default BankParams;
