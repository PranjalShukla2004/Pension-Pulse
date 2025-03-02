import React from 'react';

interface ParameterSelectorProps {
  plotType: string;
  setPlotType: (val: string) => void;
  onSubmit: () => void; // Callback to finalize changes
}

const ParameterSelector: React.FC<ParameterSelectorProps> = ({
  plotType,
  setPlotType,
  onSubmit,
}) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setPlotType(e.target.value);
  };

  return (
    <div style={styles.container}>
      <h3 style={styles.heading}>Select Plot Type</h3>
      <div style={styles.paramGroup}>
        <label style={styles.radioLabel}>
          <input
            type="radio"
            name="plotType"
            value="ir_vs_wealth"
            checked={plotType === 'ir_vs_wealth'}
            onChange={handleChange}
          />
          Interest Rate vs. Wealth
        </label>
        <label style={styles.radioLabel}>
          <input
            type="radio"
            name="plotType"
            value="ir_vs_risk"
            checked={plotType === 'ir_vs_risk'}
            onChange={handleChange}
          />
          Interest Rate vs. Risk
        </label>
        <label style={styles.radioLabel}>
          <input
            type="radio"
            name="plotType"
            value="loyalty_vs_ir"
            checked={plotType === 'loyalty_vs_ir'}
            onChange={handleChange}
          />
          Loyalty vs. Interest Rate
        </label>
        <label style={styles.radioLabel}>
          <input
            type="radio"
            name="plotType"
            value="gdp_vs_ir"
            checked={plotType === 'gdp_vs_ir'}
            onChange={handleChange}
          />
          GDP vs. Interest Rate
        </label>
        <label style={styles.radioLabel}>
          <input
            type="radio"
            name="plotType"
            value="ir_vs_inflation"
            checked={plotType === 'ir_vs_inflation'}
            onChange={handleChange}
          />
          Interest Rate vs. Inflation
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
  heading: {
    fontSize: '1.1rem',
    fontWeight: 'bold',
    marginBottom: '0.5rem',
  },
  paramGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.3rem',
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
