import React, { useState } from 'react';

const AgeSlider = () => {
  const [age, setAge] = useState(21);

  interface HandleChangeEvent extends React.ChangeEvent<HTMLInputElement> {}

  const handleChange = (event: HandleChangeEvent) => {
    setAge(Number(event.target.value));
  };

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Select Your Age</h3>
      <input
        type="range"
        min="21"
        max="80"
        value={age}
        onChange={handleChange}
        style={styles.slider}
      />
      <div style={styles.ageDisplay}>Age: <strong>{age}</strong></div>
    </div>
  );
};

const styles = {
  container: {
    padding: '20px',
    margin: '10px',
    maxWidth: '400px',
    border: '1px solid #ccc',
    borderRadius: '10px',
    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
    textAlign: 'center' as const,
    backgroundColor: '#f9f9f9',
    flex: 1 // so it can share horizontal space
  },
  title: {
    marginBottom: '20px',
    fontFamily: 'Arial, sans-serif',
  },
  slider: {
    width: '100%',
    marginBottom: '10px',
  },
  ageDisplay: {
    fontSize: '1.2rem',
    fontFamily: 'Arial, sans-serif',
  },
};

export default AgeSlider;
