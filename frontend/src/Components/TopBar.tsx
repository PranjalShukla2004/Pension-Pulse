import React from 'react';
import Logo from '../Images/LBS_LOGO.avif'; 

const TopBar: React.FC = () => {
  return (
    <>
      {/* Thin black bar on top */}
      <div style={styles.topStrip}></div>

      {/* Green header containing the logo and text */}
      <div style={styles.headerContainer}>
        <div style={styles.logoSection}>
          <img src={Logo} alt="Lloyds Bank Logo" style={styles.logo} />
          <span style={styles.text}>LLOYDS</span>
        </div>
      </div>
    </>
  );
};

const styles = {
  // Black strip on top
  topStrip: {
    backgroundColor: '#000000',
    height: '4px',
  },
  // Main green header
  headerContainer: {
    display: 'flex',
    alignItems: 'center',
    backgroundColor: '#007f3b', // Lloyds green
    color: '#ffffff',
    padding: '10px 20px',
    boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
  },
  // Holds the logo + text together
  logoSection: {
    display: 'flex',
    alignItems: 'center',
  },
  // Horse logo
  logo: {
    height: '40px',
    marginRight: '10px',
  },
  // Lloyds text next to the horse
  text: {
    fontSize: '1.8rem',
    fontFamily: 'Arial, sans-serif',
    fontWeight: 'bold',
    letterSpacing: '1px',
  },
};

export default TopBar;
