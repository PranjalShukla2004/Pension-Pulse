import { Slider, Dropdown, Button } from 'antd';
import { DownOutlined } from '@ant-design/icons';
import React, { useState } from 'react';


const dropdownOptions = [
  { value: 'option1', label: 'Option 1' },
  { value: 'option2', label: 'Option 2' },
  { value: 'option3', label: 'Option 3' },
];

function InputComponent() {
  const [selectedOption, setSelectedOption] = useState('Select an option');

  const handleMenuClick = (option: { value: string; label: string }) => {
    setSelectedOption(option.label);
  };

  return (
    <div style={{ marginBottom: '20px' }}>
      <h3>Dropdown List</h3>
      <Dropdown overlay={
        <div>
          {dropdownOptions.map(option => (
            <div key={option.value} style={{ padding: '5px' }} onClick={() => handleMenuClick(option)}>
              {option.label}
            </div>
          ))}
        </div>
      }>
        <Button>
          {selectedOption} <DownOutlined />
        </Button>
      </Dropdown>
    </div>
  );
}

export default InputComponent;