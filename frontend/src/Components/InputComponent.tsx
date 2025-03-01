import React from 'react';
import { Slider, Dropdown, Button } from 'antd';
import { DownOutlined } from '@ant-design/icons';

const marks = {
  0: '0',
  10: '10',
  20: '20',
  30: '30',
  40: '40',
  50: '50',
  60: '60',
  70: '70',
  80: '80',
  90: '90',
  100: '100',
};

const dropdownOptions = [
  { value: 'option1', label: 'Option 1' },
  { value: 'option2', label: 'Option 2' },
  { value: 'option3', label: 'Option 3' },
];

function InputComponent() {
  return (
    <div style={{ textAlign: 'center' }}>
      <h2>Simulation Parameters</h2>
      <div style={{ marginBottom: '20px' }}>
        <h3>Slider 1</h3>
        <Slider marks={marks} defaultValue={50} />
      </div>
      <div style={{ marginBottom: '20px' }}>
        <h3>Slider 2</h3>
        <Slider marks={marks} defaultValue={30} />
      </div>
      <div style={{ marginBottom: '20px' }}>
        <h3>Dropdown List</h3>
        <Dropdown overlay={
          <div>
            {dropdownOptions.map(option => (
              <div key={option.value} style={{ padding: '5px' }}>{option.label}</div>
            ))}
          </div>
        }>
          <Button>
            Select an option <DownOutlined />
          </Button>
        </Dropdown>
      </div>
    </div>
  );
}

export default InputComponent;