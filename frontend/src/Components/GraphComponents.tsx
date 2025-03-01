import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';

function GraphComponent() {
  const [graphData, setGraphData] = useState({ timestamp: '', values: [] });

  // Fetch data from the backend API
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/graph-data');
        const data = await response.json();
        setGraphData(data);
      } catch (error) {
        console.error("Error fetching graph data:", error);
      }
    };

    // Initial fetch and polling every 1 second
    fetchData();
    const interval = setInterval(fetchData, 1000);
    return () => clearInterval(interval);
  }, []);

  // Prepare chart data for Chart.js
  const chartData = {
    labels: graphData.values.map((_, index) => index + 1),
    datasets: [
      {
        label: 'Dynamic Data',
        data: graphData.values,
        fill: false,
        backgroundColor: 'rgb(75, 192, 192)',
        borderColor: 'rgba(75, 192, 192, 0.2)',
      },
    ],
  };

  return (
    <div style={{ width: '80%', margin: '0 auto' }}>
      <h2>Dynamic Graph - {graphData.timestamp && new Date(graphData.timestamp).toLocaleTimeString()}</h2>
      <Line data={chartData} />
    </div>
  );
}

export default GraphComponent;