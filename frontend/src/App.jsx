import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [data, setData] = useState('');

  // Testing connection to backend
  useEffect(() => {
    fetch('http://localhost:5000/api/ping') // Fetch from backend
      .then((res) => res.json())
      .then((data) => setData(data.message));
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <p>
          {data ? data : 'Loading...'}
        </p>
      </header>
    </div>
  );
}

export default App;