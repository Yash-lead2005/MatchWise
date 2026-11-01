import { useEffect, useState } from "react";
import "./App.css";

function App() {
  const [backendStatus, setBackendStatus] = useState("Checking...");

  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/health")
      .then((response) => response.json())
      .then((data) => setBackendStatus(data.status))
      .catch(() => setBackendStatus("Disconnected"));
  }, []);

  return (
    <main className="app">
      <div className="topbar">
        <h2>MatchWise</h2>
        <span className="status">
          Backend: {backendStatus}
        </span>
      </div>

      <section className="hero">
        <p className="eyebrow">INTELLIGENT ENTITY RESOLUTION</p>

        <h1>
          Find duplicates.
          <br />
          Make better data decisions.
        </h1>

        <p className="description">
          An intelligent product matching platform powered by
          machine learning and human review.
        </p>

        <button>Explore MatchWise</button>
      </section>
    </main>
  );
}

export default App;