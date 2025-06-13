// src/pages/PneumoniaDetection.jsx
import React, { useState } from 'react';
import { predictPneumonia } from '../utils/api';
import PneumoniaExplanation from '../components/PneumoniaExplanation';
import './PneumoniaDetection.css';

function PneumoniaDetection() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Handle file selection
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
    }
  };
  
  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please select an image first");
      return;
    }
    
    setLoading(true);
    setError(null);
    setResult(null);
    
    try {
      const response = await predictPneumonia(file);
      setResult(response);
    } catch (err) {
      setError("Error analyzing image. Please try again.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="pneumonia-detection-container">
      <h1>Pneumonia Detection</h1>
      <p className="description">
        Upload a chest X-ray image to detect pneumonia using our AI model.
        The system will analyze the image and provide a diagnosis with explanation.
      </p>
      
      <form onSubmit={handleSubmit} className="upload-form">
        <div className="file-input-container">
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            id="file-input"
            className="file-input"
          />
          <label htmlFor="file-input" className="file-input-label">
            {file ? file.name : "Choose an X-ray image"}
          </label>
        </div>
        
        {preview && (
          <div className="image-preview">
            <img src={preview} alt="Preview" />
          </div>
        )}
        
        <button 
          type="submit" 
          className="analyze-button"
          disabled={!file || loading}
        >
          {loading ? "Analyzing..." : "Analyze X-ray"}
        </button>
      </form>
      
      {error && <div className="error-message">{error}</div>}
      
      {result && (
        <div className="result-container">
          <div className="result-header">
            <h2>Analysis Result</h2>
            <div className={`result-badge ${result.result === "Pneumonia" ? "pneumonia" : "normal"}`}>
              {result.result}
            </div>
            <div className="confidence">
              Confidence: {(result.confidence * 100).toFixed(2)}%
            </div>
          </div>
          
          {result.explanation && <PneumoniaExplanation explanationData={result.explanation} />}
        </div>
      )}
    </div>
  );
}

export default PneumoniaDetection;
