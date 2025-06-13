// src/components/ResultDisplay.jsx
import React from 'react';
import './ResultDisplay.css';

function ResultDisplay({ title, confidence, details }) {
  return (
    <div className="result-display">
      <h3 className="result-title">{title}</h3>
      
      {confidence !== undefined && (
        <div className="confidence-meter">
          <div className="confidence-label">
            Confidence: {(confidence * 100).toFixed(2)}%
          </div>
          <div className="confidence-bar">
            <div 
              className="confidence-fill" 
              style={{ width: `${confidence * 100}%` }}
            ></div>
          </div>
        </div>
      )}
      
      {details && details.length > 0 && (
        <div className="result-details">
          {details.map((detail, index) => (
            <div key={index} className="detail-item">
              <span className="detail-label">{detail.label}:</span>
              <span className="detail-value">{detail.value}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default ResultDisplay;
