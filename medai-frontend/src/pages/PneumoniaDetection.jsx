// src/pages/PneumoniaDetection.jsx
import React, { useState } from 'react';
import ImageUploader from '../components/ImageUploader';
import ResultDisplay from '../components/ResultDisplay';
import PneumoniaExplanation from '../components/PneumoniaExplanation';
import { predictPneumonia } from '../utils/api';
import { FaDownload } from 'react-icons/fa';


import './ModulePage.css';

function PneumoniaDetection() {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Unified image upload handler
  const handleImageUpload = async (file) => {
    try {
      setIsLoading(true);
      setError(null);
      setResult(null);
      const response = await predictPneumonia(file);
      setResult(response);
    } catch (err) {
      setError('Error processing image. Please try again.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="module-page">
          <div className="header-with-button">
            <h1>Pneumonia Detection 🫁</h1>
            <a
              href="https://drive.google.com/drive/folders/1K3YP2rI_wZ7IR6jt9MuTc9rbyvwWgtj7?usp=sharing"
              target="_blank"
              rel="noopener noreferrer"
              className="download-button"
            >
              <FaDownload style={{ marginRight: '8px' }} />
              Download Test Dataset
            </a>
          </div>
          <p className="module-description">
            Upload a chest X-ray image to detect pneumonia using our AI model. The system will analyze the image and provide a diagnosis with explanation.
          </p>

      <div className="module-content">
        <div className="upload-section">
          <h2>Upload X-ray Image 📷</h2>
          <ImageUploader onImageUpload={handleImageUpload} />
        </div>

        <div className="result-section">
          <h2>Classification Results 📊</h2>
          {isLoading ? (
            <div className="loading-indicator">Processing image...</div>
          ) : result ? (
            <ResultDisplay
              title={`Diagnosis: ${result.result}`}
              confidence={result.confidence}
              details={[
                { label: 'Diagnosis', value: result.result },
                { label: 'Confidence', value: (result.confidence * 100).toFixed(2) + '%' }
              ]}
            />
          ) : (
            <div className="no-result">
              {error || "Upload an image to see results"}
            </div>
          )}
        </div>
      </div>

      {/* XAI Section */}
      {result && result.explanation && (
        <div className="xai-section" style={{ marginTop: "2rem" }}>
          <PneumoniaExplanation explanationData={result.explanation} />
        </div>
      )}

      {/* Info Section */}
      <div className="info-section">
        <h2>About Pneumonia Detection 🔍</h2>
        <p>
          Pneumonia is an infection that inflames the air sacs in one or both lungs. Early and accurate detection is crucial for effective treatment and better patient outcomes.
        </p>
        <h3>What is Pneumonia? 🦠</h3>
        <p>
          Pneumonia can be caused by bacteria, viruses, or fungi. It is characterized by symptoms such as cough, fever, chills, and difficulty breathing. Chest X-rays are commonly used to diagnose pneumonia by identifying lung opacities.
        </p>
        <h3>How Our AI Model Works 🤖</h3>
        <ul>
          <li>Uses a deep learning neural network trained on thousands of chest X-ray images.</li>
          <li>Automatically detects patterns associated with pneumonia and normal lungs.</li>
          <li>Employs explainable AI (XAI) techniques to highlight regions influencing the diagnosis.</li>
        </ul>
        <h3>Explainable AI in Medical Imaging 🧩</h3>
        <ul>
          <li>Grad-CAM and LIME highlight image regions most relevant to the AI's decision.</li>
          <li>SHAP provides feature attribution maps for transparency.</li>
          <li>Visual explanations help radiologists and clinicians validate AI predictions.</li>
        </ul>
        <h3>Clinical Importance ⚕️</h3>
        <ul>
          <li>Supports early diagnosis and treatment planning.</li>
          <li>Reduces diagnostic uncertainty and supports clinical decision-making.</li>
          <li>Empowers patients and clinicians with transparent AI results.</li>
        </ul>
        <h3>Disclaimer</h3>
        <p>
          <strong>This tool is for research and educational purposes only.</strong> AI results should not replace professional medical advice. Always consult a qualified healthcare provider for clinical decisions.
        </p>
      </div>
    </div>
  );
}

export default PneumoniaDetection;
