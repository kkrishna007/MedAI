// src/pages/BlindnessDetection.jsx
import React, { useState } from 'react';
import ImageUploader from '../components/ImageUploader';
import ResultDisplay from '../components/ResultDisplay';
import { predictBlindness } from '../utils/api';
import './ModulePage.css';

function BlindnessDetection() {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleImageUpload = async (file) => {
    try {
      setIsLoading(true);
      setError(null);
      const response = await predictBlindness(file);
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
      <h1>Diabetic Retinopathy Detection 👁️</h1>
      <p className="module-description">
        Upload a retinal fundus image to detect diabetic retinopathy and assess its severity.
      </p>
      
      <div className="module-content">
        <div className="upload-section">
          <h2>Upload Retinal Image 📸</h2>
          <ImageUploader onImageUpload={handleImageUpload} />
        </div>
        
        <div className="result-section">
          <h2>Detection Results 📊</h2>
          {isLoading ? (
            <div className="loading-indicator">Processing image...</div>
          ) : result ? (
            <ResultDisplay 
              title={`Diagnosis: ${result.severity}`}
              confidence={result.confidence}
              details={[
                { label: 'Severity Level', value: result.prediction },
                { label: 'Condition', value: result.severity },
              ]}
            />
          ) : (
            <div className="no-result">
              {error || "Upload an image to see results"}
            </div>
          )}
        </div>
      </div>
      
      <div className="info-section">
        <h2>About Diabetic Retinopathy 🔍</h2>
        <p>
          Diabetic retinopathy (DR) is a diabetes complication that affects the eyes. It's caused by damage to the blood vessels of the light-sensitive tissue at the back of the eye (retina). As one of the leading causes of vision loss worldwide, early detection is critical for preventing irreversible damage.
        </p>
        
        <h3>How It Develops 📈</h3>
        <p>
          In patients with diabetes, prolonged high blood sugar levels can damage the tiny blood vessels in the retina, causing them to leak fluid or bleed. As the disease progresses, new abnormal blood vessels may grow on the retina's surface, which can lead to serious vision problems.
        </p>
        
        <h3>Classification System 🏥</h3>
        <p>
          Our AI model classifies diabetic retinopathy into five stages (0-4):
        </p>
        <ul>
          <li><strong>No DR (Stage 0):</strong> No visible signs of diabetic retinopathy</li>
          <li><strong>Mild DR (Stage 1):</strong> Small microaneurysms (tiny bulges in blood vessels) may be present</li>
          <li><strong>Moderate DR (Stage 2):</strong> More microaneurysms, dot and blot hemorrhages, and hard exudates</li>
          <li><strong>Severe DR (Stage 3):</strong> More than 20 hemorrhages in each quadrant, venous beading, or intraretinal microvascular abnormalities (IRMA)</li>
          <li><strong>Proliferative DR (Stage 4):</strong> Abnormal blood vessel growth, possible retinal detachment</li>
        </ul>
        
        <h3>The Importance of Screening ⚠️</h3>
        <p>
          Regular screening is essential for all individuals diagnosed with diabetes, regardless of type. Early detection through regular eye examinations can prevent up to 90% of cases of diabetes-related blindness. Our AI system, based on DenseNet121 architecture trained on the APTOS 2019 dataset, can detect and classify DR with high accuracy, making screening more accessible.
        </p>
        
        <h3>Recent Advances 🚀</h3>
        <p>
          Recent advancements in AI-based detection systems have achieved expert-level accuracy in analyzing retinal images. For example, DeepDR, a deep learning system, has demonstrated excellent performance in detecting various retinal lesions associated with DR, including microaneurysms, cotton-wool spots, hard exudates, and hemorrhages.
        </p>
        <p>
          Our system builds on these advances to provide a reliable, accessible tool for diabetic retinopathy screening, potentially transforming early detection and intervention for this serious condition.
        </p>
      </div>
    </div>
  );
}

export default BlindnessDetection;
