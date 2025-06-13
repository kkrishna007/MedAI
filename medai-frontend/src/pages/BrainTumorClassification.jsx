// src/pages/BrainTumorClassification.jsx
import React, { useState } from 'react';
import ImageUploader from '../components/ImageUploader';
import ResultDisplay from '../components/ResultDisplay';
import { predictBrainTumor } from '../utils/api';
import './ModulePage.css';

function BrainTumorClassification() {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleImageUpload = async (file) => {
    try {
      setIsLoading(true);
      setError(null);
      const response = await predictBrainTumor(file);
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
      <h1>Brain Tumor Classification 🧠</h1>
      <p className="module-description">
        Upload a brain MRI image to detect and classify brain tumors.
      </p>
      
      <div className="module-content">
        <div className="upload-section">
          <h2>Upload MRI Image 📷</h2>
          <ImageUploader onImageUpload={handleImageUpload} />
        </div>
        
        <div className="result-section">
          <h2>Classification Results 📊</h2>
          {isLoading ? (
            <div className="loading-indicator">Processing image...</div>
          ) : result ? (
            <ResultDisplay 
              title={`Diagnosis: ${result.tumor_type}`}
              confidence={result.confidence}
              details={[
                { label: 'Tumor Type', value: result.tumor_type },
                { label: 'Class ID', value: result.prediction }
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
        <h2>About Brain Tumor Classification 🔍</h2>
        <p>
          Brain tumors are abnormal growths of cells in the brain that can be cancerous (malignant) or non-cancerous (benign). 
          Early and accurate detection is critical for effective treatment planning and improving patient outcomes.
        </p>
        
        <h3>Types of Brain Tumors 🔬</h3>
        <p>
          Our AI model classifies brain MRIs into four distinct categories:
        </p>
        <ul>
          <li><strong>Glioma:</strong> Develops from glial cells and can occur in the brain and spinal cord. Gliomas can range from low-grade (slow-growing) to high-grade (aggressive) forms.</li>
          <li><strong>Meningioma:</strong> Forms in the meninges, the membranes that enclose the brain and spinal cord. These tumors are more common in women than men and are typically benign.</li>
          <li><strong>Pituitary Tumor:</strong> Grows on the pituitary gland at the base of the brain. These tumors can affect hormone production and vision.</li>
          <li><strong>No Tumor:</strong> Normal brain tissue with no detectable tumor presence.</li>
        </ul>
        
        <h3>Our AI Approach 🤖</h3>
        <p>
          This module employs an EfficientNetB1 architecture trained on a comprehensive dataset of brain MRI scans. The model has been fine-tuned to achieve high accuracy in distinguishing between different tumor types.
        </p>
        <p>
          Key features of our approach include:
        </p>
        <ul>
          <li>Transfer learning using pre-trained weights from ImageNet</li>
          <li>Specialized preprocessing to isolate brain regions</li>
          <li>High validation accuracy across all four classification categories</li>
          <li>Robust performance verified through multiple testing scenarios</li>
        </ul>
        
        <h3>Clinical Importance ⚕️</h3>
        <p>
          Accurate classification of brain tumors is essential for:
        </p>
        <ul>
          <li>Timely diagnosis and treatment planning</li>
          <li>Reducing the need for invasive diagnostic procedures</li>
          <li>Assisting radiologists and neurologists in clinical decision-making</li>
          <li>Potentially saving lives through early detection</li>
        </ul>
        
        <h3>Technology Behind the Model 💻</h3>
        <p>
          Our model uses convolutional neural networks (CNNs) to analyze patterns in MRI images that may not be immediately visible to the human eye. The EfficientNetB1 architecture was selected for its balance of computational efficiency and diagnostic accuracy.
        </p>
        <p>
          When you upload an MRI image, the system:
        </p>
        <ol>
          <li>Preprocesses the image (resizing to 224×224 pixels and normalizing)</li>
          <li>Extracts features using deep convolutional layers</li>
          <li>Classifies the image into one of the four categories</li>
          <li>Provides a confidence score for the diagnosis</li>
        </ol>
      </div>
    </div>
  );
}

export default BrainTumorClassification;
