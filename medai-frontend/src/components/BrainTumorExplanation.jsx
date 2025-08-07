// src/components/BrainTumorExplanation.jsx
import React from 'react';
import './BrainTumorExplanation.css';

function BrainTumorExplanation({ explanationData }) {
  if (!explanationData) return null;

  return (
    <div className="explanation-container">
      <h2>Explainable AI Analysis for Brain Tumor Classification</h2>

      <div className="explanation-visuals">
        {explanationData.original_image && (
          <div className="explanation-visual">
            <h3>Original MRI</h3>
            <img 
              src={`data:image/png;base64,${explanationData.original_image}`} 
              alt="Original MRI" 
              className="medical-image"
            />
            <p className="image-description">The input MRI image analyzed by the AI model.</p>
          </div>
        )}

        {explanationData.gradcam_image && (
          <div className="explanation-visual">
            <h3>Grad-CAM Visualization</h3>
            <img 
              src={`data:image/png;base64,${explanationData.gradcam_image}`} 
              alt="Grad-CAM heatmap" 
              className="medical-image"
            />
            <p className="image-description">
              <strong>Yellow areas:</strong> Regions most influential in the model's tumor classification decision.<br/>
              <strong>Blue areas(if any):</strong> Less relevant regions.
            </p>
          </div>
        )}

        {explanationData.lime_image && (
          <div className="explanation-visual">
            <h3>LIME Feature Importance</h3>
            <img 
              src={`data:image/png;base64,${explanationData.lime_image}`} 
              alt="LIME visualization" 
              className="medical-image"
            />
            <p className="image-description">
              <strong>Yellow boundaries: </strong> Superpixel regions most important for the predicted tumor class.<br/>
              LIME highlights these regions by perturbing the image and observing the model’s response.
            </p>
          </div>
        )}

        {explanationData.shap_image && (
          <div className="explanation-visual">
            <h3>Attention Visualization</h3>
            <img 
              src={`data:image/png;base64,${explanationData.shap_image}`} 
              alt="SHAP visualization" 
              className="medical-image"
            />
            <p className="image-description">
              <strong>Green areas: </strong> Regions that most influence the probability of the predicted tumor type.<br/>
              SHAP highlights the importance of each pixel for the model’s decision.
            </p>
          </div>
        )}
      </div>

      {/* Add more XAI visualizations and descriptions as needed */}
    </div>
  );
}

export default BrainTumorExplanation;
