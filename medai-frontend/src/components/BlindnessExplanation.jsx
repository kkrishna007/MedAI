// src/components/BlindnessExplanation.jsx
import React from 'react';
import './BlindnessExplanation.css';

function BlindnessExplanation({ explanationData }) {
  if (!explanationData) return null;

  return (
    <div className="explanation-container">
      <h2>Explainable AI Analysis for Blindness Detection</h2>

      <div className="explanation-visuals">
        {explanationData.original_image && (
          <div className="explanation-visual">
            <h3>Original Fundus Image</h3>
            <img
              src={`data:image/png;base64,${explanationData.original_image}`}
              alt="Original retinal fundus"
              className="medical-image"
            />
            <p className="image-description">
              The input retinal fundus image analyzed by the AI model.
            </p>
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
              <strong>Yellow to Red regions:</strong> Highly influential areas that the model focused on while classifying the image.<br />
              <strong>Blue to Green regions:</strong> Less significant or irrelevant areas in the decision-making process.
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
              <strong>Yellow boundaries:</strong> Superpixel regions most important for the predicted DR class.<br />
              LIME highlights these regions by perturbing the image and observing the model’s response.
            </p>
          </div>
        )}

        {explanationData.shap_image && (
          <div className="explanation-visual">
            <h3>SHAP Explanation</h3>
            <img
              src={`data:image/png;base64,${explanationData.shap_image}`}
              alt="SHAP visualization"
              className="medical-image"
            />
            <p className="image-description">
              This overlay highlights the retinal regions that most influenced the model's DR severity prediction.<br />
              Brighter areas indicate stronger contribution to the diagnosis, helping to visualize which parts of the retina guided the model’s decision.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default BlindnessExplanation;
