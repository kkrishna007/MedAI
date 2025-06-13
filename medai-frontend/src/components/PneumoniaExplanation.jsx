// src/components/PneumoniaExplanation.jsx
import React from 'react';
import './PneumoniaExplanation.css';

function PneumoniaExplanation({ explanationData }) {
  if (!explanationData) return null;
  
  return (
    <div className="explanation-container">
      <h2>Explainable AI Analysis for Pneumonia Detection</h2>
      
      <div className="explanation-text">
        <div className="diagnosis-summary">
          <h3>AI Diagnosis Summary</h3>
          <p>{explanationData.text}</p>
          {explanationData.xai_techniques_used && explanationData.xai_techniques_used.length > 0 && (
            <div className="xai-status">
              <strong>XAI Techniques Applied:</strong> {explanationData.xai_techniques_used.join(', ')}
            </div>
          )}
          
          {/* NEW: Medical Validation Status */}
          {explanationData.medical_validation && (
            <div className="validation-status">
              <h4>Medical Validation Results</h4>
              <div className="validation-grid">
                <div className={`validation-item ${explanationData.medical_validation.lime_anatomical_focus ? 'passed' : 'failed'}`}>
                  <span className="validation-label">LIME Anatomical Focus:</span>
                  <span className="validation-result">{explanationData.medical_validation.lime_anatomical_focus ? '✓ Passed' : '✗ Failed'}</span>
                </div>
                <div className={`validation-item ${explanationData.medical_validation.intensity_clinical_relevance ? 'passed' : 'failed'}`}>
                  <span className="validation-label">Intensity Clinical Relevance:</span>
                  <span className="validation-result">{explanationData.medical_validation.intensity_clinical_relevance ? '✓ Passed' : '✗ Failed'}</span>
                </div>
                <div className={`validation-item ${explanationData.medical_validation.attention_pattern_consistency ? 'passed' : 'failed'}`}>
                  <span className="validation-label">Attention Pattern Consistency:</span>
                  <span className="validation-result">{explanationData.medical_validation.attention_pattern_consistency ? '✓ Passed' : '✗ Failed'}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
      
      <div className="explanation-visuals">
        <div className="explanation-visual">
          <h3>Original Chest X-ray</h3>
          <img 
            src={`data:image/png;base64,${explanationData.original_image}`} 
            alt="Original chest X-ray" 
            className="medical-image"
          />
          <p className="image-description">The input chest X-ray image analyzed by the AI model.</p>
        </div>
        
        {explanationData.gradcam_image && (
          <div className="explanation-visual">
            <h3>Intensity Heat Map</h3>
            <img 
              src={`data:image/png;base64,${explanationData.gradcam_image}`} 
              alt="Intensity heatmap visualization" 
              className="medical-image"
            />
            <p className="image-description">
              <strong>Red/Yellow areas:</strong> Regions with high intensity values that influence the AI's decision.<br/>
              <strong>Blue areas:</strong> Regions with lower intensity impact on the diagnosis.<br/>
              This visualization highlights areas based on image intensity patterns relevant to pneumonia detection.
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
              <strong>Green boundaries:</strong> Superpixel regions that support the pneumonia diagnosis.<br/>
              <strong>Red boundaries:</strong> Regions that contradict the diagnosis.<br/>
              LIME creates local explanations by perturbing image regions and analyzing how changes affect the model's prediction using 1000 samples for medical accuracy.
            </p>
          </div>
        )}
        
        {explanationData.shap_image && (
          <div className="explanation-visual">
            <h3>Attention Visualization</h3>
            <img 
              src={`data:image/png;base64,${explanationData.shap_image}`} 
              alt="Attention visualization" 
              className="medical-image"
            />
            <p className="image-description">
              <strong>Purple/Dark areas:</strong> Low feature activity with minimal diagnostic relevance.<br/>
              <strong>Green/Yellow areas:</strong> Moderate feature convergence indicating areas of interest.<br/>
              <strong>Bright Yellow areas:</strong> High feature activity where intensity, edge, and texture patterns strongly converge.<br/>
              This visualization uses the VIRIDIS colormap to show regions where multiple image analysis techniques identify potential diagnostic features.
            </p>
          </div>
        )}
      </div>

      {/* NEW: Comprehensive XAI Report Section */}
      {explanationData.comprehensive_report && (
        <div className="comprehensive-report">
          <h3>Comprehensive XAI Analysis Report</h3>
          <div className="report-grid">
            <div className="report-section">
              <h4>Primary Diagnosis</h4>
              <p className="diagnosis-result">{explanationData.comprehensive_report.primary_diagnosis}</p>
              <p className="confidence-level">Confidence: {(explanationData.comprehensive_report.confidence_level * 100).toFixed(1)}%</p>
            </div>
            
            {explanationData.comprehensive_report.clinical_interpretation && (
              <div className="report-section">
                <h4>Clinical Interpretation</h4>
                {Object.entries(explanationData.comprehensive_report.clinical_interpretation).map(([key, value]) => (
                  <p key={key} className="interpretation-item">
                    <strong>{key.replace('_', ' ').toUpperCase()}:</strong> {value}
                  </p>
                ))}
              </div>
            )}
            
            {explanationData.comprehensive_report.risk_assessment && (
              <div className="report-section">
                <h4>Risk Assessment</h4>
                {Object.entries(explanationData.comprehensive_report.risk_assessment).map(([key, value]) => (
                  <p key={key} className="risk-item">
                    <strong>{key.replace('_', ' ').toUpperCase()}:</strong> {value}
                  </p>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* NEW: Interactive Dashboard Section */}
      {explanationData.interactive_dashboard && (
        <div className="interactive-dashboard">
          <h3>XAI Quality Metrics</h3>
          <div className="metrics-grid">
            {explanationData.interactive_dashboard.quality_metrics && (
              <>
                <div className="metric-item">
                  <span className="metric-label">LIME Sample Count:</span>
                  <span className="metric-value">{explanationData.interactive_dashboard.quality_metrics.lime_sample_count}</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Processing Time:</span>
                  <span className="metric-value">{explanationData.interactive_dashboard.quality_metrics.processing_time}</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Anatomical Focus:</span>
                  <span className={`metric-value ${explanationData.interactive_dashboard.quality_metrics.anatomical_focus ? 'success' : 'warning'}`}>
                    {explanationData.interactive_dashboard.quality_metrics.anatomical_focus ? '✓ Achieved' : '⚠ Needs Review'}
                  </span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Clinical Consistency:</span>
                  <span className={`metric-value ${explanationData.interactive_dashboard.quality_metrics.clinical_consistency ? 'success' : 'warning'}`}>
                    {explanationData.interactive_dashboard.quality_metrics.clinical_consistency ? '✓ Consistent' : '⚠ Inconsistent'}
                  </span>
                </div>
              </>
            )}
          </div>
        </div>
      )}
      
      <div className="technical-details">
        <h3>XAI Techniques Applied</h3>
        <div className="tech-grid">
          <div className="tech-item">
            <h4>LIME (Real XAI)</h4>
            <p>Local Interpretable Model-agnostic Explanations creates local explanations by perturbing superpixel regions and observing prediction changes. This is a genuine research-grade XAI technique.</p>
          </div>
          <div className="tech-item">
            <h4>Intensity Analysis</h4>
            <p>Analyzes pixel intensity patterns and applies Gaussian smoothing to highlight regions with brightness characteristics typical of pneumonia consolidations.</p>
          </div>
          <div className="tech-item">
            <h4>Attention Mapping</h4>
            <p>Combines multiple image analysis techniques including intensity, edge detection, and texture analysis to create comprehensive feature maps weighted by prediction confidence.</p>
          </div>
        </div>
      </div>
      
      <div className="clinical-relevance">
        <h3>Clinical Interpretation</h3>
        <div className="clinical-content">
          <div className="clinical-section">
            <h4>Pneumonia Indicators</h4>
            <ul>
              <li><strong>Consolidation:</strong> Dense, white areas in lung fields</li>
              <li><strong>Interstitial patterns:</strong> Reticular or nodular opacities</li>
              <li><strong>Air bronchograms:</strong> Air-filled bronchi visible within consolidated areas</li>
              <li><strong>Pleural effusion:</strong> Fluid accumulation in pleural space</li>
            </ul>
          </div>
          <div className="clinical-section">
            <h4>XAI for Medical Decision Support</h4>
            <p>
              The explainable AI techniques provide complementary perspectives on the model's decision-making process. 
              LIME provides research-grade explanations, while intensity analysis and attention mapping offer clinically relevant visualizations.
              These visualizations help radiologists understand and validate the AI's reasoning for pneumonia detection.
            </p>
          </div>
        </div>
      </div>
      
      <div className="xai-methodology">
        <h3>XAI Methodology</h3>
        <div className="methodology-content">
          <div className="method-section">
            <h4>Hybrid XAI Approach</h4>
            <p>
              This implementation combines real XAI (LIME) with clinically-focused visualization techniques. 
              LIME provides genuine explainable AI following research standards, while intensity and attention mapping offer medical-specific insights.
            </p>
          </div>
          <div className="method-section">
            <h4>Medical Validation</h4>
            <p>
              All XAI outputs undergo medical validation to ensure anatomical relevance and clinical consistency. 
              The validation results help assess the reliability of the explanations for medical decision support.
            </p>
          </div>
        </div>
      </div>
      
      <div className="disclaimer">
        <h3>Important Notice</h3>
        <p>
          <strong>This AI analysis with explainable AI is for research and educational purposes only.</strong> 
          The XAI visualizations are intended to provide insights into the model's decision-making process but should not replace clinical judgment. 
          Always consult qualified healthcare professionals for medical decisions.
        </p>
      </div>
    </div>
  );
}

export default PneumoniaExplanation;
