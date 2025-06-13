// src/components/ImageUploader.jsx
import React, { useState, useRef } from 'react';
import './ImageUploader.css';

function ImageUploader({ onImageUpload }) {
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      processFile(file);
    }
  };

  const processFile = (file) => {
    // Create preview
    const reader = new FileReader();
    reader.onload = () => {
      setPreviewUrl(reader.result);
      onImageUpload(file);
    };
    reader.readAsDataURL(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files.length) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  return (
    <div className="image-uploader">
      <div 
        className={`upload-area ${isDragging ? 'dragging' : ''} ${previewUrl ? 'has-preview' : ''}`}
        onClick={triggerFileInput}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {previewUrl ? (
          <img src={previewUrl} alt="Preview" className="image-preview" />
        ) : (
          <div className="upload-prompt">
            <i className="upload-icon">📁</i>
            <p>Click or drag image to upload</p>
            <span className="upload-hint">Supported formats: JPG, PNG</span>
          </div>
        )}
      </div>
      <input 
        type="file" 
        ref={fileInputRef}
        onChange={handleFileChange}
        accept="image/*"
        className="file-input"
      />
      {previewUrl && (
        <button className="reset-button" onClick={() => {
          setPreviewUrl(null);
          fileInputRef.current.value = '';
        }}>
          Reset
        </button>
      )}
    </div>
  );
}

export default ImageUploader;
