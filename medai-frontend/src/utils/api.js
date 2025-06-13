// src/utils/api.js
const API_BASE_URL = 'http://localhost:8000';

export const predictBlindness = async (file) => {
  return uploadImage(file, 'blindness');
};

export const predictBrainTumor = async (file) => {
  return uploadImage(file, 'brain-tumor');
};

export const predictPneumonia = async (file) => {
  return uploadImage(file, 'pneumonia');
};

// Helper function for image upload
const uploadImage = async (file, endpoint) => {
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    const response = await fetch(`${API_BASE_URL}/predict/${endpoint}`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error uploading image:', error);
    throw error;
  }
};
