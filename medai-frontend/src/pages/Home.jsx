import React from 'react';
import { Link } from 'react-router-dom';
import './Home.css';

function Home() {
  const modules = [
    {
      id: 'blindness',
      title: 'Blindness Detection',
      description: 'Detect diabetic retinopathy stages from retinal fundus images',
      icon: '👁️',
      path: '/blindness-detection'
    },
    {
      id: 'brain-tumor',
      title: 'Brain Tumor Classification',
      description: 'Classify brain tumors from MRI scans into four categories',
      icon: '🧠',
      path: '/brain-tumor-classification'
    },
    {
      id: 'pneumonia',
      title: 'Pneumonia Detection',
      description: 'Detect pneumonia from chest X-ray images',
      icon: '🫁',
      path: '/pneumonia-detection'
    }
  ];

  return (
     <div className="home-container">
      <video autoPlay loop muted className="bg-video">
        <source src={process.env.PUBLIC_URL + '/bgvideo.mp4'} type="video/mp4" />
      </video>
      
      <section className="hero-section">
        <h1>MedAI: Comprehensive AI for Disease Detection</h1>
        <p>Advanced deep learning models to assist medical diagnosis</p>
      </section>
      
      <section className="modules-section">
        <h2>Select a Detection Module</h2>
        <div className="modules-grid">
          {modules.map(module => (
            <Link to={module.path} key={module.id} className="module-card">
              <div className="module-icon">{module.icon}</div>
              <h3>{module.title}</h3>
              <p>{module.description}</p>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}

export default Home;
