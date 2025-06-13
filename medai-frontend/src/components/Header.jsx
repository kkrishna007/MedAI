// src/components/Header.jsx
import React from 'react';
import { Link } from 'react-router-dom';
import './Header.css';

function Header() {
  return (
    <header className="app-header">
      <div className="container">
        <Link to="/" className="logo">
          <h1>MedAI</h1>
          <span className="tagline">Comprehensive AI for Disease Detection</span>
        </Link>
        <nav className="main-nav">
          <ul>
            <li><Link to="/">Home</Link></li>
            <li><Link to="/blindness-detection">Blindness Detection</Link></li>
            <li><Link to="/brain-tumor-classification">Brain Tumor</Link></li>
            <li><Link to="/pneumonia-detection">Pneumonia</Link></li>
          </ul>
        </nav>
      </div>
    </header>
  );
}

export default Header;
