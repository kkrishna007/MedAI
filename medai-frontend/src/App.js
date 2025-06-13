// src/App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import './App.css';


import Home from './pages/Home';
import BlindnessDetection from './pages/BlindnessDetection';
import BrainTumorClassification from './pages/BrainTumorClassification';
import PneumoniaDetection from './pages/PneumoniaDetection';

// Import components
import Header from './components/Header';
import Footer from './components/Footer';


function AppContent() {
const location = useLocation();
const isHomePage = location.pathname === '/';

return (
<div className="app-container">
{!isHomePage && <Header/>}
<main className="main-content">
<Routes>
<Route path="/" element={<Home />} />
<Route path="/blindness-detection" element={<BlindnessDetection />} />
<Route path="/brain-tumor-classification" element={<BrainTumorClassification />} />
<Route path="/pneumonia-detection" element={<PneumoniaDetection />} />
</Routes>
</main>
<Footer />
</div>
);
}

function App() {
return (
<Router>
<AppContent />
</Router>
);
}

export default App;