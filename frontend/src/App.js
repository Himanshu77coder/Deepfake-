import React, { useState, useRef } from 'react';
import { Upload, Shield, CheckCircle, XCircle, Info, Cpu, Activity, Database, Zap, AlertTriangle, Lock, Scan } from 'lucide-react';
import './App.css';
import { API_BASE_URL, ANALYZE_ENDPOINT, HEALTH_ENDPOINT, MAX_UPLOAD_MB } from './config';

export default function DeepfakeDetector() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const [result, setResult] = useState(null);
  const [backendStatus, setBackendStatus] = useState('checking');
  const videoRef = useRef(null);

  // Check backend health on component mount
  React.useEffect(() => {
    checkBackendHealth();
  }, []);

  const checkBackendHealth = async () => {
    try {
      const response = await fetch(HEALTH_ENDPOINT);
      if (response.ok) {
        setBackendStatus('connected');
      } else {
        setBackendStatus('error');
      }
    } catch (error) {
      setBackendStatus('error');
    }
  };

  const handleFileUpload = (e) => {
    const uploadedFile = e.target.files[0];
    if (!uploadedFile) return;

    const maxSize = MAX_UPLOAD_MB * 1024 * 1024;
    if (uploadedFile.size > maxSize) {
      alert(`File size exceeds ${MAX_UPLOAD_MB}MB. Please upload a smaller file.`);
      return;
    }

    setFile(uploadedFile);
    setResult(null);

    const reader = new FileReader();
    reader.onload = (event) => {
      setPreview(event.target.result);
    };
    reader.readAsDataURL(uploadedFile);
  };

  const simulateProgress = (steps) => {
    return new Promise((resolve) => {
      let currentProgress = 0;
      const stepDuration = 500;
      const stepIncrement = 100 / steps.length;
      
      const interval = setInterval(() => {
        if (currentProgress >= 90) {
          clearInterval(interval);
          resolve();
          return;
        }
        
        const stepIndex = Math.floor(currentProgress / stepIncrement);
        if (stepIndex < steps.length) {
          setCurrentStep(steps[stepIndex]);
        }
        
        currentProgress += stepIncrement;
        setProgress(Math.min(currentProgress, 90));
      }, stepDuration);
    });
  };

const handleAnalyze = async () => {
  if (!file) return;

  setAnalyzing(true);
  setResult(null);
  setProgress(0);
  setCurrentStep('Initializing security scan...');

  const steps = [
    'Uploading to secure server...',
    'Scanning for facial features...',
    'Analyzing frequency patterns...',
    'Running neural network ensemble...',
    'Checking compression artifacts...',
    'Verifying lighting consistency...',
    'Finalizing security analysis...'
  ];

  try {
    const progressPromise = simulateProgress(steps);

    const formData = new FormData();
    formData.append('file', file);

    console.log('Sending request to:', ANALYZE_ENDPOINT);
    console.log('File:', file.name, file.type, file.size);

    const response = await fetch(ANALYZE_ENDPOINT, {
      method: 'POST',
      body: formData
    });

    console.log('Response status:', response.status);
    console.log('Response ok:', response.ok);

    // Check response status
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Backend error:', errorText);
      throw new Error(`Backend error (${response.status}): ${errorText}`);
    }

    // Parse JSON response
    const data = await response.json();
    console.log('Received data:', data);

    // Validate response data
    if (!data || typeof data.is_deepfake === 'undefined') {
      console.error('Invalid response format:', data);
      throw new Error('Invalid response format from backend. Expected is_deepfake property.');
    }

    await progressPromise;
    
    setProgress(100);
    setCurrentStep('Analysis complete!');

    await new Promise(resolve => setTimeout(resolve, 500));

    // Set result with proper fallbacks
    setResult({
      is_deepfake: Boolean(data.is_deepfake),
      is_manipulated: Boolean(data.is_manipulated ?? data.is_deepfake),
      deepfake_score: Number(data.deepfake_score || data.deepfakeScore || 0),
      manipulation_score: Number(data.manipulation_score || data.deepfake_score || data.deepfakeScore || 0),
      authenticity_score: Number(
        data.authenticity_score ?? Math.max(0, 100 - Number(data.manipulation_score || data.deepfake_score || data.deepfakeScore || 0))
      ),
      confidence: Number(data.confidence || 0),
      risk_level: String(data.risk_level || data.riskLevel || 'LOW'),
      manipulation_type: String(
        data.manipulation_type || (data.is_deepfake ? 'AI_GENERATED' : 'AUTHENTIC')
      ),
      summary: String(data.summary || ''),
      reasons: Array.isArray(data.reasons) ? data.reasons : [],
      signal_scores: data.signal_scores || {},
      analysis_details: data.analysis_details || {},
      neuralNetworks: data.neuralNetworks || data.neural_networks || {},
      frame_analysis: data.frame_analysis || null
    });

    console.log('Analysis complete successfully');

  } catch (error) {
    console.error('Analysis error:', error);
    
    setCurrentStep('Analysis failed');
    
    // More detailed error message
    let errorMessage = 'Failed to analyze file.\n\n';
    
      if (error.message.includes('Failed to fetch')) {
        errorMessage += 'Cannot connect to backend server.\n' +
                     'Please ensure:\n' +
                     '1. Backend is running: python main.py\n' +
                     `2. Backend URL is: ${API_BASE_URL}\n` +
                     '3. No firewall blocking the connection';
    } else if (error.message.includes('Invalid response')) {
      errorMessage += 'Backend returned invalid data.\n' +
                     'Check backend console for errors.';
    } else {
      errorMessage += error.message;
    }
    
    alert(errorMessage);
  } finally {
    setAnalyzing(false);
    setProgress(0);
    setCurrentStep('');
  }
};

  const resetApp = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setProgress(0);
    setCurrentStep('');
  };

  const getClassificationTheme = (type) => {
    switch (type) {
      case 'AI_GENERATED':
        return {
          card: 'bg-gradient-to-br from-red-900/40 via-orange-900/40 to-red-900/40 border-red-500',
          iconWrap: 'bg-red-500/30 border-red-400',
          iconText: 'text-red-300',
          dot: 'rgba(239, 68, 68, 0.3)',
          scoreBar: 'bg-gradient-to-r from-red-500 via-orange-500 to-red-500',
          riskText: 'text-red-300'
        };
      case 'EDITED_ORIGINAL':
        return {
          card: 'bg-gradient-to-br from-amber-900/40 via-orange-900/40 to-yellow-900/40 border-amber-500',
          iconWrap: 'bg-amber-500/30 border-amber-400',
          iconText: 'text-amber-200',
          dot: 'rgba(245, 158, 11, 0.3)',
          scoreBar: 'bg-gradient-to-r from-amber-500 via-orange-400 to-yellow-500',
          riskText: 'text-amber-300'
        };
      default:
        return {
          card: 'bg-gradient-to-br from-green-900/40 via-emerald-900/40 to-green-900/40 border-green-500',
          iconWrap: 'bg-green-500/30 border-green-400',
          iconText: 'text-green-300',
          dot: 'rgba(34, 197, 94, 0.3)',
          scoreBar: 'bg-gradient-to-r from-green-500 via-emerald-400 to-green-500',
          riskText: 'text-green-300'
        };
    }
  };

  const getClassificationCopy = (type) => {
    switch (type) {
      case 'AI_GENERATED':
        return {
          title: 'AI-GENERATED CONTENT DETECTED',
          scoreLabel: 'MANIPULATION SCORE',
          meterLabel: 'AI GENERATION METRIC'
        };
      case 'EDITED_ORIGINAL':
        return {
          title: 'EDITED ORIGINAL DETECTED',
          scoreLabel: 'EDIT SCORE',
          meterLabel: 'EDITING METRIC'
        };
      default:
        return {
          title: 'LIKELY ORIGINAL / AUTHENTIC',
          scoreLabel: 'MANIPULATION SCORE',
          meterLabel: 'AUTHENTICITY METRIC'
        };
    }
  };

  const formatManipulationType = (type) => {
    switch (type) {
      case 'AI_GENERATED':
        return 'AI GENERATED';
      case 'EDITED_ORIGINAL':
        return 'EDITED ORIGINAL';
      default:
        return 'AUTHENTIC';
    }
  };

  const resultType = result?.manipulation_type || (result?.is_deepfake ? 'AI_GENERATED' : 'AUTHENTIC');
  const resultTheme = getClassificationTheme(resultType);
  const resultCopy = getClassificationCopy(resultType);

  return (
    <div className="min-h-screen bg-[#0a0e27] relative overflow-hidden">
      {/* Animated Background Grid */}
      <div className="absolute inset-0 opacity-20">
        <div className="absolute inset-0" style={{
          backgroundImage: `
            linear-gradient(rgba(0, 255, 255, 0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 255, 0.1) 1px, transparent 1px)
          `,
          backgroundSize: '50px 50px',
          animation: 'gridMove 20s linear infinite'
        }}></div>
      </div>

      {/* Floating Particles */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {[...Array(20)].map((_, i) => (
          <div
            key={i}
            className="absolute w-1 h-1 bg-cyan-400 rounded-full"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animation: `float ${5 + Math.random() * 10}s linear infinite`,
              animationDelay: `${Math.random() * 5}s`,
              opacity: 0.3 + Math.random() * 0.5
            }}
          ></div>
        ))}
      </div>

      {/* Scanning Lines */}
      {analyzing && (
        <>
          <div className="absolute inset-0 pointer-events-none">
            <div className="absolute w-full h-0.5 bg-gradient-to-r from-transparent via-cyan-400 to-transparent"
              style={{ animation: 'scanVertical 3s linear infinite' }}></div>
          </div>
          <div className="absolute inset-0 pointer-events-none">
            <div className="absolute h-full w-0.5 bg-gradient-to-b from-transparent via-blue-400 to-transparent"
              style={{ animation: 'scanHorizontal 4s linear infinite' }}></div>
          </div>
        </>
      )}
      
      {/* Main Content */}
      <div className="relative z-10 p-4 md:p-6">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-24 h-24 bg-gradient-to-br from-cyan-500 via-blue-500 to-purple-600 rounded-full mb-4 shadow-lg shadow-cyan-500/50 relative">
              <Shield className="w-12 h-12 text-white animate-pulse" />
              <div className="absolute inset-0 rounded-full border-2 border-cyan-400 animate-ping opacity-75"></div>
            </div>
            <h1 className="text-4xl md:text-6xl font-bold bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent mb-2 tracking-tight">
              MEDIA AUTHENTICITY ANALYZER
            </h1>
            <p className="text-cyan-300 text-lg tracking-wider">AI-GENERATED VS EDITED VS AUTHENTIC</p>
            
            {/* Backend Status */}
            <div className="mt-4 inline-flex items-center gap-3 px-6 py-3 rounded-full bg-black/40 backdrop-blur-sm border border-cyan-500/30">
              <Lock className="w-5 h-5 text-cyan-400" />
              <div className={`w-3 h-3 rounded-full ${
                backendStatus === 'connected' ? 'bg-green-400' : 
                backendStatus === 'error' ? 'bg-red-400' : 'bg-yellow-400'
              } animate-pulse shadow-lg ${
                backendStatus === 'connected' ? 'shadow-green-400/50' : 
                backendStatus === 'error' ? 'shadow-red-400/50' : 'shadow-yellow-400/50'
              }`}></div>
              <span className="text-sm text-cyan-200 font-mono tracking-wider">
                {backendStatus === 'connected' ? 'SECURE CONNECTION ESTABLISHED' : 
                 backendStatus === 'error' ? 'CONNECTION ERROR' : 'ESTABLISHING CONNECTION...'}
              </span>
            </div>
          </div>

          {/* Feature Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            {[
              { icon: Cpu, label: 'Model Signals', desc: 'Face + Image Models', color: 'cyan' },
              { icon: Activity, label: 'Forensic Checks', desc: 'Frequency + Lighting', color: 'green' },
              { icon: Database, label: 'Media Types', desc: 'Images, GIFs, Videos', color: 'purple' },
              { icon: Zap, label: 'Output Labels', desc: 'AI, Edited, Authentic', color: 'yellow' }
            ].map((feature, idx) => (
              <div key={idx} className="bg-black/40 backdrop-blur-lg rounded-lg p-4 border border-cyan-500/30 hover:border-cyan-400/60 transition-all hover:shadow-lg hover:shadow-cyan-500/20 group">
                <feature.icon className={`w-8 h-8 text-${feature.color}-400 mb-2 group-hover:scale-110 transition-transform`} />
                <h3 className="text-white font-semibold text-sm mb-1">{feature.label}</h3>
                <p className="text-cyan-300/70 text-xs">{feature.desc}</p>
              </div>
            ))}
          </div>

          {/* Main Upload Area with Cyber Effect */}
          <div className="bg-black/60 backdrop-blur-xl rounded-xl p-6 shadow-2xl border border-cyan-500/30 mb-6 relative overflow-hidden">
            {/* Corner Decorations */}
            <div className="absolute top-0 left-0 w-16 h-16 border-t-2 border-l-2 border-cyan-400/50"></div>
            <div className="absolute top-0 right-0 w-16 h-16 border-t-2 border-r-2 border-cyan-400/50"></div>
            <div className="absolute bottom-0 left-0 w-16 h-16 border-b-2 border-l-2 border-cyan-400/50"></div>
            <div className="absolute bottom-0 right-0 w-16 h-16 border-b-2 border-r-2 border-cyan-400/50"></div>

            {/* Upload Box */}
            <label className="flex flex-col items-center justify-center w-full h-56 border-2 border-dashed border-cyan-400/50 rounded-lg cursor-pointer hover:border-cyan-300 hover:bg-cyan-500/5 transition-all relative group">
              <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/0 via-cyan-500/5 to-cyan-500/0 group-hover:animate-shimmer"></div>
              <div className="flex flex-col items-center justify-center pt-5 pb-6 relative z-10">
                <Upload className="w-16 h-16 text-cyan-300 mb-4 group-hover:scale-110 transition-transform" />
                <p className="mb-2 text-lg text-cyan-200 font-mono">
                  <span className="font-semibold">UPLOAD FILE FOR ANALYSIS</span>
                </p>
                <p className="text-sm text-cyan-400/70 mb-2">Images (PNG, JPG, JPEG, GIF) or Videos (MP4, MOV, AVI)</p>
                <p className="text-xs text-cyan-500/60 font-mono">MAX SIZE: {MAX_UPLOAD_MB}MB | SECURE TRANSFER</p>
              </div>
              <input
                type="file"
                className="hidden"
                accept="image/*,video/*,image/gif"
                onChange={handleFileUpload}
              />
            </label>

            {/* Preview with Cyber Frame */}
            {preview && !analyzing && (
              <div className="mt-6">
                <div className="bg-black/60 rounded-lg p-4 border border-cyan-500/30 relative">
                  <div className="absolute top-2 left-2 px-3 py-1 bg-cyan-500/20 border border-cyan-400/50 rounded text-xs text-cyan-300 font-mono">
                    FILE LOADED
                  </div>
                  <div className="flex items-center gap-3 mb-3 mt-8">
                    {file.type === 'image/gif' ? (
                      <div className="text-green-400">🎞️</div>
                    ) : file.type.startsWith('image/') ? (
                      <div className="text-cyan-400">📷</div>
                    ) : (
                       <div className="text-purple-400">🎬</div>
                   )}
                    <div className="flex-1">
                      <p className="text-white font-mono text-sm">{file.name}</p>
                      <p className="text-cyan-300 text-xs font-mono">
                        {(file.size / 1024 / 1024).toFixed(2)} MB • {file.type}
                      </p>
                    </div>
                  </div>
                  
                  {file.type.startsWith('image/') ? (
                    <img src={preview} alt="Preview" className="max-h-96 mx-auto rounded shadow-lg shadow-cyan-500/20 border border-cyan-500/20" />
                  ) : (
                    <video
                      ref={videoRef}
                      src={preview}
                      controls
                      className="max-h-96 mx-auto rounded shadow-lg shadow-cyan-500/20 border border-cyan-500/20"
                    />
                  )}
                </div>
              </div>
            )}

            {/* Advanced Cyber Security Scanning Animation */}
            {analyzing && (
              <div className="mt-6 relative">
                {/* Main Scanning Container */}
                <div className="bg-black/80 backdrop-blur-xl rounded-lg p-8 border-2 border-cyan-400/50 relative overflow-hidden">
                  {/* Animated Circuit Pattern Background */}
                  <div className="absolute inset-0 opacity-10">
                    <svg className="w-full h-full" style={{ animation: 'pulse 2s ease-in-out infinite' }}>
                      <defs>
                        <pattern id="circuit" x="0" y="0" width="100" height="100" patternUnits="userSpaceOnUse">
                          <circle cx="50" cy="50" r="2" fill="#00ffff" />
                          <line x1="50" y1="0" x2="50" y2="100" stroke="#00ffff" strokeWidth="0.5" />
                          <line x1="0" y1="50" x2="100" y2="50" stroke="#00ffff" strokeWidth="0.5" />
                        </pattern>
                      </defs>
                      <rect width="100%" height="100%" fill="url(#circuit)" />
                    </svg>
                  </div>

                  {/* Scanning Radar Effect */}
                  <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-64">
                    <div className="absolute inset-0 border-2 border-cyan-400/30 rounded-full animate-ping"></div>
                    <div className="absolute inset-4 border-2 border-blue-400/40 rounded-full animate-ping" style={{ animationDelay: '0.5s' }}></div>
                    <div className="absolute inset-8 border-2 border-purple-400/50 rounded-full animate-ping" style={{ animationDelay: '1s' }}></div>
                  </div>

                  {/* Central Scanner Icon */}
                  <div className="flex items-center justify-center gap-4 mb-6 relative z-10">
                    <div className="relative">
                      <Scan className="w-16 h-16 text-cyan-400 animate-spin-slow" />
                      <div className="absolute inset-0 animate-pulse">
                        <Shield className="w-16 h-16 text-cyan-300 opacity-50" />
                      </div>
                    </div>
                    <div>
                      <h3 className="text-cyan-300 font-bold text-2xl tracking-wider font-mono">SCANNING IN PROGRESS</h3>
                      <p className="text-cyan-400 text-sm font-mono mt-1">{currentStep}</p>
                    </div>
                  </div>

                  {/* Hex Data Stream Effect */}
                  <div className="mb-6 h-24 overflow-hidden bg-black/60 rounded border border-cyan-500/30 relative">
                    <div className="absolute inset-0 flex flex-col gap-1 p-2 font-mono text-xs text-cyan-400/60 overflow-hidden" style={{ animation: 'scrollUp 3s linear infinite' }}>
                      {[...Array(20)].map((_, i) => (
                        <div key={i} className="whitespace-nowrap">
                          {`0x${Math.random().toString(16).substr(2, 8).toUpperCase()} ${Math.random().toString(16).substr(2, 8).toUpperCase()} ${Math.random().toString(16).substr(2, 8).toUpperCase()} ${Math.random().toString(16).substr(2, 8).toUpperCase()}`}
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Progress Bar with Glitch Effect */}
                  <div className="mb-4 relative">
                    <div className="flex justify-between text-sm text-cyan-300 mb-2 font-mono">
                      <span>ANALYSIS PROGRESS</span>
                      <span className="text-cyan-400 font-bold">{progress.toFixed(0)}%</span>
                    </div>
                    <div className="relative h-4 bg-black/60 rounded-full overflow-hidden border border-cyan-500/50">
                      <div
                        className="h-full bg-gradient-to-r from-cyan-500 via-blue-500 to-purple-500 transition-all duration-300 relative"
                        style={{ width: `${progress}%` }}
                      >
                        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shimmer"></div>
                      </div>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="h-1 w-full bg-gradient-to-r from-transparent via-cyan-300/50 to-transparent" style={{ animation: 'scanProgress 2s linear infinite' }}></div>
                      </div>
                    </div>
                  </div>

                  {/* Analysis Stages Grid */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2 relative z-10">
                    {['NEURAL', 'FREQUENCY', 'FACIAL', 'TEMPORAL'].map((stage, idx) => (
                      <div key={idx} className="bg-black/40 border border-cyan-500/30 rounded px-3 py-2 text-center relative overflow-hidden">
                        <div className={`absolute inset-0 bg-cyan-400/10 transition-all ${progress > (idx * 25) ? 'opacity-100' : 'opacity-0'}`}></div>
                        <div className="relative z-10">
                          <div className={`text-xs font-mono mb-1 ${progress > (idx * 25) ? 'text-cyan-400' : 'text-cyan-600'}`}>
                            {stage}
                          </div>
                          <div className={`text-xs font-bold font-mono ${progress > (idx * 25) ? 'text-green-400' : 'text-cyan-700'}`}>
                            {progress > (idx * 25) ? '✓ DONE' : 'WAITING'}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Warning Indicators */}
                  <div className="mt-4 flex items-center justify-center gap-2 text-yellow-400 text-xs font-mono animate-pulse">
                    <AlertTriangle className="w-4 h-4" />
                    <span>SECURE PROCESSING • DO NOT CLOSE WINDOW</span>
                  </div>
                </div>
              </div>
            )}

            {/* Analyze Button */}
            {file && !result && !analyzing && (
              <button
                onClick={handleAnalyze}
                disabled={backendStatus !== 'connected'}
                className={`w-full mt-6 ${
                  backendStatus === 'connected' 
                    ? 'bg-gradient-to-r from-cyan-600 via-blue-600 to-purple-600 hover:from-cyan-500 hover:via-blue-500 hover:to-purple-500 shadow-lg shadow-cyan-500/50' 
                    : 'bg-gray-700 cursor-not-allowed'
                } text-white font-bold py-5 px-6 rounded-lg transition-all flex items-center justify-center gap-3 border border-cyan-400/50 relative overflow-hidden group`}
              >
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent group-hover:animate-shimmer"></div>
                <Shield className="w-7 h-7 group-hover:scale-110 transition-transform" />
                <span className="text-xl tracking-wider font-mono">
                  {backendStatus === 'connected' ? 'INITIATE SECURITY SCAN' : 'BACKEND OFFLINE'}
                </span>
              </button>
            )}

            {/* Results */}
            {result && (
              <div className="mt-6 space-y-6">
                {/* Main Result Card */}
                <div className={`p-6 rounded-xl border-2 ${resultTheme.card} backdrop-blur-lg relative overflow-hidden`}>
                  {/* Animated Background */}
                  <div className="absolute inset-0 opacity-10">
                    <div className="absolute inset-0" style={{
                      backgroundImage: `radial-gradient(circle, ${resultTheme.dot} 1px, transparent 1px)`,
                      backgroundSize: '20px 20px',
                      animation: 'gridPulse 2s ease-in-out infinite'
                    }}></div>
                  </div>

                  <div className="flex items-start gap-4 mb-4 relative z-10">
                    {resultType === 'AI_GENERATED' ? (
                      <div className={`p-4 rounded-full border-2 animate-pulse ${resultTheme.iconWrap}`}>
                        <XCircle className={`w-12 h-12 ${resultTheme.iconText}`} />
                      </div>
                    ) : resultType === 'EDITED_ORIGINAL' ? (
                      <div className={`p-4 rounded-full border-2 animate-pulse ${resultTheme.iconWrap}`}>
                        <AlertTriangle className={`w-12 h-12 ${resultTheme.iconText}`} />
                      </div>
                    ) : (
                      <div className={`p-4 rounded-full border-2 animate-pulse ${resultTheme.iconWrap}`}>
                        <CheckCircle className={`w-12 h-12 ${resultTheme.iconText}`} />
                      </div>
                    )}
                    <div className="flex-1">
                      <h3 className="text-2xl md:text-3xl font-bold text-white mb-2 font-mono tracking-wide">
                        {resultCopy.title}
                      </h3>
                      <p className="text-white/90 mb-3 font-mono text-sm">
                        CLASSIFICATION: <span className={`font-bold text-lg ${resultTheme.riskText}`}>{formatManipulationType(resultType)}</span>
                      </p>
                      <p className="text-sm text-white/80 mb-4 leading-relaxed">
                        {result.summary || 'The system compared model signals and forensic artifacts to categorize this media.'}
                      </p>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="bg-black/40 rounded-lg p-3 border border-white/20">
                          <p className="text-xs text-white/70 mb-1 font-mono">{resultCopy.scoreLabel}</p>
                          <p className="text-4xl font-bold text-white font-mono">
                            {(result.manipulation_score ?? result.deepfake_score).toFixed(1)}%
                          </p>
                        </div>
                        <div className="bg-black/40 rounded-lg p-3 border border-white/20">
                          <p className="text-xs text-white/70 mb-1 font-mono">CONFIDENCE</p>
                          <p className="text-4xl font-bold text-white font-mono">{result.confidence.toFixed(1)}%</p>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  {/* Animated Progress Bar */}
                  <div className="mt-4 relative z-10">
                    <div className="flex justify-between text-sm text-white/90 mb-2 font-mono">
                      <span>{resultCopy.meterLabel}</span>
                      <span className="font-bold">{(result.manipulation_score ?? result.deepfake_score).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-black/40 rounded-full h-4 overflow-hidden border border-white/30">
                      <div
                        className={`h-4 rounded-full transition-all duration-1000 ${resultTheme.scoreBar} relative`}
                        style={{ width: `${result.manipulation_score ?? result.deepfake_score}%` }}
                      >
                        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shimmer"></div>
                      </div>
                    </div>
                  </div>
                </div>

                {result.reasons && result.reasons.length > 0 && (
                  <div className="bg-black/60 backdrop-blur-lg rounded-xl p-6 border border-cyan-500/30">
                    <h4 className="text-xl font-bold text-cyan-300 mb-4 flex items-center gap-2 font-mono tracking-wide">
                      <Info className="w-6 h-6" />
                      WHY THIS RESULT
                    </h4>
                    <div className="space-y-2">
                      {result.reasons.map((reason, idx) => (
                        <div key={idx} className="bg-black/40 rounded-lg p-3 border border-cyan-500/20 text-cyan-100/90 text-sm leading-relaxed">
                          {reason}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Frame Analysis for GIFs */}
                {result.frame_analysis && (
                  <div className="bg-black/60 backdrop-blur-lg rounded-xl p-6 border border-green-500/30">
                    <h4 className="text-xl font-bold text-green-300 mb-4 flex items-center gap-2 font-mono tracking-wide">
                      🎞️ FRAME-BY-FRAME ANALYSIS
                    </h4>
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                      {result.frame_analysis.map((frame, idx) => (
                        <div key={idx} className={`p-3 rounded border ${
                          frame.manipulation_type === 'AI_GENERATED'
                            ? 'bg-red-900/20 border-red-500'
                            : frame.manipulation_type === 'EDITED_ORIGINAL'
                              ? 'bg-amber-900/20 border-amber-500'
                              : 'bg-green-900/20 border-green-500'
                         }`}>
                          <p className="text-xs text-white/70 mb-1">Frame {frame.frame_number}</p>
                          <p className="text-lg font-bold text-white">{frame.score.toFixed(1)}%</p>
                          <p className="text-xs text-white/80">{formatManipulationType(frame.manipulation_type)}</p>
                        </div>
                      ))}
                     </div>
                 </div>
                )}

                {/* Analysis Details Grid */}
                {result.analysis_details && Object.keys(result.analysis_details).length > 0 && (
                  <div className="bg-black/60 backdrop-blur-lg rounded-xl p-6 border border-cyan-500/30">
                    <h4 className="text-xl font-bold text-cyan-300 mb-4 flex items-center gap-2 font-mono tracking-wide">
                      <Info className="w-6 h-6" />
                      DETAILED ANALYSIS REPORT
                    </h4>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                      {Object.entries(result.analysis_details).slice(0, 12).map(([key, value]) => (
                        <div key={key} className="bg-black/40 rounded-lg p-3 border border-cyan-500/20 hover:border-cyan-400/50 transition-all">
                          <p className="text-xs text-cyan-400/70 mb-1 font-mono uppercase">
                            {key.replace(/_/g, ' ')}
                          </p>
                          <p className="text-lg font-bold text-cyan-200 font-mono">
                            {typeof value === 'number' ? value.toFixed(1) : value}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Neural Network Results */}
                {result.neuralNetworks && Object.keys(result.neuralNetworks).length > 0 && (
                  <div className="bg-black/60 backdrop-blur-lg rounded-xl p-6 border border-purple-500/30">
                    <h4 className="text-xl font-bold text-purple-300 mb-4 flex items-center gap-2 font-mono tracking-wide">
                      <Cpu className="w-6 h-6" />
                      NEURAL NETWORK ENSEMBLE
                    </h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      {Object.entries(result.neuralNetworks).map(([model, score]) => (
                        <div key={model} className="bg-black/40 rounded-lg p-4 border border-purple-500/20 hover:border-purple-400/50 transition-all group">
                          <p className="text-xs text-purple-400/70 mb-2 font-mono uppercase tracking-wider">
                            {model}
                          </p>
                          <p className="text-3xl font-bold text-purple-200 font-mono group-hover:scale-110 transition-transform">{parseFloat(score).toFixed(1)}%</p>
                          <div className="mt-2 h-1 bg-purple-900/50 rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-gradient-to-r from-purple-500 to-pink-500 transition-all duration-1000"
                              style={{ width: `${score}%` }}
                            ></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Reset Button */}
                <button
                  onClick={resetApp}
                  className="w-full bg-black/60 hover:bg-black/80 border-2 border-cyan-500/50 hover:border-cyan-400 text-cyan-300 font-bold py-4 px-6 rounded-lg transition-all font-mono tracking-wider relative overflow-hidden group"
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-cyan-500/10 to-transparent group-hover:animate-shimmer"></div>
                  <span className="relative z-10">ANALYZE NEW FILE</span>
                </button>
              </div>
            )}
          </div>

          {/* Information Panel */}
          <div className="bg-yellow-900/20 border-2 border-yellow-500/50 rounded-lg p-5 backdrop-blur-sm">
            <div className="flex gap-3">
              <AlertTriangle className="w-6 h-6 text-yellow-400 flex-shrink-0 mt-0.5 animate-pulse" />
              <div>
                <p className="font-bold text-yellow-300 mb-2 font-mono tracking-wide">SYSTEM INFORMATION</p>
                <p className="text-sm text-yellow-200/80 font-mono leading-relaxed">
                  Media authenticity system combining model signals with frequency, lighting, facial,
                  and temporal analysis to separate AI-generated media, edited originals, and likely authentic files.
                  Secure encrypted connection to backend server at {API_BASE_URL}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
