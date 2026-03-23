import React, { useEffect, useRef, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  CheckCircle,
  Cpu,
  Database,
  Film,
  Image as ImageIcon,
  Info,
  Lock,
  RefreshCw,
  Scan,
  Shield,
  Upload,
  XCircle
} from 'lucide-react';
import './App.css';
import { API_BASE_URL, ANALYZE_ENDPOINT, HEALTH_ENDPOINT, MAX_UPLOAD_MB } from './config';

const ANALYSIS_STEPS = [
  'Uploading media to the analysis service...',
  'Scanning visual fingerprints and compression traces...',
  'Comparing model outputs across forensic detectors...',
  'Reviewing lighting, texture, and frequency cues...',
  'Aggregating ensemble votes and anomaly scores...',
  'Preparing the final authenticity report...'
];

const CAPABILITY_CARDS = [
  {
    icon: Cpu,
    title: 'Ensemble models',
    description: 'Multiple detector outputs are combined before the final verdict.'
  },
  {
    icon: Activity,
    title: 'Artifact analysis',
    description: 'Texture, lighting, and compression cues stay visible in the report.'
  },
  {
    icon: Film,
    title: 'Frame scoring',
    description: 'Animated uploads can return frame-level manipulation signals.'
  },
  {
    icon: BarChart3,
    title: 'Readable output',
    description: 'Confidence, risk, and reasons stay grouped in one result view.'
  }
];

const WORKFLOW_STEPS = [
  {
    step: '01',
    title: 'Select media',
    description: 'Choose an image, GIF, or short video and confirm the preview.'
  },
  {
    step: '02',
    title: 'Run the scan',
    description: 'Model and forensic checks are processed together.'
  },
  {
    step: '03',
    title: 'Read the result',
    description: 'Review the verdict, confidence, and supporting notes.'
  }
];

const ANALYSIS_STAGES = ['Upload', 'Signals', 'Forensics', 'Verdict'];

const clampPercent = (value) => {
  const numericValue = Number(value);

  if (!Number.isFinite(numericValue)) {
    return 0;
  }

  return Math.max(0, Math.min(100, numericValue));
};

const formatPercent = (value) => clampPercent(value).toFixed(1);

const formatKeyLabel = (key) =>
  key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());

const formatManipulationType = (type) => {
  switch (type) {
    case 'AI_GENERATED':
      return 'AI Generated';
    case 'EDITED_ORIGINAL':
      return 'Edited Original';
    default:
      return 'Authentic';
  }
};

const getStatusMeta = (status) => {
  switch (status) {
    case 'connected':
      return {
        label: 'API online',
        detail: 'Ready for analysis',
        tone: 'connected'
      };
    case 'error':
      return {
        label: 'Backend offline',
        detail: 'Start the API server or retry the connection',
        tone: 'error'
      };
    default:
      return {
        label: 'Checking backend',
        detail: 'Verifying the API before uploads',
        tone: 'checking'
      };
  }
};

const getFileMeta = (file) => {
  if (!file) {
    return null;
  }

  if (file.type === 'image/gif') {
    return {
      icon: Film,
      label: 'Animated GIF'
    };
  }

  if (file.type.startsWith('image/')) {
    return {
      icon: ImageIcon,
      label: 'Still image'
    };
  }

  return {
    icon: Film,
    label: 'Video clip'
  };
};

const getResultTone = (type) => {
  switch (type) {
    case 'AI_GENERATED':
      return {
        key: 'ai',
        title: 'High manipulation risk',
        narrative: 'Strong synthetic patterns were detected in the uploaded media.',
        primaryLabel: 'Manipulation score',
        secondaryLabel: 'Authenticity score',
        meterLabel: 'Synthetic likelihood',
        useAuthenticity: false,
        Icon: XCircle
      };
    case 'EDITED_ORIGINAL':
      return {
        key: 'edited',
        title: 'Edited original detected',
        narrative: 'The media appears rooted in a real source but carries editing signals.',
        primaryLabel: 'Edit score',
        secondaryLabel: 'Authenticity score',
        meterLabel: 'Edit likelihood',
        useAuthenticity: false,
        Icon: AlertTriangle
      };
    default:
      return {
        key: 'authentic',
        title: 'Likely authentic media',
        narrative: 'The strongest signals currently lean toward authentic capture.',
        primaryLabel: 'Authenticity score',
        secondaryLabel: 'Manipulation score',
        meterLabel: 'Authenticity confidence',
        useAuthenticity: true,
        Icon: CheckCircle
      };
  }
};

const normalizeResult = (data) => ({
  is_deepfake: Boolean(data.is_deepfake),
  is_manipulated: Boolean(data.is_manipulated ?? data.is_deepfake),
  deepfake_score: Number(data.deepfake_score || data.deepfakeScore || 0),
  manipulation_score: Number(
    data.manipulation_score || data.deepfake_score || data.deepfakeScore || 0
  ),
  authenticity_score: Number(
    data.authenticity_score ??
      Math.max(
        0,
        100 - Number(data.manipulation_score || data.deepfake_score || data.deepfakeScore || 0)
      )
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

const createProgressSimulation = (steps, setProgress, setCurrentStep) => {
  let currentProgress = 0;
  let intervalId = null;

  setCurrentStep(steps[0]);

  const promise = new Promise((resolve) => {
    intervalId = setInterval(() => {
      currentProgress = Math.min(currentProgress + 8, 90);

      const stepIndex = Math.min(
        steps.length - 1,
        Math.floor((currentProgress / 90) * steps.length)
      );

      setCurrentStep(steps[stepIndex]);
      setProgress(currentProgress);

      if (currentProgress >= 90) {
        clearInterval(intervalId);
        resolve();
      }
    }, 320);
  });

  return {
    promise,
    cancel: () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    }
  };
};

export default function DeepfakeDetector() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const [result, setResult] = useState(null);
  const [backendStatus, setBackendStatus] = useState('checking');
  const fileInputRef = useRef(null);

  useEffect(() => {
    checkBackendHealth();
  }, []);

  const checkBackendHealth = async () => {
    setBackendStatus('checking');

    try {
      const response = await fetch(HEALTH_ENDPOINT);
      setBackendStatus(response.ok ? 'connected' : 'error');
    } catch (error) {
      setBackendStatus('error');
    }
  };

  const handleFileUpload = (event) => {
    const uploadedFile = event.target.files?.[0];

    if (!uploadedFile) {
      return;
    }

    const maxSize = MAX_UPLOAD_MB * 1024 * 1024;

    if (uploadedFile.size > maxSize) {
      alert(`File size exceeds ${MAX_UPLOAD_MB}MB. Please upload a smaller file.`);
      return;
    }

    setFile(uploadedFile);
    setResult(null);
    setCurrentStep('');
    setProgress(0);

    const reader = new FileReader();

    reader.onload = (loadEvent) => {
      setPreview(loadEvent.target?.result || null);
    };

    reader.readAsDataURL(uploadedFile);
  };

  const handleAnalyze = async () => {
    if (!file) {
      return;
    }

    setAnalyzing(true);
    setResult(null);
    setProgress(0);

    const progressSimulation = createProgressSimulation(
      ANALYSIS_STEPS,
      setProgress,
      setCurrentStep
    );

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(ANALYZE_ENDPOINT, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Backend error (${response.status}): ${errorText}`);
      }

      const data = await response.json();

      if (!data || typeof data.is_deepfake === 'undefined') {
        throw new Error(
          'Invalid response format from backend. Expected is_deepfake property.'
        );
      }

      await progressSimulation.promise;
      setProgress(100);
      setCurrentStep('Report ready');
      await new Promise((resolve) => setTimeout(resolve, 250));
      setResult(normalizeResult(data));
    } catch (error) {
      progressSimulation.cancel();

      let errorMessage = 'Failed to analyze file.\n\n';

      if (error.message.includes('Failed to fetch')) {
        errorMessage +=
          'Cannot connect to backend server.\n' +
          'Please ensure:\n' +
          '1. Backend is running: python main.py\n' +
          `2. Backend URL is: ${API_BASE_URL}\n` +
          '3. No firewall is blocking the connection';
      } else if (error.message.includes('Invalid response')) {
        errorMessage +=
          'Backend returned invalid data.\nCheck the backend console for errors.';
      } else {
        errorMessage += error.message;
      }

      alert(errorMessage);
    } finally {
      progressSimulation.cancel();
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

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const statusMeta = getStatusMeta(backendStatus);
  const fileMeta = getFileMeta(file);
  const FileMetaIcon = fileMeta?.icon ?? ImageIcon;
  const resultType =
    result?.manipulation_type || (result?.is_deepfake ? 'AI_GENERATED' : 'AUTHENTIC');
  const tone = getResultTone(resultType);
  const ToneIcon = tone.Icon;
  const primaryValue = tone.useAuthenticity
    ? result?.authenticity_score
    : result?.manipulation_score ?? result?.deepfake_score;
  const secondaryValue = tone.useAuthenticity
    ? result?.manipulation_score ?? result?.deepfake_score
    : result?.authenticity_score;
  const signalEntries = Object.entries(result?.signal_scores ?? {}).slice(0, 6);
  const detailEntries = Object.entries(result?.analysis_details ?? {}).slice(0, 8);
  const networkEntries = Object.entries(result?.neuralNetworks ?? {});
  const interfaceStats = [
    {
      icon: Shield,
      label: 'Status',
      value:
        backendStatus === 'connected'
          ? 'Synced'
          : backendStatus === 'checking'
            ? 'Checking'
            : 'Offline',
      detail: statusMeta.detail
    },
    {
      icon: Database,
      label: 'Formats',
      value: 'IMG / GIF / VID',
      detail: `Up to ${MAX_UPLOAD_MB}MB`
    },
    {
      icon: Cpu,
      label: 'Engine',
      value: 'Ensemble',
      detail: 'Model + forensic checks'
    }
  ];

  return (
    <div className="app-shell">
      <div className="background-grid" />
      <div className="background-glow background-glow--one" />
      <div className="background-glow background-glow--two" />

      <main className="app-frame">
        <header className="topbar reveal reveal--1">
          <div className="brand-lockup">
            <div className="brand-mark">
              <Shield />
            </div>
            <div>
              <p className="section-tag">Media authenticity</p>
              <h1 className="brand-title">Trace</h1>
            </div>
          </div>

          <div className="topbar-actions">
            <div className={`status-pill status-pill--${statusMeta.tone}`}>
              <Lock size={16} />
              <div>
                <strong>{statusMeta.label}</strong>
                <span>{statusMeta.detail}</span>
              </div>
            </div>

            <button
              type="button"
              className="ghost-button"
              onClick={checkBackendHealth}
            >
              <RefreshCw
                size={16}
                className={backendStatus === 'checking' ? 'icon-spin' : ''}
              />
              Refresh
            </button>
          </div>
        </header>

        <section className="hero-layout">
          <section className="panel panel--dark hero-panel reveal reveal--2">
            <p className="section-tag">Deepfake detector</p>
            <h2>Analyze media for manipulation signals.</h2>
            <p className="hero-lead">
              Upload an image, GIF, or short video. Run the scan and review the
              verdict, confidence, and supporting evidence in one place.
            </p>

            <div className="hero-chip-row">
              <span className="hero-chip">
                <Upload size={14} />
                Image / GIF / Video
              </span>
              <span className="hero-chip">
                <Activity size={14} />
                Live API status
              </span>
              <span className="hero-chip">
                <Lock size={14} />
                {MAX_UPLOAD_MB}MB upload limit
              </span>
            </div>

            <div className="hero-stat-grid">
              {interfaceStats.map((item) => {
                const Icon = item.icon;

                return (
                  <article key={item.label} className="hero-stat-card">
                    <div className="hero-stat-card__icon">
                      <Icon size={16} />
                    </div>
                    <span>{item.label}</span>
                    <strong>{item.value}</strong>
                    <p>{item.detail}</p>
                  </article>
                );
              })}
            </div>

            <div className="hero-flow">
              {WORKFLOW_STEPS.map((item) => (
                <article key={item.step} className="hero-flow__item">
                  <span>{item.step}</span>
                  <strong>{item.title}</strong>
                  <p>{item.description}</p>
                </article>
              ))}
            </div>
          </section>

          <section className="panel panel--light upload-panel reveal reveal--3">
            <div className="panel-head">
              <div>
                <p className="section-tag section-tag--ink">New scan</p>
                <h3>Start with a file</h3>
              </div>

              {file ? (
                <button type="button" className="mini-button" onClick={resetApp}>
                  Clear
                </button>
              ) : null}
            </div>

            <div className="upload-band">
              <span>Select</span>
              <span>Preview</span>
              <span>Run scan</span>
            </div>

            <label className={`dropzone ${preview ? 'dropzone--compact' : ''}`}>
              <div className="dropzone-icon">
                <Upload />
              </div>
              <h3>Select image, GIF, or video</h3>
              <p>PNG, JPG, JPEG, GIF, MP4, MOV, AVI</p>
              <span>Up to {MAX_UPLOAD_MB}MB</span>
              <input
                ref={fileInputRef}
                type="file"
                className="hidden-input"
                accept="image/*,video/*,image/gif"
                onChange={handleFileUpload}
                aria-label="Upload media file"
              />
            </label>

            {preview && file && !analyzing ? (
              <div className="asset-card">
                <div className="asset-card__head">
                  <div className="asset-icon">
                    <FileMetaIcon size={18} />
                  </div>
                  <div className="asset-meta">
                    <strong>{file.name}</strong>
                    <span>
                      {fileMeta?.label || 'Media file'} •{' '}
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </span>
                  </div>
                </div>

                <div className="asset-preview">
                  {file.type.startsWith('image/') ? (
                    <img src={preview} alt="Selected media preview" />
                  ) : (
                    <video src={preview} controls />
                  )}
                </div>
              </div>
            ) : null}

            {analyzing ? (
              <div className="analysis-panel">
                <div className="analysis-panel__head">
                  <div className="scanner-orb">
                    <Scan />
                  </div>
                  <div>
                    <p className="analysis-kicker">Live analysis</p>
                    <h3>Running checks</h3>
                    <p>{currentStep}</p>
                  </div>
                </div>

                <div className="progress-meta">
                  <span>Scan progress</span>
                  <strong>{Math.round(progress)}%</strong>
                </div>
                <div className="progress-track">
                  <div
                    className="progress-fill"
                    style={{ width: `${clampPercent(progress)}%` }}
                  />
                </div>

                <div className="analysis-stage-grid">
                  {ANALYSIS_STAGES.map((stage, index) => {
                    const threshold = index * (100 / ANALYSIS_STAGES.length);
                    const isActive = progress >= threshold;

                    return (
                      <article
                        key={stage}
                        className={`analysis-stage ${isActive ? 'is-active' : ''}`}
                      >
                        <span>{stage}</span>
                        <strong>{isActive ? 'Ready' : 'Pending'}</strong>
                      </article>
                    );
                  })}
                </div>
              </div>
            ) : null}

            {file && !result && !analyzing ? (
              <>
                <button
                  type="button"
                  onClick={handleAnalyze}
                  disabled={backendStatus !== 'connected'}
                  className="primary-button"
                >
                  Run scan
                  <ArrowRight size={18} />
                </button>

                <p className="upload-caption">
                  {backendStatus === 'connected'
                    ? 'The file is uploaded only after you start the scan.'
                    : 'Reconnect the backend before starting a scan.'}
                </p>
              </>
            ) : null}

            {!file ? (
              <p className="upload-caption">
                Current API target: <span>{API_BASE_URL}</span>
              </p>
            ) : null}
          </section>
        </section>

        {result ? (
          <>
            <section className={`result-summary tone-${tone.key} reveal reveal--4`}>
              <div className="result-summary__badge">
                <ToneIcon size={34} />
              </div>

              <div className="result-summary__copy">
                <p className="section-tag section-tag--soft">Latest verdict</p>
                <h2>{tone.title}</h2>
                <p>{result.summary || tone.narrative}</p>
                <div className="result-flags">
                  <span className="result-flag">{formatManipulationType(resultType)}</span>
                  <span className="result-flag">
                    Confidence {formatPercent(result.confidence)}%
                  </span>
                </div>
              </div>

              <div className="result-metrics">
                <article className="metric-card">
                  <span>{tone.primaryLabel}</span>
                  <strong>{formatPercent(primaryValue)}%</strong>
                </article>
                <article className="metric-card">
                  <span>Confidence</span>
                  <strong>{formatPercent(result.confidence)}%</strong>
                </article>
                <article className="metric-card">
                  <span>Classification</span>
                  <strong>{formatManipulationType(resultType)}</strong>
                </article>
                <article className="metric-card">
                  <span>Risk level</span>
                  <strong>{result.risk_level}</strong>
                </article>
              </div>

              <div className="result-meter">
                <div className="result-meter__top">
                  <span>{tone.meterLabel}</span>
                  <strong>{formatPercent(primaryValue)}%</strong>
                </div>
                <div className="result-meter__track">
                  <div
                    className="result-meter__fill"
                    style={{ width: `${clampPercent(primaryValue)}%` }}
                  />
                </div>
                <p className="result-meter__caption">
                  {tone.secondaryLabel}:{' '}
                  <strong>{formatPercent(secondaryValue)}%</strong>
                </p>
              </div>
            </section>

            <section className="result-grid reveal reveal--5">
              <div className="result-column">
                {result.reasons?.length ? (
                  <section className="panel panel--soft">
                    <div className="panel-heading-stack">
                      <p className="section-tag section-tag--ink">Why it landed here</p>
                      <h3>Supporting notes</h3>
                    </div>
                    <div className="reason-list">
                      {result.reasons.map((reason) => (
                        <article key={reason} className="reason-item">
                          <Info size={16} />
                          <p>{reason}</p>
                        </article>
                      ))}
                    </div>
                  </section>
                ) : null}

                {result.frame_analysis ? (
                  <section className="panel panel--soft">
                    <div className="panel-heading-stack">
                      <p className="section-tag section-tag--ink">Animated media</p>
                      <h3>Frame-level breakdown</h3>
                    </div>

                    <div className="frame-grid">
                      {result.frame_analysis.map((frame) => {
                        const frameTone = getResultTone(frame.manipulation_type || 'AUTHENTIC');

                        return (
                          <article
                            key={`${frame.frame_number}-${frame.score}`}
                            className={`frame-card tone-${frameTone.key}`}
                          >
                            <span>Frame {frame.frame_number}</span>
                            <strong>{formatPercent(frame.score)}%</strong>
                            <p>{formatManipulationType(frame.manipulation_type)}</p>
                          </article>
                        );
                      })}
                    </div>
                  </section>
                ) : null}
              </div>

              <div className="result-column">
                {signalEntries.length ? (
                  <section className="panel panel--dark">
                    <div className="panel-heading-stack">
                      <p className="section-tag">Signal breakdown</p>
                      <h3>What contributed to the verdict</h3>
                    </div>

                    <div className="score-list">
                      {signalEntries.map(([key, value]) => (
                        <article key={key} className="score-row">
                          <div className="score-row__top">
                            <span>{formatKeyLabel(key)}</span>
                            <strong>{formatPercent(value)}%</strong>
                          </div>
                          <div className="score-bar">
                            <div
                              className="score-bar__fill"
                              style={{ width: `${clampPercent(value)}%` }}
                            />
                          </div>
                        </article>
                      ))}
                    </div>
                  </section>
                ) : null}

                {detailEntries.length ? (
                  <section className="panel panel--soft">
                    <div className="panel-heading-stack">
                      <p className="section-tag section-tag--ink">Detailed analysis</p>
                      <h3>Key report values</h3>
                    </div>

                    <div className="detail-grid">
                      {detailEntries.map(([key, value]) => (
                        <article key={key} className="detail-card">
                          <span>{formatKeyLabel(key)}</span>
                          <strong>
                            {typeof value === 'number' ? value.toFixed(1) : String(value)}
                          </strong>
                        </article>
                      ))}
                    </div>
                  </section>
                ) : null}

                {networkEntries.length ? (
                  <section className="panel panel--dark">
                    <div className="panel-heading-stack">
                      <p className="section-tag">Model ensemble</p>
                      <h3>Neural network outputs</h3>
                    </div>

                    <div className="network-grid">
                      {networkEntries.map(([model, score]) => (
                        <article key={model} className="network-card">
                          <span>{formatKeyLabel(model)}</span>
                          <strong>{formatPercent(score)}%</strong>
                          <div className="score-bar">
                            <div
                              className="score-bar__fill"
                              style={{ width: `${clampPercent(score)}%` }}
                            />
                          </div>
                        </article>
                      ))}
                    </div>
                  </section>
                ) : null}
              </div>
            </section>

            <button
              type="button"
              className="secondary-button reveal reveal--6"
              onClick={resetApp}
            >
              Analyze another file
            </button>
          </>
        ) : (
          <section className="overview-grid reveal reveal--4">
            <section className="panel panel--soft">
              <div className="panel-heading-stack">
                <p className="section-tag section-tag--ink">What you'll get</p>
                <h3>Output built for quick review.</h3>
              </div>

              <div className="capability-grid">
                {CAPABILITY_CARDS.map((item) => {
                  const Icon = item.icon;

                  return (
                    <article key={item.title} className="capability-card">
                      <div className="capability-card__icon">
                        <Icon size={18} />
                      </div>
                      <h4>{item.title}</h4>
                      <p>{item.description}</p>
                    </article>
                  );
                })}
              </div>
            </section>

            <section className="panel panel--dark utility-panel">
              <div className="panel-heading-stack">
                <p className="section-tag">Connection</p>
                <h3>{statusMeta.label}</h3>
              </div>

              <p className="supporting-copy">Current backend endpoint</p>
              <p className="endpoint-pill">{API_BASE_URL}</p>

              <div className="utility-note">
                <BarChart3 size={18} />
                <p>
                  Confidence, risk level, model outputs, and optional frame analysis
                  are shown after each scan.
                </p>
              </div>
            </section>
          </section>
        )}

        <footer className="footer-note reveal reveal--6">
          <Lock size={16} />
          <p>
            Authenticity review for images, GIFs, and short videos. Backend endpoint:{' '}
            <span>{API_BASE_URL}</span>
          </p>
        </footer>
      </main>
    </div>
  );
}
