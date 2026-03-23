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
  XCircle,
  Zap
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
    title: 'Ensemble verdicts',
    description: 'Multiple learned detectors vote together before a label is shown.'
  },
  {
    icon: Activity,
    title: 'Forensic checks',
    description: 'Frequency, lighting, and artifact analysis help catch hidden edits.'
  },
  {
    icon: Database,
    title: 'Mixed media support',
    description: 'Run the same workflow against images, animated GIFs, and short videos.'
  },
  {
    icon: Zap,
    title: 'Readable output',
    description: 'You get a compact verdict, supporting reasons, and model-level evidence.'
  }
];

const WORKFLOW_STEPS = [
  {
    step: '01',
    title: 'Upload once',
    description: 'Choose an image, GIF, or video clip from your device and preview it before scanning.'
  },
  {
    step: '02',
    title: 'Wait for the checks',
    description: 'The app runs model signals and media forensics side by side while the progress panel updates.'
  },
  {
    step: '03',
    title: 'Read the verdict',
    description: 'Review the classification, risk level, reasons, and optional frame-by-frame breakdown in one place.'
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
        detail: 'Ready for secure analysis',
        tone: 'connected'
      };
    case 'error':
      return {
        label: 'Backend offline',
        detail: 'Retry or start the API server',
        tone: 'error'
      };
    default:
      return {
        label: 'Checking backend',
        detail: 'Verifying connection before uploads',
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
        narrative: 'Strong synthetic patterns were detected across the uploaded media.',
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
        narrative: 'The media appears rooted in a real source but carries clear editing signals.',
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
        narrative: 'The strongest signals currently lean toward authentic capture with lower manipulation risk.',
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

  return (
    <div className="app-shell">
      <div className="ambient ambient--one" />
      <div className="ambient ambient--two" />
      <div className="ambient ambient--three" />

      <main className="app-frame">
        <header className="topbar reveal reveal--1">
          <div className="brand-lockup">
            <div className="brand-mark">
              <Shield />
            </div>
            <div>
              <p className="section-tag">Authenticity intelligence</p>
              <h1 className="brand-title">SignalScope</h1>
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

        <section className="hero-grid">
          <div className="panel panel--dark hero-copy reveal reveal--2">
            <p className="section-tag">Upload once. Review fast.</p>
            <h2>Spot AI-generated or edited media before it spreads.</h2>
            <p className="hero-lead">
              The detector logic stays intact, but the experience now reads like a
              clean review desk instead of a noisy effects demo. Upload an image,
              GIF, or video, then scan the verdict, confidence, and supporting
              evidence in one pass.
            </p>

            <div className="hero-chip-row">
              <span className="hero-chip">
                <Upload size={14} />
                Images, GIFs, and video
              </span>
              <span className="hero-chip">
                <Activity size={14} />
                Live backend status
              </span>
              <span className="hero-chip">
                <Lock size={14} />
                {MAX_UPLOAD_MB}MB secure upload cap
              </span>
            </div>

            <div className="hero-stat-grid">
              <article className="stat-card">
                <span className="stat-value">3</span>
                <span className="stat-label">Verdict types</span>
                <p>AI generated, edited original, or likely authentic.</p>
              </article>
              <article className="stat-card">
                <span className="stat-value">GIF</span>
                <span className="stat-label">Frame review</span>
                <p>Animated uploads can return frame-level manipulation scores.</p>
              </article>
              <article className="stat-card">
                <span className="stat-value">Live</span>
                <span className="stat-label">Status checks</span>
                <p>The backend heartbeat stays visible before you upload anything.</p>
              </article>
            </div>
          </div>

          <div className="panel panel--light upload-panel reveal reveal--3">
            <div className="panel-head">
              <div>
                <p className="section-tag section-tag--ink">New scan</p>
                <h2>Drop media into the review desk</h2>
              </div>

              {file ? (
                <button type="button" className="mini-button" onClick={resetApp}>
                  Clear
                </button>
              ) : null}
            </div>

            <label className={`dropzone ${preview ? 'dropzone--compact' : ''}`}>
              <div className="dropzone-icon">
                <Upload />
              </div>
              <h3>Choose image, GIF, or video</h3>
              <p>Supported formats: PNG, JPG, JPEG, GIF, MP4, MOV, AVI</p>
              <span>Maximum upload size: {MAX_UPLOAD_MB}MB</span>
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
                    {fileMeta ? <fileMeta.icon size={18} /> : <ImageIcon size={18} />}
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
                    <h3>Running authenticity checks</h3>
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
                      <div
                        key={stage}
                        className={`analysis-stage ${isActive ? 'is-active' : ''}`}
                      >
                        <span>{stage}</span>
                        <strong>{isActive ? 'Ready' : 'Pending'}</strong>
                      </div>
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
                  Start authenticity scan
                  <ArrowRight size={18} />
                </button>

                <p className="upload-caption">
                  {backendStatus === 'connected'
                    ? 'Your file stays local until you press the scan button.'
                    : 'Reconnect the backend before starting a scan.'}
                </p>
              </>
            ) : null}

            {!file ? (
              <p className="upload-caption">
                Current API target: <span>{API_BASE_URL}</span>
              </p>
            ) : null}
          </div>
        </section>

        <section className="capability-grid reveal reveal--4">
          {CAPABILITY_CARDS.map((item) => {
            const Icon = item.icon;

            return (
              <article key={item.title} className="capability-card">
                <div className="capability-card__icon">
                  <Icon size={20} />
                </div>
                <h3>{item.title}</h3>
                <p>{item.description}</p>
              </article>
            );
          })}
        </section>

        {result ? (
          <>
            <section className={`result-hero tone-${tone.key} reveal reveal--5`}>
              <div className="result-badge">
                <ToneIcon size={34} />
              </div>

              <div className="result-copy">
                <p className="section-tag section-tag--soft">Latest verdict</p>
                <h2>{tone.title}</h2>
                <p>{result.summary || tone.narrative}</p>
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

            <section className="results-layout reveal reveal--6">
              <div className="results-main">
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

              <div className="results-side">
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

            <button type="button" className="secondary-button reveal reveal--6" onClick={resetApp}>
              Analyze another file
            </button>
          </>
        ) : (
          <section className="workflow-grid reveal reveal--5">
            <section className="panel panel--soft">
              <div className="panel-heading-stack">
                <p className="section-tag section-tag--ink">Review flow</p>
                <h2>Built for quick moderation and trust checks.</h2>
              </div>

              <div className="workflow-list">
                {WORKFLOW_STEPS.map((item) => (
                  <article key={item.step} className="workflow-item">
                    <span className="workflow-step">{item.step}</span>
                    <div>
                      <h3>{item.title}</h3>
                      <p>{item.description}</p>
                    </div>
                  </article>
                ))}
              </div>
            </section>

            <section className="panel panel--dark">
              <div className="panel-heading-stack">
                <p className="section-tag">Connection</p>
                <h2>{statusMeta.label}</h2>
              </div>

              <p className="supporting-copy">
                Uploads are sent only when you start a scan. Current backend endpoint:
              </p>
              <p className="endpoint-pill">{API_BASE_URL}</p>

              <div className="utility-note">
                <BarChart3 size={18} />
                <p>
                  Results combine detector confidence, manipulation score, and optional
                  breakdowns from frame and model-level analysis.
                </p>
              </div>
            </section>
          </section>
        )}

        <footer className="footer-note reveal reveal--6">
          <Lock size={16} />
          <p>
            Media authenticity review using model signals, artifact checks, and
            temporal analysis. Backend endpoint: <span>{API_BASE_URL}</span>
          </p>
        </footer>
      </main>
    </div>
  );
}
