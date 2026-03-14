const normalizeApiBaseUrl = (value) => value.replace(/\/+$/, '');

const defaultApiBaseUrl = 'http://localhost:8000';
const rawApiBaseUrl = process.env.REACT_APP_API_BASE_URL || defaultApiBaseUrl;
const rawMaxUploadMb = Number(process.env.REACT_APP_MAX_UPLOAD_MB || '100');

export const API_BASE_URL = normalizeApiBaseUrl(rawApiBaseUrl);
export const HEALTH_ENDPOINT = `${API_BASE_URL}/health`;
export const ANALYZE_ENDPOINT = `${API_BASE_URL}/api/analyze`;
export const MAX_UPLOAD_MB = Number.isFinite(rawMaxUploadMb) && rawMaxUploadMb > 0
  ? rawMaxUploadMb
  : 100;
