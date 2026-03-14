# Deepfake-main Handoff

## Project Status

This project is now running locally as a full-stack app:

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

The app has been moved from a binary "threat detected" style result to a more useful media-authenticity workflow.

Current backend output categories:

- `AUTHENTIC`
- `EDITED_ORIGINAL`
- `AI_GENERATED`

## Current Features

### Frontend

- Upload UI for images, GIFs, and videos
- Backend health check on load
- Result card with category-aware styling
- Result explanation section: "Why this result"
- Frame-by-frame display for GIF analysis
- Env-based API configuration through `frontend/src/config.js`

Main frontend files:

- `frontend/src/App.js`
- `frontend/src/config.js`
- `frontend/.env.example`

### Backend

- FastAPI upload endpoint at `POST /api/analyze`
- Health endpoint at `GET /health`
- Model info and stats endpoints
- Face-focused ensemble scoring for detected faces
- HuggingFace fallback image scoring
- Frequency-domain analysis
- Lighting consistency analysis
- Temporal analysis for videos and GIFs
- Category-based response fields:
  - `manipulation_type`
  - `manipulation_score`
  - `authenticity_score`
  - `is_manipulated`
  - `summary`
  - `reasons`
  - `signal_scores`
- Env-based runtime config for:
  - host
  - port
  - public base URL
  - CORS origins
  - upload size
  - log level

Main backend files:

- `backend/main.py`
- `backend/models/huggingface_detector.py`
- `backend/.env.example`

## What Changed In This Work

- Set up and verified local backend + frontend
- Installed backend Python dependencies and frontend npm dependencies
- Confirmed end-to-end upload flow works locally
- Reviewed and explained the backend model pipeline
- Reworked classification logic so the app does not mark nearly every file as a threat
- Updated the UI to show `AI_GENERATED`, `EDITED_ORIGINAL`, or `AUTHENTIC`
- Removed hardcoded `localhost:8000` usage from the frontend
- Added env-based deployment configuration for frontend and backend
- Updated root README deployment notes

## Current Detection Behavior

### What works better now

- Clean non-face images are less likely to be falsely flagged
- Edited images can now surface as `EDITED_ORIGINAL`
- AI-generated face content is more likely to surface as `AI_GENERATED`
- The frontend now explains the result instead of only showing a "threat" style warning

### How uploads are processed

- Images and GIFs are processed in memory by the backend
- Videos are temporarily written to disk during analysis and then deleted
- When running locally, inference happens inside the backend Python process you start from terminal/CMD
- When deployed, inference will happen on the deployed backend server, not on your laptop

## Known Issues / Limitations

- The system is still partly heuristic-heavy; it is not a fully trained production-grade authenticity classifier
- Non-face AI artwork may still be harder to classify reliably than AI-generated face images
- Some reported metrics are derived or heuristic, not all are direct model outputs
- The HuggingFace fallback options in `backend/models/huggingface_detector.py` are limited; only the currently working model loads cleanly in this setup
- Large models may increase cold-start time on deployment platforms
- CPU-only hosting will work, but inference will be slower than GPU hosting
- There is no database yet for upload history, saved reports, or user accounts
- There is no authentication, rate limiting, or abuse protection yet

## Deployment Steps

For the platform-specific version of these steps, use:

- `DEPLOYMENT.md` for GitHub Pages + Render

### Backend

1. Copy `backend/.env.example` to `backend/.env`
2. Set:
   - `APP_HOST`
   - `APP_PORT`
   - `PUBLIC_BASE_URL`
   - `CORS_ORIGINS`
   - `MAX_UPLOAD_SIZE_MB`
   - `LOG_LEVEL`
3. Install dependencies in the backend environment
4. Start the backend with `python main.py`
5. Verify:
   - `GET /health`
   - `GET /docs`
   - `POST /api/analyze`

### Frontend

1. Copy `frontend/.env.example` to `frontend/.env`
2. Set:
   - `REACT_APP_API_BASE_URL`
   - `REACT_APP_MAX_UPLOAD_MB`
3. Run `npm run build` for production
4. Deploy the built frontend
5. Confirm the live frontend can call the live backend

### Important deployment rule

Do not leave the frontend pointing to `localhost`. For live deployment, `REACT_APP_API_BASE_URL` must point to the deployed backend URL.

## Recommended Next Improvements

### High priority

- Replace or strengthen the AI-image fallback model, especially for non-face AI-generated images
- Add authentication and basic rate limiting
- Add structured request logging and error monitoring
- Add persistent storage or optional report saving
- Add deployment-specific start instructions for your target platform

### Product improvements

- Upload history for users
- Downloadable analysis report
- Admin dashboard
- API key support
- Better per-frame video explanation
- Confidence calibration

### Engineering improvements

- Split `backend/main.py` into smaller modules:
  - config
  - models
  - analyzers
  - scoring
  - routes
- Add automated tests for:
  - authentic sample
  - edited sample
  - AI-generated sample
  - unsupported file type
  - oversize upload
- Add a dedicated deployment guide for your chosen host

## Quick Verification Checklist

- Frontend loads on `localhost:3000`
- Backend health is healthy on `localhost:8000/health`
- Backend docs load on `localhost:8000/docs`
- Uploading a clean image returns `AUTHENTIC`
- Uploading an edited image can return `EDITED_ORIGINAL`
- Uploading AI-like face content can return `AI_GENERATED`
- Frontend shows category-specific result styling and explanation

## Suggested Next Action

Pick the deployment target first, then create a platform-specific deployment guide for:

- Render
- Railway
- Vercel + separate backend host
- VPS / EC2 / Azure VM

That should be the next document after this handoff.
