# Media Authenticity Analyzer

A full-stack web application for analyzing uploaded images, GIFs, and videos and classifying them as authentic, edited, or AI-generated.

This project combines a FastAPI backend, a React frontend, neural-network inference, and classical computer-vision heuristics to provide an explanation-first media authenticity workflow. It is designed for demos, academic work, and portfolio projects rather than as a certified forensic product.

## Live Deployment

- Frontend: [https://himanshu77coder.github.io/Deepfake-/](https://himanshu77coder.github.io/Deepfake-/)
- Backend API: [https://himanshu07coder-deepfake-backend-api.hf.space](https://himanshu07coder-deepfake-backend-api.hf.space)
- API docs: [https://himanshu07coder-deepfake-backend-api.hf.space/docs](https://himanshu07coder-deepfake-backend-api.hf.space/docs)
- Health check: [https://himanshu07coder-deepfake-backend-api.hf.space/health](https://himanshu07coder-deepfake-backend-api.hf.space/health)

## What This Project Does

The application accepts uploaded media and returns a classification with supporting signals and summary text. Current result categories are:

| Label | Meaning |
| --- | --- |
| `AUTHENTIC` | Likely original media with no strong manipulation signal |
| `EDITED_ORIGINAL` | Likely real media that has been altered or edited |
| `AI_GENERATED` | Likely synthetic or AI-generated media |

The UI presents:

- media preview before upload
- backend connection status
- manipulation and authenticity scoring
- category-aware result styling
- explanation text under "Why this result"
- per-frame analysis for GIFs when available

## Key Features

- Image, GIF, and video upload support
- FastAPI backend with `POST /api/analyze`
- React frontend designed for both localhost and deployed use
- Face-aware ensemble inference for detected faces
- Frequency-domain and lighting-consistency checks
- Temporal checks for videos and animated GIFs
- Cleaner output labels than a binary "threat detected" workflow
- Environment-based configuration for frontend and backend
- Live deployment using GitHub Pages and Hugging Face Spaces

## High-Level Pipeline

1. The frontend uploads a selected file to the backend.
2. The backend validates file type and size.
3. Media is analyzed using a mix of:
   - face detection
   - ensemble model scoring
   - frequency-domain checks
   - lighting and artifact checks
   - temporal heuristics for videos and GIFs
4. The backend combines signals into a final classification and explanation payload.
5. The frontend renders the result with scores, reasons, and category-specific styling.

## Tech Stack

### Frontend

- React
- JavaScript
- CSS
- Lucide React icons

### Backend

- FastAPI
- Uvicorn
- PyTorch
- OpenCV
- facenet-pytorch
- timm
- Pillow / ImageIO / NumPy

### Hosting

- GitHub Pages for the frontend
- Hugging Face Spaces for the backend

## Repository Structure

```text
Deepfake-main/
|-- backend/
|   |-- main.py
|   |-- requirements.txt
|   |-- .env.example
|   `-- models/
|       `-- huggingface_detector.py
|-- frontend/
|   |-- src/
|   |   |-- App.js
|   |   `-- config.js
|   `-- .env.example
|-- HANDOFF.md
|-- DEPLOYMENT.md
`-- README.md
```

## Local Development

### 1. Clone the repository

```powershell
git clone https://github.com/Himanshu77coder/Deepfake-.git
cd Deepfake-
```

### 2. Start the backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

Backend default URLs:

- `http://localhost:8000`
- `http://localhost:8000/docs`
- `http://localhost:8000/health`

### 3. Start the frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm start
```

Frontend default URL:

- `http://localhost:3000`

## Configuration

Example environment files are included:

- `backend/.env.example`
- `frontend/.env.example`

### Backend variables

- `APP_HOST`
- `APP_PORT`
- `PUBLIC_BASE_URL`
- `CORS_ORIGINS`
- `MAX_UPLOAD_SIZE_MB`
- `LOG_LEVEL`

### Frontend variables

- `REACT_APP_API_BASE_URL`
- `REACT_APP_MAX_UPLOAD_MB`

Note: the frontend is configured to use `http://localhost:8000` during local development and the deployed Hugging Face backend when running from the live site.

## API Summary

### `GET /health`

Returns backend health and model readiness information.

### `GET /docs`

Interactive FastAPI Swagger documentation.

### `POST /api/analyze`

Uploads an image, GIF, or video and returns analysis results.

Example request:

```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "file=@sample.jpg"
```

## Deployment Overview

Current live stack:

- Frontend: GitHub Pages
- Backend: Hugging Face Spaces

Important deployment rules:

- Do not leave the frontend pointed at localhost for production.
- The backend must allow your frontend origin in `CORS_ORIGINS`.
- Free hosting may introduce cold-start delays after inactivity.

## Current Limitations

- This is still partly heuristic-driven and should not be treated as a final forensic authority.
- Face-centric media is handled more reliably than non-face AI artwork.
- Edited-image detection is improved, but not perfect for every real-world edit.
- Free-tier hosting may be slower than local or GPU-backed deployment.
- There is no authentication, rate limiting, or user history layer yet.

## Recommended Next Improvements

- Add stronger AI-image detection for non-face synthetic content
- Split the backend into smaller modules
- Add automated tests for authentic, edited, and AI-generated samples
- Add request logging, rate limiting, and monitoring
- Add downloadable reports and optional upload history

## License

This project is intended for educational and research use.
