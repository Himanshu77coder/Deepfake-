# Deployment Guide

This project is set up to use:

- GitHub Pages for the frontend
- Render for the backend

## 1. Create and push the Git repository

From the project root:

```powershell
git init -b main
git add .
git commit -m "Prepare GitHub Pages and Render deployment"
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## 2. Deploy the backend on Render

1. In Render, choose **New +** -> **Blueprint**.
2. Connect your GitHub account and select this repository.
3. Render will read `render.yaml`.
4. When Render asks for `CORS_ORIGINS`, enter your future frontend URL:
   - `https://<your-username>.github.io/<your-repo>`
5. Finish the deploy and wait for the service to build.
6. Copy the live backend URL from Render, for example:
   - `https://deepfake-main-backend.onrender.com`

Render will use:

- `backend/requirements.txt` for dependencies
- `python main.py` as the start command
- `GET /health` as the health check

The backend now automatically uses Render's `PORT` and `RENDER_EXTERNAL_URL` environment values.

## 3. Deploy the frontend from GitHub

1. Open your GitHub repository.
2. Go to **Settings** -> **Secrets and variables** -> **Actions** -> **Variables**.
3. Add a repository variable:
   - `REACT_APP_API_BASE_URL` = your Render backend URL
4. Optional:
   - `REACT_APP_MAX_UPLOAD_MB` = `100`
5. Go to **Settings** -> **Pages**.
6. Ensure the source is **GitHub Actions**.
7. Push to `main` again, or run the workflow manually from **Actions**.

The workflow file is:

- `.github/workflows/deploy-frontend.yml`

It builds the React app from `frontend/` and publishes `frontend/build` to GitHub Pages.

## 4. Final check

After both deploys are live:

1. Open your GitHub Pages site.
2. Confirm the frontend loads.
3. Upload an image.
4. Verify the request reaches the Render backend.
5. Confirm the result appears in the UI.

## Notes

- If you change your repo name, your GitHub Pages URL also changes.
- If your frontend URL changes, update `CORS_ORIGINS` in Render.
- If your backend URL changes, update `REACT_APP_API_BASE_URL` in GitHub repository variables and redeploy the frontend.
