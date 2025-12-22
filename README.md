# CareerAlign
Resume-driven job matching web app that analyzes resumes and job descriptions to rank relevant roles using NLP-based similarity.

## Prerequisites
- Python 3.8+
- Node.js (16+) & npm

## Backend (Flask)
1. Create and activate a virtual environment:
- Powershell: 
```sh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2. Install required packages:
```sh
pip install Flask flask-cors
```
3. Run the backend:
```sh
python app.py
```
The backend serves GET /api/data on port 5000.

## Frontend (React + Vite)
1. Install dependencies and start dev server:
```sh
cd client
npm install
```
2. Open the Vite dev server
```sh
npm run dev
```
3. The frontend fetches `/api/data` and Vite proxies `/api` to the Flask backend during development.

## Quick checks
- Test backend directly: `curl http://localhost:5000/api/data`
- Start backend before running the frontend so the proxy can reach the API.