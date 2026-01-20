# CareerAlign
Resume-driven job matching web app that analyzes resumes and job descriptions to rank relevant roles using NLP-based similarity.

## Prerequisites
- Python 3.8+
- Node.js (16+) & npm

## Backend (Flask)
### Initial project setup
1. Create and activate a virtual environment:
- Powershell: 
```sh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2. Create and run Docker container:
Make sure Docker Desktop is running.
- Powershell:
```
docker run -d `
   --name pgvector `
   -e POSTGRES_PASSWORD=pwd `
   -p 5432:5432 `
   pgvector/pgvector:pg16
```
3. Register new server on pgAdmin 4 using credentials from (2)
- Server name can be whatever
- Host must be localhost
- Can keep port as 5432 or change it, but must verify that no other processes are running on it
- Username: postgres
- Password: (Whatever was chosen for POSTGRES_PASSWORD above)
You should now be able to see "postgres" under Databases.
4. Create global .env file with database URL
- URL should be formatted as follows:
DATABASE_URL = "postgresql://postgres:[pwd]@localhost:5432/postgres" 
5. Install required packages:
```sh
pip install -r backend/requirements.txt
```
6. Run the Docker container (if not already running):
Make sure Docker Desktop is running.
```sh
docker start pgvector
``` 
Verify that the container is running with
```sh
docker ps
```
7. Reactivate the Python virtual environment if needed
```sh
.\.venv\Scripts\Activate.ps1
```
8. Run the backend:
```sh
cd backend
uvicorn main:app --reload --port 5000
```
The backend serves GET /api/ping on port 5000 for testing purposes.

### Returning to project
After the initial project setup, the only steps that need to be completed to run the backend again are 6-8. Backend endpoints can be tested by visiting http://127.0.0.1:5000/docs.

## Frontend (React + Vite)
1. Install dependencies and start dev server:
```sh
cd frontend
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