.PHONY: install dev-api dev-web test lint evals build docker demo data

install:
	cd backend && pip install -r requirements-dev.txt
	cd frontend && npm ci

dev-api:
	cd backend && uvicorn app.main:app --reload --port 8000

dev-web:
	cd frontend && npm run dev

test:
	cd backend && pytest
	cd frontend && npm test

lint:
	cd backend && ruff check app tests evals
	cd frontend && npm run typecheck

evals:
	cd backend && python -m evals.run_evals

build:
	cd frontend && npm run build

docker:
	docker compose up --build

demo:
	python scripts/record_demo.py

data:
	python scripts/generate_sample_data.py
