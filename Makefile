.PHONY: install data etl train stream drift serve test lint all clean
export MLFLOW_DISABLE_AGENT_HINT=1

install:  ## install package + dev tools
	pip install -e ".[dev]"
data:     ## generate synthetic raw transactions
	python -m propintel.data.generate
etl:      ## bronze -> silver -> gold (+ online feature store)
	python -m propintel.pipeline.medallion
train:    ## train, evaluate, track and register the model
	python -m propintel.model.train
stream:   ## replay listings through the real-time scorer
	python -m propintel.streaming.processor
drift:    ## drift-detection demo
	python scripts/demo_drift.py
serve:    ## run the API on :8000
	uvicorn propintel.serving.app:app --reload
test:
	pytest -q
lint:
	ruff check src tests
all: data etl train stream drift  ## full pipeline
clean:
	rm -rf data models reports mlruns .pytest_cache
