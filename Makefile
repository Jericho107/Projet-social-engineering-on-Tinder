install:
	python -m pip install -e ".[dev]"

lint:
	ruff check app.py src/governance.py tests

test:
	pytest -q

compile:
	python -m py_compile app.py src/governance.py

validate: compile lint test

dashboard:
	streamlit run app.py
