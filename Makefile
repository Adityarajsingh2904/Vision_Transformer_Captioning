clean:
	find . -type d -name __pycache__ -exec rm -rf {} \+
	find . -name '*.pyc' -delete
