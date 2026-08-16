PYTHON ?= python
PIP ?= $(PYTHON) -m pip
STREAMLIT ?= streamlit

MODEL_NAME ?= openai-community/gpt2-medium
ALPHA ?= 5.0
BLOCK_SIZE ?= 128
REPORT_DIR ?= outputs
RUN_NAME ?= demo-cpu
DEVICE ?= cpu
TARGET_TEXT ?= Without_GPU/data/synthetic_universe/target_corpus.txt
SWEEP_DIR ?= outputs/alpha_sweep
MODEL_NAMES ?= openai-community/gpt2-medium gpt2
BENCHMARK_DIR ?= outputs/model_benchmark

.PHONY: setup test demo-cpu alpha-sweep model-benchmark dashboard

setup:
	$(PIP) install --upgrade pip
	$(PIP) install -e .[dev]

test:
	$(PYTHON) -m compileall Without_GPU dashboard tests
	pytest

demo-cpu:
	$(PYTHON) Without_GPU/main.py \
		--target_text $(TARGET_TEXT) \
		--model_name $(MODEL_NAME) \
		--alpha $(ALPHA) \
		--block_size $(BLOCK_SIZE) \
		--device $(DEVICE) \
		--report_dir $(REPORT_DIR) \
		--run_name $(RUN_NAME)

alpha-sweep:
	$(PYTHON) -m Without_GPU.experiments.run_alpha_sweep \
		--target_text $(TARGET_TEXT) \
		--model_name $(MODEL_NAME) \
		--device $(DEVICE) \
		--block_size $(BLOCK_SIZE) \
		--sweep_dir $(SWEEP_DIR)

model-benchmark:
	$(PYTHON) -m Without_GPU.experiments.run_model_benchmark \
		--target_text $(TARGET_TEXT) \
		--models $(MODEL_NAMES) \
		--device $(DEVICE) \
		--alpha $(ALPHA) \
		--block_size $(BLOCK_SIZE) \
		--benchmark_dir $(BENCHMARK_DIR)

dashboard:
	$(STREAMLIT) run dashboard/app.py
