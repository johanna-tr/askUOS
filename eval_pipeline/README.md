# Automated Evaluation Pipeline

*This folder contains an automated evaluation pipeline for comparing the performance of LLMs in the RAG-based chatbot askUOS.*

## Features

* A single workflow to compare a **baseline** model configuration with multiple **local** model configurations.
* Supports test queries in two languages (German and English).
* Comprehensive trace generation via customized recording processes. 
* Automated switching between LLM configurations during a single pipeline execution.
* Integrated Ragas evaluation metrics: topic adherence, tool call accuracy, tool call F1, and faithfulness.
* Structured outputs stored under `runs/` for comparison.
* Additional methods for running trace generation and Ragas evaluation separately.

## Getting Started

### Prerequisites

* Set up the chatbot askUOS (see askUOS/README.md).
* Install required dependencies:

  ```bash
  pip install -r eval_pipeline/eval_requirements.txt
  ```

### Configuration

1. Add your API keys and base URLs to your `.env.dev` file in the askUOS root project.

2. Copy the example files:

   ```bash
   cp eval_pipeline/eval_config_example.yaml eval_pipeline/eval_config.yaml
   cp eval_pipeline/data/questions_example.csv eval_pipeline/data/questions.csv
   ```

3. Edit the settings in `eval_config.yaml` (Pydantic-validated):
* Set a unique `run_name`.
* Define the **baseline model** configuration.
* Define one or more **test model** configurations to evaluate.
* For Ragas metrics: Define an evaluator model.

4. Edit `questions.csv`: Insert your test queries.

### How to Run an Evaluation

* Run the pipeline by executing the main evaluation script:

```bash
    python eval_pipeline/main_pipeline.py --config eval_pipeline/eval_config.yaml
```

* Inspect the results in the `/runs` folder (traces and metric results)


## Overview

The Pipeline workflow consists of two main phases: it first runs the baseline model configuration to generate reference traces and reference metric results, then it iterates through the local model configurations and computes the metrics for each trace.

### Folder Structure

The structure only displays the parts of the chatbot source code that contain pipeline-specific code.

```project-root/
askUOS/
├─ .env.dev                           # environment file for API keys etc.
├─ config.yaml                        # global settings
├─ eval_pipeline/                     # main pipeline folder
│   ├─ config/
│   │   ├─ eval_core_config.py
│   │   └─ eval_models.py
│   ├─ data/
│   │   └─ questions_example.csv      # queries for evaluation
│   ├─ metrics/
│   │   ├─ faithfulness.py
│   │   ├─ tool_call_accuracy_and_f1.py
│   │   └─ topic_adherence.py
│   ├─ runs/
│   ├─ visualizations/                # scripts for visualizing figures (not relevant for pipeline execution)
│   │   ├─ latency.py
│   │   ├─ recursion_limits.py
│   │   └─ requirements.txt           
│   ├─ eval_config_example.yaml       # pipeline settings
│   ├─ eval_helpers.py
│   ├─ eval_requirements.txt
│   ├─ main_pipeline.py           # pipeline main function
│   ├─ main_ragas.py              # standalone method for computing metrics on existing traces
│   ├─ main_traces.py             # standalone method for generating a trace (not part of pipeline workflow)
│   ├─ README.md
│   └─ trace_generator.py
├─ src/
│   └─ chatbot/
│       └─ agents/
│           ├─ utils/
│           │   └─ agent_helpers.py   # manages active LLM configurations and switches
│           └─ agent_lang_graph.py    # main agent script, contains wrapper methods for tracing

```

### Evaluation Metics: The Ragas Framework
Ragas is a RAG-specific evaluation framework. This pipeline uses four Ragas metrics to evaluate the performance of LLMs in askUOS.

1. **Topic Adherence**: Assesses the ability of a model to stay within predefined topics.
2. **Tool Call Accuracy**: Measures the accuracy of selecting the correct tools (tool name and arguments) for a given task.
3. **Tool Call F1**: Calculates the F1-score of a model's tool-call ability.
4. **Faithfulness**: Evaluates how well the statements in a model’s responses are supported by the retrieved context.

For additional details about these metrics, see the official documentation: [Ragas metrics documentation](https://github.com/vibrantlabsai/ragas/tree/main/docs/concepts/metrics/available_metrics)

---