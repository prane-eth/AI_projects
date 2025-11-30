# Agent Evaluation

Evaluate AI agents with structured metrics and workflows

## Setup

1. Set your OpenAI API key (or other LLM provider):
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

2. Install dependencies:
   ```bash
   pip install ragas openai
   ```

## Running the Example

Run the evaluation:
```bash
python app.py
```

Or run via the CLI:
```bash
ragas evals evals/evals.py --dataset test_data --metrics [metric_names]
```

## Project Structure

```
agent_evals/
├── app.py              # Your application code (RAG system, agent, etc.)
├── evals/              # Evaluation-related code and data
│   ├── evals.py       # Evaluation metrics and experiment definitions
│   ├── datasets/      # Test datasets
│   ├── experiments/   # Experiment results
│   └── logs/          # Evaluation logs and traces
└── README.md
```

This structure separates your application code from evaluation code, making it easy to:
- Develop and test your application independently
- Run evaluations without mixing concerns
- Track evaluation results separately from application logic

## Next Steps

1. Implement your application logic in `app.py`
2. Review and modify the metrics in `evals/evals.py`
3. Customize the dataset in `evals/datasets/`
4. Run experiments and analyze results
5. Iterate on your prompts and system design

## Documentation

Visit https://docs.ragas.io for more information.
