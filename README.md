# Multi-variable Regression Agent

A LangGraph-based multiple linear regression agent with automated data
cleaning, optional Gemini-assisted variable selection, statistical feature
pruning, retry memory, and an untouched final test-set gate.

## Workflow

```text
load CSV
  -> select variables
  -> clean and split (train / validation / test)
  -> fit statsmodels OLS
  -> evaluate train and validation MSE + feature p-values
  -> retry variable selection when needed
  -> evaluate once on the final test set
```

Each failed attempt is retained in `attempt_history`, including the selected
variables, train and validation MSE, feature p-values, weak features, and the
failure reasons. Retry selection avoids previously rejected combinations and
uses backward elimination before exploring other subsets.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For Gemini-assisted selection, create `.env` locally:

```dotenv
GEMINI_API_KEY=your-key
```

The API key and model name are configured in `configs/setting.py`. All other
human-tunable settings are in `configs/agent_config.py`.

## Run

Run with the unified configuration:

```bash
python main.py
```

Or override it for one run:

```bash
python main.py test/test.csv \
  --target target \
  --features x1 x2 category \
  --mse-threshold 5 \
  --max-iterations 5 \
  --no-use-llm
```

The command exits with status `0` when the final MSE gate passes and status `2`
when it fails.

## Tests

```bash
python -m pytest -q
```

The tests cover the MSE gate, retry exhaustion, recovery from a poor initial
feature set, and automatic pruning of statistically insignificant variables.
