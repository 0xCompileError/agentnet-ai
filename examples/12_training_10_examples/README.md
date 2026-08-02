# Training 10 Examples

Train over a 10-example dataset through the same beginner-facing `an.train`
API used for ordinary Python `X` and `y` values.

```bash
uv run python examples/12_training_10_examples/main.py
```

Use OpenAI for real LLM calls by opting in explicitly:

```bash
AGENTNET_TRAINING_LLM=openai \
uv run --env-file .env python examples/12_training_10_examples/main.py
```

The `.env` file must define `OPENAI_API_KEY` and `OPENAI_MODEL`. AgentNet does
not load `.env` files itself; `uv` supplies those values for this command. The
script writes progress to stderr and keeps its final JSON result on stdout.
