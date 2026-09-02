TASK_UNDERSTANDING_PROMPT = """
You are a data scientist configuring a multiple linear regression.

User request: {user_query}
Available columns: {column_info}
Previous failed attempts: {attempt_history}

Choose one target column and one or more predictor columns. Use column names
exactly as supplied. Return JSON only:
{{
  "target_variable": "column name",
  "selected_variables": ["predictor 1", "predictor 2"],
  "context_analysis": "short reason"
}}

Do not repeat a previously failed predictor combination. Use the failure
history to propose a different set of predictors.
"""


EVALUATION_PROMPT = """
Summarize this multiple linear regression in no more than three sentences.
Test metrics: {metrics}
Coefficient table: {coefficient_table}
The required test MSE threshold is {mse_threshold}.
State clearly whether the model passed the threshold. Do not invent metrics.
"""
