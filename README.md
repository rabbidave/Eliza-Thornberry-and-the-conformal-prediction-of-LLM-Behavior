# Eliza Thornberry and the conformal prediction of LLM Behavior
Python code for use in predicting a range of LLM outputs by dimension (e.g. sentiment, topic) across fixed domain values (e.g. positive, neutral, negative) or other attributes (e.g. account details, feedback, help needed), and thus monitoring the anticipated possible drift based on the most recent outputs, as a measure of LLM Drift Detection

[Colab Notebook Here](https://colab.research.google.com/github/rabbidave/Eliza-Thornberry-and-the-conformal-prediction-of-LLM-Behavior/blob/main/Eliza.ipynb)

## ♫ The Dream of the 90's ♫ is alive in ~~Portland~~ ["a weird suite of Enterprise LLM tools"](https://github.com/users/rabbidave/projects/1) named after [Nicktoons](https://en.wikipedia.org/wiki/Nicktoons)
### by [some dude in his 30s](https://www.linkedin.com/in/davidisaacpierce)
#
## Utility 5) # Eliza Thornberry and the conformal prediction of LLM Behavior

<img src="https://static.wikia.nocookie.net/wildthornberrys/images/9/97/Eliza_Thornberry.png/revision/latest/scale-to-width-down/1000?cb=20220424052936" alt="Eliza" title="Eliza" width="40%">

## Description:
A python script to for demonstrating how conformal prediction can be used in forecasting LLM drift at different prediction intervals; to be run (and reconfigured using our actual outputs) either in batch or upon breach of a threshold (e.g. X-number of net-new outputs to the configured directory)

i.e. affect batch or event-driven prediction of the range of possible values; could be configured to raise an exception as part of a pipeline, publish to an SNS queue, etc
#
## Rationale:

1) User experience, instrumentation, and metadata capture are crucial to the adoption of LLMs for orchestration of [multi-modal agentic systems](https://en.wikipedia.org/wiki/Multi-agent_system); predicting the range of possible values at set prediction intervals allows for early warning of LLM Drift
## Intent:

The intent of "Eliza Thornberry and the conformal prediction of LLM Behavior" is to efficiently calculate needed values for evaluation of LLM Drift.

The goal being to detect if the model is likely to experience drift at those set prediction intervals
    
If incorporated into a pipeline or event-drive threshold it could be used to give early warning or drifting outputs from either a fixed (read: original) set of values or from a set window (e.g. previous 24 hours)

#
### Note: Needs additional error-handling; this is mostly conceptual and assumes the use of environment variables rather than hard-coded values for prediction intervals/p-values.