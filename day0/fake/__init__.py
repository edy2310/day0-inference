"""Built-in fake OpenAI-compatible server and its toy language model.

The fake server lets every layer of day0 (client, parsers, backends, metrics, gate, report) run
without a GPU. It is deterministic, and its outcomes are calibrated so that tests can assert them.
"""
