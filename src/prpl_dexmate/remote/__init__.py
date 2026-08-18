"""The RPC boundary between the orchestrator and the robot's skill server.

The orchestrator (a lab machine running bilevel planning) sends one directive per skill
invocation; the skill server (on the robot's Jetson) executes it at the full control
rate and reports a result. Nothing in this package streams per-control-step commands
across the network.
"""
