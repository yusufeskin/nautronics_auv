"""Layer 1: sensor gates.

Each gate validates one sensor, converts it into the stack's conventions, and
reports its health. No gate integrates, fuses, or commands anything - that
separation is what lets the estimator be replaced without touching sensor
handling.
"""
