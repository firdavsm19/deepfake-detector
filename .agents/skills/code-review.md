---
name: code-review
description: Code review guidelines for deepfake-detector
triggers:
- /codereview
---

# Deepfake Detector — Code Review Guidelines

You are reviewing a Python-based deepfake detection project. Be thorough, direct, and actionable.

## Review Decisions

### When to APPROVE
- Documentation or comment-only changes
- Test-only changes that don't touch production code
- Config changes that follow existing patterns
- Minor refactors with no behavior change

### When to REQUEST CHANGES
- Any security vulnerability (exposed keys, unsafe inputs, unsafe deserialization)
- Broken or missing error handling around model inference
- Changes that would break the inference pipeline
- Missing input validation on uploaded files or media

### When to COMMENT
- Performance concerns in preprocessing or model loading
- Suggestions to improve code clarity
- Questions about design decisions
- Non-critical improvements worth considering

## Core Principles

1. **Correctness first**: The model must produce reliable predictions. Any change touching inference logic needs extra scrutiny.
2. **Security**: User-uploaded files are untrusted. Always validate file type, size, and content before processing.
3. **Simplicity**: Prefer readable code over clever code. ML pipelines are already complex enough.
4. **Performance awareness**: Model loading is expensive. Flag any code that loads models inside loops or per-request without caching.

## What to Check

### Machine Learning Code
- Models should be loaded once at startup, not per inference call
- Preprocessing steps must match what the model was trained on (normalization, resize, channel order)
- Random seeds should be set for reproducibility in experiments
- No hardcoded thresholds — use config or constants

### Security
- File uploads must be validated (extension, MIME type, max size)
- No user input should be passed directly to shell commands or file paths
- API keys and model paths must come from environment variables, never hardcoded
- Dependencies should be pinned in requirements files

### Error Handling
- All model inference calls must be wrapped in try/except
- Errors should be logged with context, not silently swallowed
- Return meaningful error messages to the caller, not stack traces

### Code Quality
- Functions should do one thing
- No functions longer than 50 lines without a good reason
- Type hints required on all function signatures
- No commented-out code in PRs

### Tests
- New inference logic must have unit tests
- Tests should use small dummy tensors, not real model weights
- Mock external dependencies (S3, APIs) in tests

## Repository Conventions
- Python 3.10+
- Use `black` for formatting, `ruff` for linting
- Tests live in `tests/`
- Use `torch.no_grad()` for all inference code
- Log using the standard `logging` module, not `print`