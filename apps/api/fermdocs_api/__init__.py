"""fermdocs-api — FastAPI backend for the hypothesis-stage frontend.

Plan ref: plans/2026-05-03-hypothesis-debate-v0.md (v0.5a: backend).

Wraps the existing fermdocs ingest → characterize → diagnose → hypothesize
pipeline behind a small HTTP/WebSocket surface so a Next.js frontend can
upload bundles, watch the debate live, answer open questions, and see
final hypotheses.

Stage 4-friendly: zero changes to the underlying pipeline. The API is a
thin orchestration layer over already-tested CLI surfaces.
"""
