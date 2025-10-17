# workshops_ai/wizard/__init__.py
"""
AI Wizard module for onboarding flow.

Provides endpoints for generating business services, assets, impacts, risks, controls, and evaluation.
Each step uses LLM with structured output and maintains state in Redis per org_id/user_id.
"""

__all__ = ["router"]

try:
    from workshops_ai.wizard.router import router
except ImportError:
    router = None
