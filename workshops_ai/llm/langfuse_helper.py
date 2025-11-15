from __future__ import annotations

import logging
from typing import Optional, Any
from functools import lru_cache

logger = logging.getLogger(__name__)

_langfuse_client: Optional[Any] = None


def get_langfuse_client():
    """
    Get or create the global Langfuse client.

    Returns None if Langfuse is disabled or credentials are missing.
    """
    global _langfuse_client

    if _langfuse_client is not None:
        return _langfuse_client

    from workshops_ai.settings import get_settings
    settings = get_settings()

    if not settings.langfuse.enabled:
        logger.debug("Langfuse is disabled")
        return None

    if not settings.langfuse.public_key or not settings.langfuse.secret_key:
        logger.warning("Langfuse enabled but missing public_key or secret_key")
        return None

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=settings.langfuse.public_key,
            secret_key=settings.langfuse.secret_key,
            host=settings.langfuse.base_url,
        )

        logger.info(
            f"Langfuse client initialized: {settings.langfuse.base_url}"
        )
        return _langfuse_client

    except ImportError:
        logger.warning("langfuse package not installed, skipping")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse client: {e}")
        return None


def get_langfuse_callback_handler():
    """
    Get a Langfuse CallbackHandler for LangChain tracing.

    Returns:
        CallbackHandler instance or None if Langfuse is disabled/unavailable

    Note:
        Session_id and metadata should be set using propagate_attributes()
        context manager when invoking the chain.
    """
    client = get_langfuse_client()
    if client is None:
        return None

    try:
        from langfuse.langchain import CallbackHandler

        handler = CallbackHandler()
        logger.debug("Langfuse CallbackHandler created")
        return handler

    except ImportError:
        logger.warning("langfuse.langchain not available")
        return None
    except Exception as e:
        logger.error(f"Failed to create Langfuse CallbackHandler: {e}")
        return None


def get_langchain_prompt_from_langfuse(
    prompt_name: str,
    fallback_template: Optional[str] = None,
    fallback_variables: Optional[list[str]] = None,
    version: Optional[int] = None,
    label: Optional[str] = None,
) -> Optional[Any]:
    """
    Fetch a prompt from Langfuse and convert to LangChain format.

    Args:
        prompt_name: Name of the prompt in Langfuse
        fallback_template: Template string to use if Langfuse fetch fails
        fallback_variables: List of variable names for fallback template
        version: Specific version to fetch (default: latest production)
        label: Label to fetch (e.g., "production", "staging")

    Returns:
        LangChain PromptTemplate or ChatPromptTemplate or None if both Langfuse and fallback fail
    """
    client = get_langfuse_client()

    # Try Langfuse first
    if client is not None:
        try:
            logger.debug(f"Fetching prompt '{prompt_name}' from Langfuse")

            fetch_kwargs = {"name": prompt_name}
            if version is not None:
                fetch_kwargs["version"] = version
            if label is not None:
                fetch_kwargs["label"] = label

            langfuse_prompt = client.get_prompt(**fetch_kwargs)

            # Check if it's a Chat Prompt (has messages) or Text Prompt
            if hasattr(langfuse_prompt, 'prompt') and isinstance(langfuse_prompt.prompt, list):
                # Chat Prompt - use Langfuse conversion and set metadata as attribute
                from langchain_core.prompts import ChatPromptTemplate

                # Convert to Langchain prompt (get_langchain_prompt already converts {{}} -> {})
                langchain_prompt = ChatPromptTemplate.from_messages(
                    langfuse_prompt.get_langchain_prompt()
                )

                # Add metadata as attribute for linking generations
                langchain_prompt.metadata = {"langfuse_prompt": langfuse_prompt}

                logger.info(
                    f"Successfully loaded Chat Prompt '{prompt_name}' from Langfuse "
                    f"(version: {langfuse_prompt.version})"
                )
                return langchain_prompt
            else:
                # Text Prompt - use built-in conversion
                langchain_prompt = langfuse_prompt.get_langchain_prompt()

                logger.info(
                    f"Successfully loaded prompt '{prompt_name}' from Langfuse "
                    f"(version: {langfuse_prompt.version})"
                )
                return langchain_prompt

        except Exception as e:
            logger.warning(
                f"Failed to fetch prompt '{prompt_name}' from Langfuse: {e}. "
                f"Using fallback."
            )

    # Fallback to local template
    if fallback_template is not None:
        try:
            from langchain_core.prompts import PromptTemplate

            logger.info(f"Using fallback template for prompt '{prompt_name}'")
            return PromptTemplate(
                template=fallback_template,
                input_variables=fallback_variables or [],
            )
        except Exception as e:
            logger.error(f"Failed to create fallback prompt: {e}")
            return None

    logger.error(f"No prompt available for '{prompt_name}' (Langfuse unavailable, no fallback)")
    return None


def flush_langfuse():
    """
    Flush any pending Langfuse events.

    Call this during shutdown to ensure all traces are sent.
    """
    client = get_langfuse_client()
    if client is not None:
        try:
            client.flush()
            logger.info("Langfuse events flushed")
        except Exception as e:
            logger.error(f"Failed to flush Langfuse events: {e}")
