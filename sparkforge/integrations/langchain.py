"""LangChain adapter (optional; imports guarded so zero-dep still works)."""


class LangChainAdapter:
    """Expose SparkEngine.complete as a LangChain-compatible LLM-like callable.

    Only imports langchain lazily; without it the adapter reports unavailable.
    """

    def __init__(self, engine):
        self.engine = engine

    def invoke(self, prompt, **kwargs):
        return self.engine.complete(prompt, max_tokens=kwargs.get("max_tokens", 16))

    def __call__(self, prompt, **kwargs):
        return self.invoke(prompt, **kwargs)

    @staticmethod
    def available():
        try:
            import langchain  # noqa: F401
            return True
        except ImportError:
            return False
