def register() -> None:
    from vllm.model_executor.model_loader import register_model_loader

    from sllm_store.vllm_loader import ServerlessLLMLoader

    register_model_loader("serverless_llm")(ServerlessLLMLoader)
