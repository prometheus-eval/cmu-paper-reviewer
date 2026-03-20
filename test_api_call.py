from backend.services.review_service import ReviewService
import openhands.sdk.llm.llm as sdk

resp = sdk.litellm_completion(
    model="litellm_proxy/azure_ai/gpt-5.4",
    api_key="sk-8o5iAmbfQqtIiRC05p9rcA",
    api_base="https://cmu.litellm.ai",
    messages=[{"role": "user", "content": "hi"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "t",
            "description": "t",
            "parameters": {"type": "object", "properties": {}},
        },
    }],
    tool_choice="auto",
)
print("GPT-5.4 TOOL CALL OK")
