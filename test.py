import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

endpoint = os.getenv("AZURE_INFERENCE_ENDPOINT", "https://models.github.ai/inference")
# prefer environment-configured model names; keep original as default
model = os.getenv("OPENAI_MODEL", "openai/gpt-5")
# optional fallback model if the first model is unavailable
fallback_model = os.getenv("FALLBACK_MODEL", "gpt-4o-mini")
token = os.getenv("OPENAI_API_KEY")
if not token:
    raise RuntimeError("OPENAI_API_KEY (or equivalent) environment variable is not set")

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

def do_complete(chosen_model: str):
    return client.complete(
        messages=[
            SystemMessage("You are a helpful assistant."),
            UserMessage("What is google network connectivity centre?"),
        ],
        model=chosen_model,
    )


try:
    response = do_complete(model)
except HttpResponseError as e:
    err_text = str(e)
    print(f"Request failed for model '{model}': {err_text}")
    if "unavailable_model" in err_text or "Unavailable model" in err_text:
        print(f"Attempting fallback model '{fallback_model}'...")
        try:
            response = do_complete(fallback_model)
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            raise
    else:
        raise

print(response.choices[0].message.content)

