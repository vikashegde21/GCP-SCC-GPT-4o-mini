import os
import json
import signal
import subprocess
from dotenv import load_dotenv
from google.cloud import pubsub_v1
from openai import OpenAI

# Optional: Azure AI Inference support
try:
    from azure.ai.inference import ChatCompletionsClient
    from azure.core.credentials import AzureKeyCredential
    _AZURE_AVAILABLE = True
except Exception:
    _AZURE_AVAILABLE = False

load_dotenv()

# ---- Environment Variables ----
PROJECT_ID = os.environ.get("PROJECT_ID") or os.environ.get("GCP_PROJECT")
SUBSCRIPTION_ID = os.environ.get("PUBSUB_SUBSCRIPTION_ID", "scc-sub")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gpt-4o-mini")

# Azure configuration (default to Azure like test.py does)
AZURE_INFERENCE_ENDPOINT = os.environ.get("AZURE_INFERENCE_ENDPOINT", "https://models.github.ai/inference")
AZURE_API_KEY = os.environ.get("AZURE_API_KEY") or OPENAI_API_KEY

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set.")

# Use Azure Inference Client (works with GitHub token and OpenAI keys)
if not _AZURE_AVAILABLE:
    raise RuntimeError("Azure Inference SDK is not available. Install 'azure-ai-inference'.")

ai_client = ChatCompletionsClient(endpoint=AZURE_INFERENCE_ENDPOINT, credential=AzureKeyCredential(AZURE_API_KEY))
use_azure = True
client = None


# ---- Helper Functions ----
def execute_command(command: str, timeout: int = 30) -> dict:
    """Execute a terminal command and capture output"""
    try:
        print(f"\n[EXEC] Running command: {command}")
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd()
        )
        
        output = {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout[:2000] if result.stdout else "",  # Limit output
            "stderr": result.stderr[:2000] if result.stderr else "",
            "success": result.returncode == 0
        }
        
        print(f"[EXEC] Return code: {result.returncode}")
        if result.stdout:
            print(f"[EXEC] Output: {result.stdout[:500]}")
        if result.stderr:
            print(f"[EXEC] Error: {result.stderr[:500]}")
        
        return output
    except subprocess.TimeoutExpired:
        return {
            "command": command,
            "success": False,
            "error": f"Command timed out after {timeout} seconds"
        }
    except Exception as e:
        return {
            "command": command,
            "success": False,
            "error": str(e)
        }


def read_file(filepath: str, max_lines: int = 100) -> dict:
    """Read file contents"""
    try:
        if not os.path.exists(filepath):
            return {"success": False, "error": f"File not found: {filepath}"}
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()[:max_lines]
            content = ''.join(lines)
        
        return {
            "success": True,
            "filepath": filepath,
            "lines": len(lines),
            "content": content[:3000]  # Limit size
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def write_file(filepath: str, content: str) -> dict:
    """Write content to file"""
    try:
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "success": True,
            "filepath": filepath,
            "bytes_written": len(content.encode('utf-8'))
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def parse_scc_finding(data: dict) -> dict:
    """Extract and normalize SCC finding data"""
    # Handle both direct findings and wrapped payloads
    if isinstance(data, dict):
        # Try common SCC finding fields
        finding = {
            "name": data.get("name", data.get("finding_id", "Unknown")),
            "category": data.get("category", data.get("type", "Unknown")),
            "resource_name": data.get("resource_name", data.get("resource", "Unknown")),
            "severity": data.get("severity", data.get("state", "UNKNOWN")),
            "description": data.get("description", ""),
            "finding_class": data.get("finding_class", ""),
            "state": data.get("state", ""),
            "create_time": data.get("create_time", ""),
            "event_time": data.get("event_time", ""),
            "source_properties": data.get("source_properties", {}),
            "security_marks": data.get("security_marks", {}),
            "raw_data": data
        }
        return finding
    return {"error": "Invalid finding format", "raw_data": data}


def format_prompt(finding: dict) -> str:
    """Create comprehensive security analysis prompt"""
    if "error" in finding:
        return f"Error parsing finding: {finding.get('error')}\nRaw data: {json.dumps(finding.get('raw_data'), indent=2)}"
    
    return f"""You are an expert GCP Cloud Security Analyst with deep knowledge of Google Cloud Platform security best practices, compliance requirements, and threat landscape.

Analyze this Security Command Center (SCC) finding in detail:

**Finding Details:**
- Name: {finding.get('name', 'N/A')}
- Category: {finding.get('category', 'N/A')}
- Severity: {finding.get('severity', 'N/A')}
- Resource: {finding.get('resource_name', 'N/A')}
- Finding Class: {finding.get('finding_class', 'N/A')}
- State: {finding.get('state', 'N/A')}
- Created: {finding.get('create_time', 'N/A')}
- Source Properties: {json.dumps(finding.get('source_properties', {}), indent=2)}

**Your Analysis Should Include:**
1. **Executive Summary**: Concise description of the security issue
2. **Threat Level**: Rate as Critical, High, Medium, Low with justification
3. **Root Cause**: Technical explanation of why this issue exists
4. **Business Impact**: How this could affect GCP operations and compliance
5. **Affected Components**: What GCP services/resources are impacted
6. **Immediate Actions**: Steps to remediate within 24 hours
7. **Long-term Fix**: Permanent solution and prevention strategy
8. **Compliance Mapping**: Relevant compliance frameworks (CIS, PCI-DSS, HIPAA, SOC2, ISO27001)
9. **Detection & Monitoring**: How to detect similar issues in the future
10. **Terminal Commands**: If applicable, provide specific gcloud, kubectl, or other CLI commands to remediate

Provide actionable, specific guidance based on GCP security best practices.

**Available Tools:**
- You can execute gcloud commands to gather more information or remediate issues
- You can read/write files for configuration or documentation
- Use command execution to validate fixes or gather system info
"""


def ask_model(prompt: str, model_name: str):
    return ai_client.complete(
        messages=[
            {"role": "system", "content": "You are a GCP security analyst."},
            {"role": "user", "content": prompt},
        ],
        model=model_name,
    )


def callback(message: pubsub_v1.subscriber.message.Message):
    print("\n" + "="*70)
    print("NEW SCC FINDING RECEIVED")
    print("="*70)

    try:
        # Decode and parse the message
        message_data = json.loads(message.data.decode("utf-8"))
        print(f"\n[LOG] Raw message: {json.dumps(message_data, indent=2)}")
    except Exception as e:
        print(f"[ERROR] Failed to parse message JSON: {e}")
        message.ack()
        return

    # Parse SCC finding
    finding = parse_scc_finding(message_data)
    print(f"\n[PARSED] Finding Name: {finding.get('name')}")
    print(f"[PARSED] Category: {finding.get('category')}")
    print(f"[PARSED] Severity: {finding.get('severity')}")
    print(f"[PARSED] Resource: {finding.get('resource_name')}")

    # Create analysis prompt
    prompt = format_prompt(finding)

    # Call AI model for analysis
    print(f"\n[AI] Calling {MODEL} for analysis...")
    try:
        response = ask_model(prompt, MODEL)
        content = response.choices[0].message.content
    except Exception as e:
        print(f"[ERROR] Primary model ({MODEL}) failed: {e}")
        print(f"[INFO] Attempting fallback model ({FALLBACK_MODEL})...")
        
        try:
            response = ask_model(prompt, FALLBACK_MODEL)
            content = response.choices[0].message.content
        except Exception as e2:
            print(f"[ERROR] Fallback model also failed: {e2}")
            message.ack()
            return

    # Print analysis results
    print("\n" + "="*70)
    print("AI SECURITY ANALYSIS")
    print("="*70)
    print(content)
    print("="*70 + "\n")

    # Acknowledge message to Pub/Sub
    message.ack()
    print("[SUCCESS] Message acknowledged and processed.\n")


# ---- Main Program ----
def main():
    if not PROJECT_ID:
        raise RuntimeError("PROJECT_ID environment variable required")

    print("\n" + "="*70)
    print("GCP SECURITY COMMAND CENTER AI AGENT")
    print("="*70)
    print(f"[CONFIG] Project: {PROJECT_ID}")
    print(f"[CONFIG] Subscription: {SUBSCRIPTION_ID}")
    print(f"[CONFIG] AI Model: {MODEL}")
    print(f"[CONFIG] Endpoint: {AZURE_INFERENCE_ENDPOINT}")
    print(f"[AUTH] Using key.json credentials via GOOGLE_APPLICATION_CREDENTIALS")
    print("="*70)

    try:
        subscriber = pubsub_v1.SubscriberClient()
        subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)
        print(f"\n[CONNECTED] Listening for SCC findings on:")
        print(f"  {subscription_path}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Pub/Sub subscriber: {e}")
        raise

    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)

    def shutdown_handler(signum, frame):
        print("\n[SHUTDOWN] Interrupt signal received. Shutting down...")
        streaming_pull_future.cancel()
        print("[SHUTDOWN] Subscriber cancelled. Exiting.")

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    print("\n[STATUS] Waiting for findings. Press Ctrl+C to stop.\n")

    try:
        streaming_pull_future.result()
    except Exception as e:
        print(f"[STOPPED] Listener stopped: {e}")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    print("Choose Mode:")
    print("1. Run SCC Listener")
    print("2. Chat with GPT (CLI)")
    mode = input("Select 1 or 2: ")

    if mode == "1":
        main()

    elif mode == "2":
        print("Interactive Chat Mode Activated!")
        print("You can ask questions or request commands to be executed.")
        print("Commands: 'exec: <command>', 'read: <filepath>', 'write: <filepath>', 'exit'")
        print()
        
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                break

            # Handle terminal commands
            if user_input.startswith("exec:"):
                command = user_input[5:].strip()
                result = execute_command(command)
                print(f"\nCommand Result:\n{json.dumps(result, indent=2)}\n")
                continue
            
            if user_input.startswith("read:"):
                filepath = user_input[5:].strip()
                result = read_file(filepath)
                if result.get("success"):
                    print(f"\n[READ] {filepath}:\n{result.get('content')}\n")
                else:
                    print(f"\n[ERROR] {result.get('error')}\n")
                continue
            
            if user_input.startswith("write:"):
                parts = user_input[6:].strip().split(" ", 1)
                if len(parts) == 2:
                    filepath, content = parts
                    result = write_file(filepath, content)
                    print(f"\n[WRITE] {json.dumps(result, indent=2)}\n")
                else:
                    print("[ERROR] Format: write: <filepath> <content>\n")
                continue

            # Regular AI chat
            try:
                response = ask_model(user_input, MODEL)
                content = response.choices[0].message.content
                print(f"\nGPT:\n{content}\n")
            except Exception as e:
                print(f"Error calling model: {e}\n")