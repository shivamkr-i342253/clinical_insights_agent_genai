"""
Verification script for Azure OpenAI configuration.
Ensures all required environment variables are set and correct.
"""

import os
from dotenv import load_dotenv

# Force reload of .env file
load_dotenv(override=True)

print("=" * 70)
print("AZURE OPENAI CONFIGURATION VERIFICATION")
print("=" * 70)

# Check each environment variable
configs = {
    "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "AZURE_OPENAI_DEPLOYMENT": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    "AZURE_OPENAI_API_KEY": "***" if os.getenv("AZURE_OPENAI_API_KEY") else "NOT SET",
    "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION"),
}

all_set = True
for key, value in configs.items():
    status = "✅" if value and value != "NOT SET" else "❌"
    print(f"{status} {key:30} = {value}")
    if not value or value == "NOT SET":
        all_set = False

print("=" * 70)

if all_set:
    print("✅ All Azure OpenAI configurations are properly set!")
    api_version = configs["AZURE_OPENAI_API_VERSION"]
    if api_version and "2024-08-01" in api_version:
        print("✅ API version supports structured outputs (json_schema)")
    else:
        print(f"⚠️  API version {api_version} may not support structured outputs.")
        print("   Required: 2024-08-01-preview or later")
else:
    print("❌ Some required configurations are missing or incorrect!")
    print("\nPlease ensure your .env file contains:")
    print("  - AZURE_OPENAI_ENDPOINT=https://ai-proxy.lab.epam.com")
    print("  - AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini-2024-07-18")
    print("  - AZURE_OPENAI_API_KEY=your-api-key")
    print("  - AZURE_OPENAI_API_VERSION=2024-08-01-preview")

print("=" * 70)
