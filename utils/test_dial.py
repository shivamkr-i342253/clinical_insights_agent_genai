from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_endpoint="https://ai-proxy.lab.epam.com",
    azure_deployment="gpt-4o-mini-2024-07-18",
    api_key="dial-yxrfxj9qdrlx7h63n24avf9cwtg",
    api_version="2024-02-01",
    temperature=0
)

# Use the LLM
response = llm.invoke("Hello, world!")
print(response.content)
