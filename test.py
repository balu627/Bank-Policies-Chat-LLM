# from openai import OpenAI

# client = OpenAI(
#   api_key="your-openai-api-key"
# )

# response = client.responses.create(
#   model="gpt-5-nano",
#   input="Hi",
#   store=True,
# )

# print(response.output_text)

from google import genai

client = genai.Client(
    api_key="your-gemini-api-key"
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Hi",
)

print(response.text)