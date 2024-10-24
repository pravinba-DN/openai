from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the OPENAPI_KEY
openapi_key = os.getenv('OPENAPI_KEY')

if not openapi_key:
    raise Exception("OPENAPI_KEY is not set in environment variables")

# Use the key in your application logic
print(f"Your OpenAPI key is: {openapi_key}")
