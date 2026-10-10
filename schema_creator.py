# pip install -U langchain langchain-google-genai pydantic
import os
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv


load_dotenv()

class DatasetRequirements(BaseModel):
    """Requirements for finding datasets to train an ML model."""
    search_terms: list[str] = Field(
        description="Search terms for relevant datasets based on the user query"
    )
    dataset_type: list[str] = Field(
        description="The type of image dataset needed (Classification, Detection, Contrastive Learning)"
    )
    classifiers: list[str] = Field(
        description="The classes/labels that need to be present within the data"
    )

SYSTEM_PROMPT = (
    "You are an AI agent that extracts the features other agents need to search "
    "for relevant datasets. Based on the user's stated purpose, populate the fields "
    "with the data they would need to train an ML model for that purpose."
)

def queryRequirements(sample_limit, query):
    model = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",  # or any other Gemini model you have access to
        google_api_key=os.environ["GOOGLE_API_KEY"],  # or paste your key directly
        temperature=0,
    )

    agent = create_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        response_format=DatasetRequirements,
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )

    requirements = result["structured_response"]
    print(requirements)
    return requirements

if __name__ == "__main__":
    queryRequirements(100, "I want to train a model that can detect if my dog is on the counter")
