from typing import Dict, Literal
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from pprint import pprint
from dotenv import load_dotenv, dotenv_values
from langgraph.checkpoint.memory import MemorySaver
from langsmith import traceable
from langchain_community.callbacks import get_openai_callback
import os
from aiohttp import ClientSession
from langchain_core.callbacks import AsyncCallbackManagerForToolRun
import timeit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter




load_dotenv(override=True, dotenv_path=".env")

os.getenv("OPENAI_API_KEY")
os.getenv("TAVILY_API_KEY")

# WEB SEARCH TOOL
web_search_tool = TavilySearchResults(k=2)


from pydantic import BaseModel, Field
from typing import List, Optional

# Input schema
# class StoryInput(BaseModel):
#     title: str = Field(..., description="The title of the story.")
#     genre: str = Field(..., description="The genre of the story, e.g., Action, Thriller, Sci-Fi.")
#     tone: str = Field(..., description="The overall tone, e.g., Dark, Suspenseful, Humorous.")
#     target_audience: str = Field(..., description="The target audience for the story, e.g., Adults, Teens.")
#     length: int = Field(..., description="The total duration of the video in seconds.")
#     key_themes: List[str] = Field(..., description="List of key themes such as Revenge, Betrayal, Love.")
#     main_characters: List[str] = Field(..., description="List of main characters and their descriptions.")
#     objective: str = Field(..., description="The main objective of the story.")
#     number_of_prompts: int = Field(..., description="Number of 5-second prompts to generate.")
#     climactic_twist: Optional[str] = Field(None, description="A specific twist to include in the story.")
#     setting_description: Optional[str] = Field(None, description="Description of the setting for worldbuilding.")

# Output schema for a single prompt
class StoryPrompt(BaseModel):
    prompt_number: int = Field(..., description="The sequential number of the prompt.")
    scene_type: str = Field(..., description="Type of scene, e.g., Emotional Setup, Action.")
    description: str = Field(..., description="A vivid, detailed description of the scene.")
    key_highlights: List[str] = Field(..., description="Key elements of the scene to focus on.")
    voice_over_text: str = Field(..., description="Voice-Over Narration for each 5-second scene.")
    

# Overall output schema
class StoryOutput(BaseModel):
    title: str = Field(..., description="The title of the story.")
    genre: str = Field(..., description="The genre of the story.")
    tone: str = Field(..., description="The overall tone.")
    total_prompts: int = Field(..., description="Total number of prompts generated.")
    prompts: List[StoryPrompt] = Field(..., description="List of prompts detailing the story.")

# Example for dynamic user preferences or system defaults
class UserPreferences(BaseModel):
    pacing: str = Field(..., description="Desired pacing, e.g., Fast, Medium, Slow.")
    focus_on: List[str] = Field(..., description="What aspects to emphasize, e.g., Action, Suspense.")
    output_format: Optional[str] = Field("Text", description="Preferred format, e.g., Text, JSON.")

# Example system defaults for fallback
class SystemDefaults(BaseModel):
    default_genre: str = "Action Thriller"
    default_tone: str = "Suspenseful"
    default_length: int = 180  # Seconds
    default_number_of_prompts: int = 36



def search_query(query):
    try:
        docs = web_search_tool.invoke({"query": query})
        web_docs, web_url = zip(*[(d["content"], d["url"]) for d in docs])
        return {"url": "\n".join(web_url), "vector_store": "\n".join(web_docs)}
    except Exception as e:
        print(f"Error searching query '{query}': {str(e)}")
        return None


class TopicGenerator(BaseModel):
    topic: str

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_for_topic = llm.with_structured_output(TopicGenerator)
structured_llm_for_topic_content = llm.with_structured_output(StoryOutput)




template_for_story_output = """
You are a master storyteller, screenwriter, and director tasked with creating a cinematic story for a 3-minute video. 
Your job is to craft an engaging narrative with detailed prompts for every 5-second scene. 
Follow these instructions:
Given a topic, your task is to generate the following key elements for the story:

1. **Story Details:**
   - **Title:** A captivating name for the story.
   - **Genre:** Choose a compelling genre (e.g., Action, Thriller, Sci-Fi, Romance).
   - **Tone:** Set the overall mood (e.g., Suspenseful, Dark, Emotional, Lighthearted).
   - **Target Audience:** Define the audience (e.g., Adults, Teens, Families).
   - **Key Themes:** Highlight 2-3 central themes (e.g., Revenge, Betrayal, Redemption, Love).
   - **Main Characters:** Create 2-3 compelling characters with names and brief descriptions.
   - **Objective:** Define the primary goal driving the story (e.g., "Escape a threat," "Overthrow a villain").
   - **Setting Description:** Provide a vivid and imaginative depiction of the story's setting.
   - **Climactic Twist:** Include a surprising twist that shocks or intrigues the audience.

2. **Narrative Output:**
   - Write a **Story Overview** summarizing the plot in 100-150 words.
   - Divide the story into **36 prompts**, each representing a 5-second scene.  
     For each scene, include:
     - **Scene Type:** Emotional Setup, Action, Suspense, Climax, etc.
     - **Description:** A vivid, detailed description of the visuals, sounds, character actions, and pacing.
     - **Key Highlights:** Mention 2-3 specific elements or moments to emphasize in the scene.

3. **Additional Instructions:**
   - Ensure the story flows seamlessly, with a clear buildup to a powerful climax.
   - Make the narrative immersive, visually stunning, and emotionally gripping.
   - The climax should leave the audience in awe and eager to share the story with others.

Now, generate the full story and its corresponding scene breakdown:
"""

story_output_prompt = ChatPromptTemplate.from_messages([
    ("system", template_for_story_output),
    ("user", "{topic}")
])

story_output_chain = story_output_prompt | structured_llm_for_topic_content


for output in story_output_chain.stream({"topic": "A story about a man who discovers a hidden treasure in the woods"}):
    print(output)