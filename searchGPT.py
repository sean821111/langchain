import os
from langchain.llms import OpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

from dotenv import load_dotenv


# get openai api key



load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERP_API_KEY")

def search_gpt(text):    
    # First, let's load the language model we're going to use to control the agent.
    llm = OpenAI(api_key=OPENAI_API_KEY, temperature=0)

    # Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
    # tools = load_tools(["serpapi", "llm-math"], llm=llm)
    tools = load_tools(["serpapi"], llm=llm)


    # Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    # Now let's test it out!
    # agent.run("What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?")

    agent.run(text)



if __name__ == '__main__':
    example_text = "Can you provide the company name manufacturer of portable detector wihch have monitor on it? answering me at least 3 companies names."
    
    result = search_gpt(example_text)
    print(result)