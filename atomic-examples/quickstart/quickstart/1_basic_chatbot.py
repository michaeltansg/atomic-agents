import os
import instructor
from langfuse.openai import OpenAI # Use the OpenAI class from the Langfuse library
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from atomic_agents.lib.components.agent_memory import AgentMemory
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseAgentInputSchema, BaseAgentOutputSchema

from dotenv import load_dotenv
load_dotenv()

from langfuse import Langfuse

langfuse = Langfuse(
  secret_key = os.getenv("LANGFUSE_SECRET_KEY"),
  public_key = os.getenv("LANGFUSE_PUBLIC_KEY"),
  host = os.getenv("LANGFUSE_HOST")
)

# Initialize a Rich Console for pretty console outputs
console = Console()

# Memory setup
memory = AgentMemory()

# Initialize memory with an initial message from the assistant
initial_message = BaseAgentOutputSchema(chat_message="Hello! How can I assist you today?")
memory.add_message("assistant", initial_message)

# OpenAI client setup using the Instructor library
client = instructor.from_openai(OpenAI())

# Agent setup with specified configuration
agent = BaseAgent(
    config=BaseAgentConfig(
        client=client,
        model="gpt-4o-mini",
        memory=memory,
    )
)

# Generate the default system prompt for the agent
default_system_prompt = agent.system_prompt_generator.generate_prompt()
# Display the system prompt in a styled panel
console.print(Panel(default_system_prompt, width=console.width, style="bold blue"), style="bold blue")

# Display the initial message from the assistant
console.print(Text("Agent:", style="bold red"), end=" ")
console.print(Text(initial_message.chat_message, style="bold red"))

# Start an infinite loop to handle user inputs and agent responses
while True:
    # Prompt the user for input with a styled prompt
    user_input = console.input("[bold blue]You:[/bold blue] ")
    # Check if the user wants to exit the chat
    if user_input.lower() in ["/exit", "/quit"]:
        console.print("Exiting chat...")
        break

    # Process the user's input through the agent and get the response
    input_schema = BaseAgentInputSchema(chat_message=user_input)
    response = agent.run(input_schema)

    agent_message = Text(response.chat_message, style="bold red")
    console.print(Text("Agent:", style="bold red"), end=" ")
    console.print(agent_message)
