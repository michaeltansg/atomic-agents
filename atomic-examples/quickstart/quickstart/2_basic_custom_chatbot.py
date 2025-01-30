import os
import instructor
from langfuse.openai import OpenAI # Use the OpenAI class from the Langfuse library
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from atomic_agents.lib.components.agent_memory import AgentMemory
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseAgentOutputSchema

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
initial_message = BaseAgentOutputSchema(
    chat_message="How do you do? What can I do for you? Tell me, pray, what is your need today?"
)
memory.add_message("assistant", initial_message)

# OpenAI client setup using the Instructor library
# Note, you can also set up a client using any other LLM provider, such as Anthropic, Cohere, etc.
# See the Instructor library for more information: https://github.com/instructor-ai/instructor
client = instructor.from_openai(OpenAI())

# Instead of the default system prompt, we can set a custom system prompt
system_prompt_generator = SystemPromptGenerator(
    background=[
        "This assistant is a general-purpose AI designed to be helpful and friendly.",
    ],
    steps=["Understand the user's input and provide a relevant response.", "Respond to the user."],
    output_instructions=[
        "Provide helpful and relevant information to assist the user.",
        "Be friendly and respectful in all interactions.",
        "Always answer in rhyming verse.",
    ],
)
console.print(Panel(system_prompt_generator.generate_prompt(), width=console.width, style="bold blue"), style="bold blue")

# Agent setup with specified configuration
agent = BaseAgent(
    config=BaseAgentConfig(
        client=client,
        model="gpt-4o-mini",
        system_prompt_generator=system_prompt_generator,
        memory=memory,
    )
)

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

    # Process the user's input through the agent and get the response and display it
    response = agent.run(agent.input_schema(chat_message=user_input))
    agent_message = Text(response.chat_message, style="bold red")
    console.print(Text("Agent:", style="bold red"), end=" ")
    console.print(agent_message)
