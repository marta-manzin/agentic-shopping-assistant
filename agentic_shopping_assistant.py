# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-editable,-slideshow,-colab
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/marta-manzin/agentic-shopping-assistant/blob/main/agentic_shopping_assistant.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="TOnQyK7obg_s"
# # 🛒 Agentic Shopping Assistant
#
# This notebook will go through all the steps to create an agentic shopping assistant. \
# We will:
# 1. Connect to OpenAI
# 2. Create a simple agent
# 3. Create an MCP server
# 4. Create a LangGraph agent
# <br/>
# <img src="https://drive.google.com/uc?export=view&id=1to-6-8fnbAJ9bLTBWSf5buay6d2h94qw" width="500">
#

# %% [markdown] id="4irdiFHM8Nz7"
# # ⚙️ Setup

# %% [markdown] id="GBEKX8f4UE8F"
# Setup
# Before we start using OpenAI models, you need to set an API key. \
# If you don't already have an key, you can generate one at: https://platform.openai.com/api-keys. \
# Save the key as a Colab Secret variable called "OPENAI_API_KEY":
# 1. Click on the key icon in the left bar menu.
# 2. Click on `+ Add new secret`.
# 3. Name the variable and paste the key in the value field.
# 4. Enable notebook access.
#
# <img src="https://drive.google.com/uc?export=view&id=1lMPgLbeqZ1lxYMQwbe5F3n9Qko4u55FH" width="450">
#
#
#

# %% [markdown] id="kZxTvnYm8Q1O"
# Let's test it. First, import the key into the notebook:

# %%
import os
try:
  from google.colab import userdata
  os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
except:
  pass

# %%
# Where are we running
from IPython import get_ipython
if get_ipython() is not None:
    IN_JUPYTER = True
else:
    IN_JUPYTER = False
("IN_JUPYTER:", IN_JUPYTER)

# %% [markdown] id="aIdi2xjLbWOX"
# Then, make a test call to OpenAI:

# %% id="FvlxGVjz8S7l" outputId="59d7e89b-f3f3-4eef-c05c-43fe696cf0a1"
import openai
client = openai.OpenAI()
model = "gpt-4o"

# Test that the LLM is set up correctly
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Say 'OK' if you can read this."}],
    max_tokens=10
)
print(f"LLM test: {response.choices[0].message.content}")

# %% [markdown] id="4ToWYG3qNDzH"
# # 🤖 Creating an Agent

# %% [markdown] id="ATPYxIUbIigj"
# In Python, a set is an unordered collection of unique elements. \
# We will build an agent that adds and removes strings from a set.

# %% [markdown] id="ujra-LLDcaeP"
# The System Prompt gives some context to the LLM.

# %% id="u2p8auWt_fGE"
SYSTEM_PROMPT = """
You are a helpful assistant that adds and removes strings from a set.

You have access to tools that let you:
1. Add a string, if it is not already in the set.
2. Remove a string.
"""

# %% [markdown] id="XwzdXKyycs0P"
# Here are the available tools:

# %% id="eM8NowbiOv74"
MY_SET = set()

def insertion_tool(s: str):
  """Tool: Add a string to a set."""
  MY_SET.add(s)

def removal_tool(s: str):
  """Tool: Remove a string from a set."""
  if s in MY_SET:
    MY_SET.remove(s)


# %% [markdown] id="AOlTPB1RcVDH"
# Provide a description of each tool to the LLM. \
# The LLM will use it to decide which tools to call and with what arguments.

# %% id="qY3nlJhU-x7U"
tools = [
    {
        "type": "function",
        "function": {
            "name": "insertion_tool",
            "description": "Add a string to a set.",
            "parameters": {
                "type": "object",
                "properties": {
                    "s": {
                        "type": "string",
                        "description": "The string to be added."
                    },
                },
                "required": ["s"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "removal_tool",
            "description": "Add a string to a set.",
            "parameters": {
                "type": "object",
                "properties": {
                    "s": {
                        "type": "string",
                        "description": "The string to be removed."
                    },
                },
                "required": ["s"]
            }
        }
    },
]

# %% [markdown] id="FTe3UobJc8DF"
# If the LLM decides to run a tool, it will respond with a "tool call" object. \
# A tool call looks like this:
#
# ```
# {
#   id: <unique-id>,
#   function: {
#     arguments: '{"s":"my_string"}',
#     name: 'insertion_tool'
#   },
#   type: 'function'
# }
# ```
#
# The following code parses a tool call and runs the tool.
#
#

# %% id="u5YkPNiU_iKR"
import json

def execute(tool_call) -> str:
    """Execute a tool call and return the result, if any."""
    # Extract the function name from the tool call
    function_name = tool_call.function.name

    # Parse the arguments from JSON string to dictionary
    arguments = json.loads(tool_call.function.arguments)

    # Look up the function by name in the global scope
    tool_func = globals().get(function_name)

    # Check if the function exists and is callable
    if tool_func is None or not callable(tool_func):
        return f"Unknown function: {function_name}"

    # Call the function with the unpacked arguments
    response = tool_func(**arguments)

    # Return the result of the function call, if any
    if response:
      return str(response)
    else:
      return ""



# %% [markdown] id="QyOsu-4tleGP"
# And last, the agent logic. \
# Instead of using a ready-made framework, the code below does *direct orchestration*.

# %% id="u4B9C8CKXmOm"
import itertools

def submit_request(
    user_prompt: str,
    verbose: bool = True
    ):
    """Submit a request to the agent and run any tools it calls."""
    # Initialize the chat history
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    for iteration in itertools.count(1):

        # Ask the agent what to do next
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        ).choices[0].message

        # Update the chat history with the agent's response
        messages.append({
            "role": "assistant",
            "content": response.content,
            "tool_calls": response.tool_calls
        })

        # If agent did not call any tools, we are done
        if not response.tool_calls:
            if verbose:
              print(f"\n⭐ The resulting set is: {MY_SET}")
            break

        # Execute all tool calls
        for tool_call in response.tool_calls:
            if verbose:
              print(f"\n🔧 The agent is calling a tool: "
                  f"{tool_call.function.name}"
                  f"({json.loads(tool_call.function.arguments)})")

            outcome = execute(tool_call)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(outcome)
            })


# %% [markdown] id="NF26AUcTeM9C"
# Let's test the agent!

# %% id="tfQWYK7ceHVY" outputId="5a260f5c-ba5c-4b83-fa20-d763911a678a"
submit_request("Please add 'apples', 'oranges' and 'pears' to the set.")

# %% id="dnCP2npeeYev" outputId="b3d7b53a-00c0-48ec-eb80-a13ed9ece4dd"
submit_request("Please remove 'oranges' from the set.")

# %% [markdown] id="W_P3GEVCNTQ6"
# # 🗄️ Creating an MCP Server

# %% [markdown] id="rD6I4qqM3-K-"
# Create the MCP server.

# %% id="Xem5Szb0AQqv" outputId="f8e8d00c-7514-41e9-9b11-dc2d53e0acb2"
# %pip install --quiet mcp
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("set-server")
print("✓ Server created")


# %% [markdown] id="PuRlZnAYATtj"
# Create an MCP wrapper for listing the available tools.

# %% id="R4RQ-E6QArjt" outputId="93c8a3c4-94a1-49e3-c0f3-c3e72f11160e"
async def list_tools() -> list[Tool]:
    """Return the list of available tools from our tools definition."""
    # Create an empty list to store MCP Tool objects
    mcp_tools = []

    # Convert each tool from our OpenAI format to MCP format
    for tool_def in tools:
        # Extract the function definition from the OpenAI tool format
        func_def = tool_def["function"]

        # Create an MCP Tool object with the same information
        mcp_tools.append(Tool(
            name=func_def["name"], # the function name
            description=func_def["description"], # what the tool does
            inputSchema=func_def["parameters"] # the JSON schema for parameters
        ))

    # Return the list of MCP Tool objects
    return mcp_tools

# Register the list_tools function with the server
# This tells the MCP server to use this function when clients ask for available tools
server.list_tools()(list_tools)

# %% [markdown] id="NM4w_xtuAvHM"
# Create an MCP wrapper for executing tools.

# %% id="ib0oaEEQBJp2" outputId="38fbf6af-0db0-42d9-90fe-2d074d1e9ec4"
from types import SimpleNamespace

async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle MCP tool calls by delegating to our existing tools."""
    # Convert MCP format to the format expected by execute()
    # execute() expects: tool_call.function.name and tool_call.function.arguments

    # Create the inner function object with name and arguments
    function = SimpleNamespace(
        name=name,
        arguments=json.dumps(arguments)  # Convert dict to JSON string
    )

    # Create the tool_call object with the function attribute
    tool_call = SimpleNamespace(function=function)

    # Execute the tool using our existing execute() function
    result = execute(tool_call)

    # Convert result to MCP response format
    result_text = str(result) if result is not None else "Success"
    return [TextContent(type="text", text=result_text)]

# Register the call_tool function with the server
server.call_tool()(call_tool)

# %% [markdown] id="5_UTk9kMBXdv"
# Expose an HTTP/SSE endpoint for the server.

# %% id="BJ57ifFcBbDx" outputId="a430872a-860f-411b-f7c3-9bf02c745ee2"
# FastAPI is a framework for building REST APIs
# %pip install --quiet fastapi
from mcp.server.sse import SseServerTransport
from fastapi import FastAPI, Request
from fastapi.responses import Response

# Create an SSE transport that will handle messages at the "/messages" path
sse = SseServerTransport("/messages")

# Create a FastAPI web application
app = FastAPI()


async def handle_sse(request: Request):
    """Handle incoming SSE connections from MCP clients."""
    # Connect the SSE transport to get read/write streams
    async with sse.connect_sse(
        request.scope, request.receive, request._send
    ) as (read_stream, write_stream):
        # Run the MCP server with these streams
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )
    return Response()

# Register the GET endpoint with the FastAPI app
# Clients connect to http://host:port/sse to establish SSE connection
app.add_api_route("/sse", handle_sse, methods=["GET"])

# Mount the POST handler for receiving messages
# Clients send messages to http://host:port/messages
app.mount("/messages", sse.handle_post_message)

print("✓ FastAPI app created")

# %% [markdown] id="v1Sp7zRqBjV7"
# Start the MCP server in the background.

# %%
# Uvicorn is a web server that handles HTTP requests and asynchronous code
# %pip install --quiet uvicorn
import threading
import uvicorn
import sys
import random

# The port number where the server will listen
server_port = random.randint(49152, 65535)

def run_server():
    """Run the uvicorn server. This will be called in a background thread."""
    try:
        # Start the server on all network interfaces (0.0.0.0) at the specified port
        uvicorn.run(app, host="0.0.0.0", port=server_port, log_level="warning")
    except Exception as e:
        # Print any errors to stderr
        print(f"✗ Server error: {e}", file=sys.stderr)

# Start server in background thread
server_thread = threading.Thread(
    target=run_server, # thread will automatically stop when main program exits
    daemon=True
  )
server_thread.start()

print(f"✓ Starting MCP HTTP server on port {server_port} in background...")
print(f"  Server available at http://127.0.0.1:{server_port}/sse")

# %% [markdown] id="YFpcTtJWB2-W"
# Verify that the server port is open and listening.

# %% id="VnEligIZB4qz" outputId="d9d4e412-3bd0-437e-aefc-5519f5c43870"
import time
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1)

# Try up to 5 times to verify the server started successfully
for attempt in range(1, 6):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    
    try:
        sock.connect(("127.0.0.1", server_port))
        print(f"✓ Port {server_port} is open (attempt {attempt}/5)")
        sock.close()
        break   # Exit the loop early since we confirmed the server is running
    except: 
        print(f"⏳ Attempt {attempt}/5: Port {server_port} not ready yet...")
        # Wait 1 second before trying again (unless this is the last attempt)
        if attempt < 5:
            time.sleep(1)

else:
    # This else block runs if we never broke out of the loop (all 5 attempts failed)
    print(f"✗ Port {server_port} is not open after 5 attempts")
    print("Make sure the server is running (previous cell)")

# %% [markdown] id="6l0hKhJyCB7s"
# Test the server with a dummy client.

# %% id="QQk64JgCvFrP" outputId="1ce97827-bf5b-4ff3-fa62-2975ebb941f5"
from mcp import ClientSession
from mcp.client.sse import sse_client

# Build the URL where our server is listening
server_url = f"http://127.0.0.1:{server_port}/sse"

async def test_client():
    """Test that the MCP server works by calling tools as a client."""
    # Connect to the server using SSE client
    async with sse_client(server_url) as (read, write):
        # Create a client session with the read/write streams
        async with ClientSession(read, write) as session:
            # Initialize the session (required handshake)
            await session.initialize()

            # List available tools from the server
            available_tools = await session.list_tools()
            print("Available tools:", [t.name for t in available_tools.tools])

            # Test the insertion_tool by adding 'cherries' to the set
            print("\nTesting insertion_tool with 'cherries':")
            result = await session.call_tool("insertion_tool", {"s": "cherries"})
            print("Result:", result.content[0].text)
            print("Current set:", MY_SET)

            # Test the removal_tool by removing 'cherries' from the set
            print("\nTesting removal_tool with 'cherries':")
            result = await session.call_tool("removal_tool", {"s": "cherries"})
            print("Result:", result.content[0].text)
            print("Current set:", MY_SET)


# %%
# I had to separate out the runing of the async functions at the top level because what works in jupyter
# doesn't work in straight python.  And vice versa.  I will remove the day before class

# %% tags=["active-ipynb"]
# # Run the async test function
# await test_client()

# %% raw_mimetype=""
# py version
import asyncio
if not IN_JUPYTER:
    asyncio.run(test_client())

# %% [markdown] id="hhA0L8dUNvYJ"
# # 🧠 Orchestration with LangGraph

# %% id="qrIbDfYJN3y0" outputId="2f685b79-a59b-4f6d-9253-5f1f871ae4a7"
# %pip uninstall -y -qqq langchain  
# %pip install --quiet "langchain-openai>=0.2,<1.0" "langchain_mcp_adapters" "langgraph"

# %% id="5x5y8R-nkMrq" outputId="e9218aef-d93e-4a0a-8420-51855447b3fc"
from langchain_mcp_adapters.client import MultiServerMCPClient

# Create MCP client that connects to your set-server
client = MultiServerMCPClient(
    {
        "set-server": {
            "transport": "sse",
            "url": f"http://localhost:{server_port}/sse",
        }
    }
)

# %% tags=["active-ipynb"]
# # Get available tools from the MCP server
# tools_from_mcp = await client.get_tools()
# print(f"✓ Loaded {len(tools_from_mcp)} tools from MCP server")
# for tool in tools_from_mcp:
#     print(f"  - {tool.name}: {tool.description}")

# %%
if not IN_JUPYTER:
    tools_from_mcp = asyncio.run(client.get_tools())

# %% id="t_p_I0JokOjL" outputId="fb80bb10-ac50-4e5c-c93f-f744e81fae33"
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# Create a LangGraph agent using the tools we already loaded
agent_executor = create_react_agent(
    ChatOpenAI(model="gpt-4o", temperature=0),
    tools_from_mcp,
)

print("✓ LangGraph agent created")

# %% tags=["active-ipynb"]
# result = await agent_executor.ainvoke({
#     "messages": [{"role": "user", "content": "Please add 'grapes', 'kiwi', and 'mango' to the set."}]
# })

# %%
if not IN_JUPYTER:
    result = asyncio.run(
        agent_executor.ainvoke({
            "messages": [{
                "role": "user", 
                "content": "Please add 'grapes', 'kiwi', and 'mango' to the set."}]
        })
    )

# %% id="7zu7HEPAkQlc" outputId="637a962e-c096-4f8e-8671-0cf1b49f936d"
# Display the conversation
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

for message in result['messages']:
    if isinstance(message, HumanMessage):
        print("Human: \033[32m" + message.content + "\033[0m")
    elif isinstance(message, AIMessage):
        if message.content:
            print("AI: \033[34m" + message.content + "\033[0m")
    elif isinstance(message, ToolMessage):
        if "Error" not in message.content:
            print(f"Tool Result: \033[32mSuccess\033[0m")

print(f"\n⭐ The resulting set is: {MY_SET}")

# %% [markdown] id="K0Cw_2GpvqKp"
# # 🧹 Cleanup
#
# Stop the MCP server.

# %% id="7kTywUxOwKua"
# Kill any process running uvicorn on our server port
# !pkill -f "uvicorn.*{server_port}"
print("✓ Server stopped")

# %% [markdown] id="dvwl7EKJTZSL"
# # Thank you!

# %% [markdown] id="H4N9Oqg7No6i"
# ###
