"""
Test ReAct Asset Agent with JSON Patches

Simple LangGraph ReAct agent that can:
1. Create new assets
2. Update existing assets using JSON patches
3. List current assets

Example usage:
    python test_asset_agent.py
"""
import json
import jsonpatch
from typing import Annotated, TypedDict, Literal
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# =============================================================================
# Pydantic Models
# =============================================================================

class Asset(BaseModel):
    """Asset model matching API schema"""
    id: str | None = None
    name: str
    description: str | None = None
    isPrimary: bool = False
    confidentiality: Literal["PUBLIC", "INTERNAL", "PROTECTED", "STRICTLY_PROTECTED"] | None = None
    integrity: Literal["LOW", "MEDIUM", "HIGH"] | None = None
    availability: Literal["LOW", "MEDIUM", "HIGH"] | None = None
    assetType: str


class AssetPatchRequest(BaseModel):
    """Request to update asset using JSON patches"""
    asset_name: str = Field(description="Name of the asset to update")
    operations: list[dict] = Field(
        description="""JSON Patch operations (RFC 6902). Each operation is a dict with:
        - op: "replace", "add", or "remove"
        - path: JSON pointer (e.g., "/name", "/description")
        - value: new value (not needed for "remove")

        Examples:
        [{"op": "replace", "path": "/name", "value": "New Name"}]
        [{"op": "add", "path": "/description", "value": "New description"}]
        [{"op": "remove", "path": "/description"}]
        """
    )


# =============================================================================
# Agent State
# =============================================================================

class AgentState(TypedDict):
    """Agent state with messages and assets"""
    messages: Annotated[list, add_messages]
    assets: list[dict]  # Current assets state


# =============================================================================
# Tools
# =============================================================================

def create_asset(
    name: str,
    asset_type: str,
    description: str | None = None,
    is_primary: bool = False,
) -> str:
    """
    Create a new asset.

    Args:
        name: Asset name
        asset_type: Type of asset (e.g., "Database", "Server", "Application")
        description: Optional description
        is_primary: Whether this is a primary asset (requires CIA ratings)

    Returns:
        Success message with created asset details
    """
    # Get current state (in real agent this comes from state)
    # For now we'll store in global variable for simplicity
    global ASSETS_STATE

    # Generate ID
    asset_id = str(len(ASSETS_STATE) + 1)

    asset = {
        "id": asset_id,
        "name": name,
        "description": description,
        "isPrimary": is_primary,
        "confidentiality": None,
        "integrity": None,
        "availability": None,
        "assetType": asset_type,
    }

    ASSETS_STATE.append(asset)

    return f"✓ Created asset '{name}' (ID: {asset_id}, Type: {asset_type})"


def update_asset(asset_name: str, operations: str) -> str:
    """
    Update an existing asset using JSON Patch operations.

    Args:
        asset_name: Name of the asset to update
        operations: JSON string with patch operations in RFC 6902 format (e.g., '[{"op": "replace", "path": "/name", "value": "New Name"}]')

    Returns:
        Success message with update details
    """
    global ASSETS_STATE

    # Find asset by name
    asset = next((a for a in ASSETS_STATE if a["name"] == asset_name), None)
    if not asset:
        return f"❌ Asset '{asset_name}' not found. Available: {[a['name'] for a in ASSETS_STATE]}"

    # Parse operations
    try:
        patch_ops = json.loads(operations)
    except json.JSONDecodeError as e:
        return f"❌ Invalid JSON in operations: {e}"

    # Apply patch
    try:
        updated_asset = jsonpatch.apply_patch(asset, patch_ops)

        # Update in state
        for i, a in enumerate(ASSETS_STATE):
            if a["name"] == asset_name:
                ASSETS_STATE[i] = updated_asset
                break

        # Build change summary
        changes = []
        for op in patch_ops:
            field = op["path"].lstrip("/")
            if op["op"] == "replace":
                changes.append(f"{field}: '{op['value']}'")
            elif op["op"] == "add":
                changes.append(f"added {field}: '{op['value']}'")
            elif op["op"] == "remove":
                changes.append(f"removed {field}")

        return f"✓ Updated asset '{asset_name}': {', '.join(changes)}"

    except jsonpatch.JsonPatchException as e:
        return f"❌ Patch failed: {e}"


def list_assets() -> str:
    """
    List all current assets.

    Returns:
        Formatted list of assets with details
    """
    global ASSETS_STATE

    if not ASSETS_STATE:
        return "No assets yet. Use create_asset to add one."

    result = f"Current assets ({len(ASSETS_STATE)}):\n"
    for asset in ASSETS_STATE:
        result += f"\n• {asset['name']} (ID: {asset['id']})\n"
        result += f"  Type: {asset['assetType']}\n"
        if asset['description']:
            result += f"  Description: {asset['description']}\n"
        result += f"  Primary: {asset['isPrimary']}\n"

    return result


# =============================================================================
# Agent Logic
# =============================================================================

# Global state for simplicity (in real app this would be in StateGraph state)
ASSETS_STATE = []

# Create tools list
tools = [create_asset, update_asset, list_assets]

# Create LLM with tools
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Decide whether to continue or end"""
    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, route to tools node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Otherwise, end
    return "end"


def call_model(state: AgentState) -> dict:
    """Call the LLM"""
    messages = state["messages"]

    # Add system message with current state
    system_msg = f"""You are an asset management assistant.

Current assets state:
{json.dumps(state.get('assets', []), indent=2)}

You can:
1. create_asset - Create new assets
2. update_asset - Update existing assets using JSON Patch operations
3. list_assets - Show all current assets

When updating assets, use JSON Patch format (RFC 6902):
- {{"op": "replace", "path": "/field", "value": "new value"}} - Change existing field
- {{"op": "add", "path": "/field", "value": "value"}} - Add new field
- {{"op": "remove", "path": "/field"}} - Remove field

Always use list_assets first to see current state before updating.
"""

    response = llm_with_tools.invoke([{"role": "system", "content": system_msg}] + messages)

    return {"messages": [response]}


# =============================================================================
# Build Graph
# =============================================================================

def create_asset_agent():
    """Create the ReAct asset agent graph"""

    # Create graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))

    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        }
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# =============================================================================
# Example Usage
# =============================================================================

def run_example_conversation():
    """Run example conversation with agent"""

    print("=" * 80)
    print("Asset Agent Test - ReAct with JSON Patches (Example Mode)")
    print("=" * 80)

    # Create agent
    agent = create_asset_agent()

    # Example conversation
    conversations = [
        "Create a new asset called 'Customer Database' of type 'Database' with description 'Main customer data storage'",
        "Create another asset called 'Web Server' of type 'Server'",
        "List all assets",
        "Update the Customer Database - change its name to 'Client Database'",
        "Update the Web Server - add description 'Production web server' and set isPrimary to true",
        "List all assets to see the changes",
    ]

    state = {"messages": [], "assets": []}

    for user_msg in conversations:
        print(f"\n{'─' * 80}")
        print(f"👤 User: {user_msg}")
        print(f"{'─' * 80}")

        # Run agent
        state["messages"].append(HumanMessage(content=user_msg))
        state["assets"] = ASSETS_STATE  # Sync global state

        result = agent.invoke(state)

        # Print agent response
        for msg in result["messages"][len(state["messages"]):]:
            if isinstance(msg, AIMessage):
                if msg.content:
                    print(f"🤖 Agent: {msg.content}")
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"   🔧 Calling: {tc['name']}({tc['args']})")
            elif isinstance(msg, ToolMessage):
                print(f"   ✓ Result: {msg.content}")

        state = result

    print("\n" + "=" * 80)
    print("Final Assets State:")
    print("=" * 80)
    print(json.dumps(ASSETS_STATE, indent=2))


def run_interactive():
    """Run interactive chat with agent"""
    global ASSETS_STATE

    print("=" * 80)
    print("Asset Agent - Interactive Mode")
    print("=" * 80)
    print("Commands:")
    print("  - Type your message to chat with the agent")
    print("  - Type 'state' to see current assets")
    print("  - Type 'clear' to reset all assets")
    print("  - Type 'exit' to quit")
    print("=" * 80)

    # Create agent
    agent = create_asset_agent()
    state = {"messages": [], "assets": []}

    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() == "exit":
                print("\n👋 Goodbye!")
                break

            elif user_input.lower() == "state":
                print("\n" + "=" * 80)
                print("Current Assets State:")
                print("=" * 80)
                if ASSETS_STATE:
                    print(json.dumps(ASSETS_STATE, indent=2))
                else:
                    print("No assets yet.")
                continue

            elif user_input.lower() == "clear":
                ASSETS_STATE = []
                state = {"messages": [], "assets": []}
                print("\n✓ Cleared all assets and conversation history")
                continue

            # Run agent
            state["messages"].append(HumanMessage(content=user_input))
            state["assets"] = ASSETS_STATE  # Sync global state

            result = agent.invoke(state)

            # Print agent response
            print()
            for msg in result["messages"][len(state["messages"]):]:
                if isinstance(msg, AIMessage):
                    if msg.content:
                        print(f"🤖 Agent: {msg.content}")
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"   🔧 Calling: {tc['name']}({tc['args']})")
                elif isinstance(msg, ToolMessage):
                    print(f"   ✓ Result: {msg.content}")

            state = result

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def main():
    """Main entry point - choose mode"""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        run_interactive()
    else:
        run_example_conversation()


if __name__ == "__main__":
    main()
