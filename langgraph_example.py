# ============================================================================
# ⚠️  DEPRECATED: Use workflow_manager.py instead!
# ============================================================================
# This file contains the OLD regex-based query analysis approach.
# The NEW LLM-powered system is in workflow_manager.py with these benefits:
# 
# ✅ Works for ANY domain (not just investments)
# ✅ Uses DeepSeek reasoner for intelligent analysis  
# ✅ Generates personalized follow-up questions
# ✅ ChatGPT-style proactive query understanding
#
# To use the advanced system:
#   from workflow_manager import create_investment_advisor
#   agent = create_investment_advisor()
#   agent.run_query("your question here")
# ============================================================================

from dotenv import load_dotenv
import os
import re
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

load_dotenv()

# Get the API keys from the environment
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

@tool
def tavily_web_search(query: str):
    """
    Searches the web using Tavily and returns the results.
    """
    from tavily import TavilyClient
    client = TavilyClient(api_key=TAVILY_API_KEY)
    return client.search(query, search_depth="advanced")

tool_node = ToolNode([tavily_web_search])

model = ChatDeepSeek(model="deepseek-reasoner", api_key=DEEPSEEK_API_KEY, temperature=0)
model = model.bind_tools([tavily_web_search])

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def analyze_user_query(state):
    """
    Routing function that analyzes the initial user query to determine if it needs clarification.
    This is proactive analysis BEFORE the AI responds.
    """
    messages = state['messages']
    if not messages:
        return "proceed_to_agent"
    
    # Get the most recent user message
    user_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg
            break
    
    if not user_message:
        return "proceed_to_agent"
    
    query = user_message.content.lower()
    
    # Detect vague investment queries that need clarification
    investment_patterns = [
        r'\b(recommend|suggest|advise)\b.*\b(stock|stocks|investment|invest|portfolio)\b',
        r'\b(what|which)\b.*\b(should|to|can)\b.*\b(buy|invest|purchase)\b',
        r'\b(what|which)\b.*\b(stock|stocks|investment)\b.*\b(should|to)\b.*\b(buy|invest)\b',
        r'\b(best|good|top)\b.*\b(stock|stocks|investment|portfolio)\b',
        r'\b(invest|investment)\b.*\b(option|advice|recommendation)\b',
        r'\b(where|how)\b.*\b(invest|put)\b.*\b(money|fund)\b',
        r'\b(build|create)\b.*\b(portfolio|investment)\b'
    ]
    
    for pattern in investment_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            # Check if query is vague (lacks specific details)
            specific_indicators = [
                r'\$\d+',  # Dollar amounts
                r'\b(conservative|aggressive|moderate)\b.*\brisk\b',  # Risk tolerance
                r'\b(short|long)[\-\s]?term\b',  # Time horizon
                r'\b(tech|technology|healthcare|energy|finance)\b',  # Sectors
                r'\b(us|usa|international|emerging)\b.*\bmarket\b'  # Geography
            ]
            
            # If it's an investment query but lacks specific details, needs clarification
            has_specifics = any(re.search(indicator, query, re.IGNORECASE) 
                              for indicator in specific_indicators)
            
            if not has_specifics:
                return "need_investment_clarification"
    
    return "proceed_to_agent"

def investment_clarification(state):
    """
    Asks specific follow-up questions for investment advice,
    similar to ChatGPT's deep research feature.
    """
    clarification_msg = AIMessage(
        content="""I'd be happy to help you with investment recommendations! To provide you with the most suitable advice, I need to understand your specific situation better. Could you please provide details about:

**📊 Investment Profile:**
• **Risk Tolerance**: Are you comfortable with conservative (low risk), moderate (balanced), or aggressive (high risk) investments?
• **Time Horizon**: Are you investing for the short-term (1-2 years), medium-term (3-7 years), or long-term (8+ years)?
• **Investment Amount**: What's your approximate budget or investment range?

**🌍 Preferences:**
• **Geographic Focus**: Are you interested in US stocks, international markets, or emerging markets?
• **Sector Preferences**: Any specific industries you're interested in (tech, healthcare, energy, etc.) or want to avoid?
• **Current Portfolio**: Do you already have investments, or is this a fresh start?

**🎯 Goals:**
• **Primary Objective**: Are you looking for growth, income (dividends), or a balanced approach?
• **Special Considerations**: Any ESG (environmental/social) preferences or other criteria?

The more details you provide, the better I can tailor my recommendations to your specific needs and goals!"""
    )
    
    return {"messages": [clarification_msg]}

def should_continue(state):
    """
    Determines next step in the workflow based on AI's response.
    Returns:
    - 'execute_tools' if AI wants to use tools
    - 'need_clarification' if AI needs more info from user
    - 'escalate_to_human' if issue requires human intervention
    - 'provide_final_answer' if ready to respond
    """
    try:
        messages = state['messages']
        
        # Safety check - ensure we have messages
        if not messages:
            return "provide_final_answer"
        
        last_message = messages[-1]
        
        # Check if AI wants to use tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "execute_tools"
        
        # Check message content for special cases using improved regex patterns
        content = last_message.content if hasattr(last_message, 'content') else ""
        
        # Improved clarification detection using regex patterns
        clarification_patterns = [
            r'\b(need|require|would\s+need)\b.*\b(more|additional|further)\b.*\b(information|details|context|clarification)\b',
            r'\b(could|can|would)\s+you\b.*\b(clarify|specify|explain|elaborate|provide)\b',
            r'\b(what|which)\b.*\b(exactly|specifically)\b.*\b(do\s+you|are\s+you)\b',
            r'\b(help|assist)\s+you\s+better.*\b(need|require)\b.*\b(understand|know|details)\b',
            r'\b(unclear|unsure|confused)\b.*\b(about|regarding|concerning)\b',
            r'\b(tell|let)\s+me\s+(know\s+)?more\b.*\b(about|regarding)\b',
            r'\b(provide|share|give)\s+(me\s+)?(more|additional)\b.*\b(details|information|context)\b'
        ]
        
        for pattern in clarification_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return "need_clarification"
        
        # Improved escalation detection using regex patterns
        escalation_patterns = [
            r'\b(transfer|connect|forward|escalate)\b.*\b(human|agent|specialist|representative|support)\b',
            r'\b(beyond|outside|exceed)\b.*\b(capabilities|expertise|knowledge|scope)\b',
            r'\b(require|need)\b.*\b(human|expert|specialist|professional)\b.*\b(assistance|help|support)\b',
            r'\b(not\s+(qualified|equipped|able)|unable\s+to\s+handle|cannot\s+(help|assist))\b',
            r'\b(legal|medical|financial|technical)\b.*\b(advice|consultation|expert|professional)\b',
            r'\b(contact|speak\s+with|consult|see)\b.*\b(professional|expert|specialist|doctor|lawyer)\b',
            r'\b(this\s+requires|you\s+(should|need\s+to))\b.*\b(professional|expert|specialist)\b'
        ]
        
        for pattern in escalation_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return "escalate_to_human"
        
        # Check for repeated failures (too many messages without resolution)
        if len(messages) > 15:  # Long conversation, might need human help
            recent_ai_messages = [msg for msg in messages[-6:] 
                                if hasattr(msg, 'content') and 
                                not hasattr(msg, 'tool_calls')]
            if len(recent_ai_messages) > 3:  # Multiple attempts without tools
                return "escalate_to_human"
        
        return "provide_final_answer"
            
    except (IndexError, AttributeError, KeyError) as e:
        # Safe fallback - if anything goes wrong, end gracefully
        print(f"Warning in should_continue: {e}")
        return "provide_final_answer"

def call_model(state):
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

def call_tool(state):
    messages = state['messages']
    last_message = messages[-1]
    # Use the tool_node directly with the state
    return tool_node.invoke(state)

def ask_for_clarification(state):
    """
    Handles cases where AI needs more information from the user.
    Prompts user for specific details needed to proceed.
    """
    messages = state['messages']
    
    # Create a clarification request message
    clarification_msg = AIMessage(
        content="I need some additional information to help you better. "
                "Could you please provide more specific details about what you're looking for? "
                "The more details you can share, the better I can assist you."
    )
    
    return {"messages": [clarification_msg]}

def human_handoff(state):
    """
    Handles cases requiring human intervention.
    This could be integrated with customer service systems, ticketing systems, etc.
    """
    messages = state['messages']
    
    # In a real system, this would:
    # 1. Create a support ticket
    # 2. Notify human agents
    # 3. Transfer conversation context
    # 4. Set up human-to-user communication channel
    
    handoff_msg = AIMessage(
        content="I understand this requires human assistance. "
                "I'm connecting you with a human representative who can better help you with this matter. "
                "Please hold on while I transfer your conversation. "
                "\n\n[In a production system, this would trigger human agent notification "
                "and transfer the conversation context to customer service systems.]"
    )
    
    return {"messages": [handoff_msg]}

def query_router(state):
    """
    Simple passthrough node that just returns the state unchanged.
    The actual routing logic is in the conditional edges.
    """
    return {"messages": []}  # No new messages, just pass through

# Create the workflow with proactive query analysis
workflow = StateGraph(AgentState)

# Add all nodes
workflow.add_node("query_router", query_router)
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)
workflow.add_node("clarification", ask_for_clarification)
workflow.add_node("investment_clarification", investment_clarification)
workflow.add_node("human_handoff", human_handoff)

# Set entry point to query router (proactive analysis)
workflow.set_entry_point("query_router")

# Add conditional edges from query router
workflow.add_conditional_edges(
    "query_router",
    analyze_user_query,
    {
        "proceed_to_agent": "agent",
        "need_investment_clarification": "investment_clarification",
    },
)

# Existing conditional edges from agent
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "execute_tools": "action",
        "need_clarification": "clarification",
        "escalate_to_human": "human_handoff",
        "provide_final_answer": END,
    },
)

# Regular edges
workflow.add_edge("action", "agent")
workflow.add_edge("clarification", END)
workflow.add_edge("investment_clarification", END)  # Wait for user's detailed response
workflow.add_edge("human_handoff", END)

app = workflow.compile()

if __name__ == "__main__":
    # query = "Could you please recommend me some good investment options in stocks?"
    query = "How to learn ai agent creation?"
    inputs = {"messages": [HumanMessage(content=query)]}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(value)
        print("\n---\n")

