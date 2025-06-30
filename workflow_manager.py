from dotenv import load_dotenv
import os
from typing import TypedDict, Annotated, Sequence, Optional, Dict, Any
import operator
import re
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ============================================================================
# TOOLS
# ============================================================================

@tool
def tavily_web_search(query: str):
    """
    Searches the web using Tavily and returns the results.
    """
    from tavily import TavilyClient
    client = TavilyClient(api_key=TAVILY_API_KEY)
    return client.search(query, search_depth="advanced")

# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# ============================================================================
# WORKFLOW MANAGER CLASS
# ============================================================================

class AIAgentWorkflow:
    """
    Production-ready LangGraph workflow manager with proper encapsulation,
    configuration options, and reusability.
    """
    
    def __init__(self, 
                 model_name: str = "deepseek-reasoner",
                 api_key: Optional[str] = None,
                 temperature: float = 0.0,
                 tools: Optional[list] = None):
        """
        Initialize the AI Agent Workflow.
        
        Args:
            model_name: Name of the model to use
            api_key: API key for the model (uses env var if None)
            temperature: Model temperature setting
            tools: List of tools to include (uses default if None)
        """
        self.model_name = model_name
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.temperature = temperature
        self.tools = tools or [tavily_web_search]
        
        # Initialize components
        self.model = self._create_model()
        self.tool_node = ToolNode(self.tools)
        self.app = None  # Will be compiled when needed
    
    def _create_model(self):
        """Create and configure the language model."""
        model = ChatDeepSeek(
            model=self.model_name, 
            api_key=self.api_key, 
            temperature=self.temperature
        )
        return model.bind_tools(self.tools)
    
    # ========================================================================
    # NODE FUNCTIONS
    # ========================================================================
    
    def analyze_user_query(self, state):
        """
        Proactive query analysis to route specialized queries to clarification.
        Detects investment queries and checks if they need more specificity.
        """
        messages = state.get('messages', [])
        if not messages:
            return "proceed_to_agent"
        
        user_message = messages[0].content if messages else ""
        
        # Investment query detection patterns
        investment_patterns = [
            r'\b(recommend|suggest|advice|tips)\b.*\b(stock|invest|investment|portfolio|trading)\b',
            r'\b(good|best|top)\b.*\b(stock|investment|shares|equity)\b',
            r'\b(what|which|where)\b.*\b(invest|investment|stock|portfolio)\b',
            r'\b(buy|purchase|acquire)\b.*\b(stock|shares|investment)\b',
        ]
        
        is_investment_query = any(re.search(pattern, user_message, re.IGNORECASE) 
                                for pattern in investment_patterns)
        
        if is_investment_query:
            # Check if query has sufficient detail
            specificity_indicators = [
                r'\b(risk tolerance|conservative|aggressive|moderate)\b',
                r'\b(\$\d+|amount|budget|range)\b',
                r'\b(short.term|long.term|time horizon|years?)\b',
                r'\b(sector|industry|tech|healthcare|finance)\b',
                r'\b(dividend|growth|income|value)\b',
                r'\b(US|international|emerging|market)\b'
            ]
            
            has_specifics = any(re.search(indicator, user_message, re.IGNORECASE) 
                              for indicator in specificity_indicators)
            
            if not has_specifics:
                return "need_investment_clarification"
        
        return "proceed_to_agent"
    
    def investment_clarification(self, state):
        """Specialized clarification for investment queries."""
        clarification_msg = AIMessage(
            content="""I'd be happy to help you with investment recommendations! To provide the most suitable suggestions for your situation, I need to understand your specific needs better.

**📊 Investment Profile:**
• **Risk Tolerance**: Are you comfortable with high-risk/high-reward investments, or do you prefer conservative, stable options?
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
    
    def should_continue(self, state):
        """Intelligent routing based on AI response analysis."""
        try:
            messages = state['messages']
            if not messages:
                return "provide_final_answer"
            
            last_message = messages[-1]
            
            # Check if AI wants to use tools
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "execute_tools"
            
            content = last_message.content if hasattr(last_message, 'content') else ""
            
            # Clarification detection
            clarification_patterns = [
                r'\b(need|require|would\s+need)\b.*\b(more|additional|further)\b.*\b(information|details|context|clarification)\b',
                r'\b(could|can|would)\s+you\b.*\b(clarify|specify|explain|elaborate|provide)\b',
                r'\b(what|which)\b.*\b(exactly|specifically)\b.*\b(do\s+you|are\s+you)\b',
                r'\b(help|assist)\s+you\s+better.*\b(need|require)\b.*\b(understand|know|details)\b',
            ]
            
            for pattern in clarification_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return "need_clarification"
            
            # Escalation detection
            escalation_patterns = [
                r'\b(transfer|connect|forward|escalate)\b.*\b(human|agent|specialist|representative|support)\b',
                r'\b(beyond|outside|exceed)\b.*\b(capabilities|expertise|knowledge|scope)\b',
                r'\b(require|need)\b.*\b(human|expert|specialist|professional)\b.*\b(assistance|help|support)\b',
            ]
            
            for pattern in escalation_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return "escalate_to_human"
            
            return "provide_final_answer"
            
        except Exception as e:
            print(f"Warning in should_continue: {e}")
            return "provide_final_answer"
    
    def call_model(self, state):
        """Main AI model invocation."""
        messages = state['messages']
        response = self.model.invoke(messages)
        return {"messages": [response]}
    
    def call_tool(self, state):
        """Tool execution node."""
        return self.tool_node.invoke(state)
    
    def ask_for_clarification(self, state):
        """General clarification handler."""
        clarification_msg = AIMessage(
            content="I need some additional information to help you better. "
                    "Could you please provide more specific details about what you're looking for? "
                    "The more details you can share, the better I can assist you."
        )
        return {"messages": [clarification_msg]}
    
    def human_handoff(self, state):
        """Human escalation handler."""
        handoff_msg = AIMessage(
            content="I understand this requires human assistance. "
                    "I'm connecting you with a human representative who can better help you with this matter. "
                    "Please hold on while I transfer your conversation."
        )
        return {"messages": [handoff_msg]}
    
    def query_router(self, state):
        """Entry point router (passthrough for conditional logic)."""
        return {"messages": []}
    
    # ========================================================================
    # WORKFLOW CREATION AND MANAGEMENT
    # ========================================================================
    
    def create_workflow(self) -> StateGraph:
        """
        Factory method to create and configure the workflow graph.
        
        Returns:
            StateGraph: Configured but not yet compiled workflow
        """
        workflow = StateGraph(AgentState)
        
        # Add all nodes
        workflow.add_node("query_router", self.query_router)
        workflow.add_node("agent", self.call_model)
        workflow.add_node("action", self.call_tool)
        workflow.add_node("clarification", self.ask_for_clarification)
        workflow.add_node("investment_clarification", self.investment_clarification)
        workflow.add_node("human_handoff", self.human_handoff)
        
        # Set entry point
        workflow.set_entry_point("query_router")
        
        # Configure edges
        self._add_workflow_edges(workflow)
        
        return workflow
    
    def _add_workflow_edges(self, workflow: StateGraph):
        """Configure all workflow edges."""
        # Conditional edges from query router
        workflow.add_conditional_edges(
            "query_router",
            self.analyze_user_query,
            {
                "proceed_to_agent": "agent",
                "need_investment_clarification": "investment_clarification",
            },
        )
        
        # Conditional edges from agent
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
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
        workflow.add_edge("investment_clarification", END)
        workflow.add_edge("human_handoff", END)
    
    def compile_workflow(self):
        """Compile the workflow for execution."""
        if self.app is None:
            workflow = self.create_workflow()
            self.app = workflow.compile()
        return self.app
    
    # ========================================================================
    # EXECUTION METHODS
    # ========================================================================
    
    def run_query(self, query: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Execute a single query through the workflow.
        
        Args:
            query: User query string
            verbose: Whether to print intermediate outputs
            
        Returns:
            Dict containing final results and metadata
        """
        app = self.compile_workflow()
        inputs = {"messages": [HumanMessage(content=query)]}
        
        results = []
        for output in app.stream(inputs):
            if verbose:
                for key, value in output.items():
                    print(f"Output from node '{key}':")
                    print("---")
                    print(value)
                    print("\n" + "="*50 + "\n")
            results.append(output)
        
        return {
            "query": query,
            "results": results,
            "final_state": results[-1] if results else None
        }
    
    def run_conversation(self, queries: list, verbose: bool = True) -> list:
        """
        Run multiple queries in sequence (useful for testing).
        
        Args:
            queries: List of query strings
            verbose: Whether to print outputs
            
        Returns:
            List of results for each query
        """
        conversation_results = []
        
        for i, query in enumerate(queries):
            print(f"\n🔍 QUERY {i+1}: {query}")
            print("="*60)
            
            result = self.run_query(query, verbose)
            conversation_results.append(result)
            
            print("\n" + "🏁 COMPLETED" + "\n")
        
        return conversation_results
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """Get information about the workflow configuration."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "tools": [tool.name for tool in self.tools],
            "node_count": len(self.create_workflow().nodes) if self.app is None else len(self.app.get_graph().nodes),
            "compiled": self.app is not None
        }
    
    def reset_workflow(self):
        """Reset the compiled workflow (useful for configuration changes)."""
        self.app = None

# ============================================================================
# FACTORY FUNCTIONS FOR DIFFERENT USE CASES
# ============================================================================

def create_basic_agent(temperature: float = 0.0) -> AIAgentWorkflow:
    """Create a basic AI agent with default settings."""
    return AIAgentWorkflow(temperature=temperature)

def create_research_agent(temperature: float = 0.1) -> AIAgentWorkflow:
    """Create an AI agent optimized for research tasks."""
    return AIAgentWorkflow(
        temperature=temperature,
        tools=[tavily_web_search]  # Could add more research tools
    )

def create_investment_advisor(temperature: float = 0.0) -> AIAgentWorkflow:
    """Create an AI agent specialized for investment advice."""
    return AIAgentWorkflow(
        temperature=temperature,
        tools=[tavily_web_search]  # Could add financial data tools
    )

# ============================================================================
# DEMO AND TESTING
# ============================================================================

def demo_workflow():
    """Demonstrate the workflow with various query types."""
    agent = create_research_agent()
    
    print("🚀 AI AGENT WORKFLOW DEMO")
    print("="*50)
    print(f"Configuration: {agent.get_workflow_info()}")
    print("\n")
    
    # Test queries
    test_queries = [
        "How to learn AI agent creation?",
        "What are some good conservative investment options?",
        "Could you recommend some good stocks?",  # Should trigger investment clarification
    ]
    
    results = agent.run_conversation(test_queries, verbose=True)
    return results

if __name__ == "__main__":
    # Example usage patterns
    print("Choose demo mode:")
    print("1. Single query")
    print("2. Multiple queries")
    print("3. Custom agent configuration")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        agent = create_basic_agent()
        query = input("Enter your query: ")
        agent.run_query(query)
        
    elif choice == "2":
        demo_workflow()
        
    elif choice == "3":
        print("Creating custom agent...")
        agent = AIAgentWorkflow(
            temperature=0.2,
            model_name="deepseek-reasoner"
        )
        print(f"Custom agent info: {agent.get_workflow_info()}")
        
        query = input("Enter your query: ")
        agent.run_query(query)
        
    else:
        print("Running default demo...")
        demo_workflow() 