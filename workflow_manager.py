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
        🧠 INTELLIGENT QUERY ANALYSIS using LLM (ChatGPT style)
        
        Uses DeepSeek reasoner to determine if ANY query needs clarification.
        Works for all topics, not just investments.
        """
        messages = state.get('messages', [])
        if not messages:
            return "proceed_to_agent"
        
        # Get the user's query
        user_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break
        
        if not user_message:
            return "proceed_to_agent"
        
        # Create analyzer model (separate from main model for analysis)
        analyzer_model = ChatDeepSeek(
            model="deepseek-reasoner", 
            api_key=self.api_key, 
            temperature=0.1
        )
        
        # LLM Analysis Prompt (this is the key to ChatGPT-like behavior!)
        analysis_prompt = f"""You are a query analysis expert. Your job is to determine if a user's question needs clarification to provide the best possible answer.

USER QUERY: "{user_message}"

Analyze this query and determine:
1. Is this query specific enough to provide a comprehensive answer?
2. What important details are missing that would help give a better response?
3. Would asking follow-up questions significantly improve the answer quality?

Guidelines for when to ask for clarification:
- Query is too broad/vague (e.g., "help me with investing" vs "recommend conservative tech stocks for retirement")
- Missing critical context for personalized advice (investment, career, health, legal advice)
- Question could benefit from understanding user's specific situation
- Multiple approaches possible without knowing user preferences
- Personal recommendations that depend on individual circumstances

Guidelines for when NOT to ask for clarification:
- Factual questions with clear answers (e.g., "What is the capital of France?")
- Specific technical questions (e.g., "How to install Python on macOS?")
- General information requests that don't require personalization

Respond with a JSON object:
{{
    "needs_clarification": true/false,
    "reason": "brief explanation of why clarification is needed",
    "follow_up_questions": ["question1", "question2", "question3"]
}}

Be selective - only ask for clarification when it would genuinely improve the answer quality."""

        try:
            # Ask DeepSeek to analyze the query
            analysis_response = analyzer_model.invoke([
                HumanMessage(content=analysis_prompt)
            ])
            
            # Parse the LLM's analysis
            analysis_text = analysis_response.content
            
            # Extract JSON from the response
            start_idx = analysis_text.find('{')
            end_idx = analysis_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = analysis_text[start_idx:end_idx]
                import json
                analysis = json.loads(json_str)
                
                # Store analysis in instance variable for smart_clarification to use
                if analysis.get('needs_clarification', False):
                    self._current_analysis = analysis
                    return "need_smart_clarification"
            
        except Exception as e:
            print(f"Query analysis error: {e}")
            # Fallback to proceeding normally
        
        return "proceed_to_agent"
    
    def smart_clarification(self, state):
        """
        🎯 DYNAMIC CLARIFICATION using LLM analysis
        Generates personalized follow-up questions based on the query type.
        """
        # Get analysis from instance variable (stored by analyze_user_query)
        analysis = getattr(self, '_current_analysis', {})
        follow_up_questions = analysis.get('follow_up_questions', [])
        reason = analysis.get('reason', 'I need more information to help you better.')
        
        # Format the follow-up questions nicely
        questions_text = ""
        if follow_up_questions:
            questions_text = "\n\n**To give you the best recommendations, could you please share:**\n\n"
            for i, question in enumerate(follow_up_questions, 1):
                questions_text += f"{i}. {question}\n"
        
        clarification_msg = AIMessage(
            content=f"""{reason}{questions_text}

The more specific details you provide, the more tailored and useful my recommendations will be! 🎯"""
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
        workflow.add_node("smart_clarification", self.smart_clarification)
        workflow.add_node("human_handoff", self.human_handoff)
        
        # Set entry point
        workflow.set_entry_point("query_router")
        
        # Configure edges
        self._add_workflow_edges(workflow)
        
        return workflow
    
    def _add_workflow_edges(self, workflow: StateGraph):
        """Configure all workflow edges."""
        # Conditional edges from query router (LLM-powered!)
        workflow.add_conditional_edges(
            "query_router",
            self.analyze_user_query,
            {
                "proceed_to_agent": "agent",
                "need_smart_clarification": "smart_clarification",
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
        workflow.add_edge("smart_clarification", END)  # Wait for user's detailed response
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
# CONVERSATION SESSION MANAGER
# ============================================================================

class ConversationSession:
    """
    🗣️ MULTI-TURN CONVERSATION MANAGER
    
    Handles conversation continuity across clarification cycles.
    Maintains context and allows users to respond to follow-up questions.
    """
    
    def __init__(self, agent: AIAgentWorkflow):
        """
        Initialize conversation session.
        
        Args:
            agent: The AIAgentWorkflow instance to use
        """
        self.agent = agent
        self.conversation_history = []
        self.waiting_for_clarification = False
        self.last_clarification_type = None
        
    def start_conversation(self, initial_query: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Start a new conversation with an initial query.
        
        Args:
            initial_query: The user's initial question
            verbose: Whether to print outputs
            
        Returns:
            Dict containing response and conversation state
        """
        print(f"🚀 Starting conversation: '{initial_query}'")
        
        # Clear any previous conversation
        self.conversation_history = []
        self.waiting_for_clarification = False
        
        # Add initial query to history
        initial_message = HumanMessage(content=initial_query)
        self.conversation_history.append(initial_message)
        
        # Run the workflow
        result = self._run_workflow_with_history(verbose)
        
        # Check if we're waiting for clarification
        self._update_conversation_state(result)
        
        return {
            "response": self._extract_response(result),
            "waiting_for_clarification": self.waiting_for_clarification,
            "conversation_id": id(self),
            "message_count": len(self.conversation_history)
        }
    
    def continue_conversation(self, user_response: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Continue conversation after user provides clarification details.
        
        Args:
            user_response: User's response to clarification questions
            verbose: Whether to print outputs
            
        Returns:
            Dict containing response and updated conversation state
        """
        if not self.waiting_for_clarification:
            return {
                "error": "Not waiting for clarification. Use start_conversation() for new queries.",
                "waiting_for_clarification": False
            }
        
        print(f"💬 Continuing conversation with: '{user_response[:100]}...'")
        
        # Add user's clarification response to history
        clarification_message = HumanMessage(content=user_response)
        self.conversation_history.append(clarification_message)
        
        # Reset clarification state
        self.waiting_for_clarification = False
        
        # Run workflow again with enriched context
        result = self._run_workflow_with_history(verbose)
        
        # Check if we need more clarification (unlikely but possible)
        self._update_conversation_state(result)
        
        return {
            "response": self._extract_response(result),
            "waiting_for_clarification": self.waiting_for_clarification,
            "conversation_id": id(self),
            "message_count": len(self.conversation_history)
        }
    
    def _run_workflow_with_history(self, verbose: bool) -> Dict[str, Any]:
        """Run the workflow with full conversation history."""
        app = self.agent.compile_workflow()
        inputs = {"messages": self.conversation_history.copy()}
        
        results = []
        for output in app.stream(inputs):
            if verbose:
                for key, value in output.items():
                    if key != "query_router":  # Skip empty router output
                        print(f"Output from node '{key}':")
                        print("---")
                        print(value)
                        print("\n" + "="*50 + "\n")
            results.append(output)
        
        return {
            "results": results,
            "final_state": results[-1] if results else None
        }
    
    def _update_conversation_state(self, result: Dict[str, Any]):
        """Update conversation state based on workflow results."""
        if not result.get("results"):
            return
            
        # Check the final node that was executed
        final_result = result["results"][-1]
        final_node = list(final_result.keys())[0] if final_result else None
        
        if final_node == "smart_clarification":
            self.waiting_for_clarification = True
            self.last_clarification_type = "smart"
            
            # Add AI's clarification message to conversation history
            if final_result[final_node].get("messages"):
                ai_message = final_result[final_node]["messages"][0]
                self.conversation_history.append(ai_message)
                
        elif final_node == "clarification":
            self.waiting_for_clarification = True
            self.last_clarification_type = "general"
            
            # Add AI's clarification message to conversation history
            if final_result[final_node].get("messages"):
                ai_message = final_result[final_node]["messages"][0]
                self.conversation_history.append(ai_message)
        else:
            # Conversation completed normally
            self.waiting_for_clarification = False
            
            # Add final AI response to history
            if final_result.get(final_node, {}).get("messages"):
                for message in final_result[final_node]["messages"]:
                    self.conversation_history.append(message)
    
    def _extract_response(self, result: Dict[str, Any]) -> str:
        """Extract the AI's response from workflow results."""
        if not result.get("results"):
            return "No response generated."
            
        final_result = result["results"][-1]
        if not final_result:
            return "Empty response."
            
        # Get the last node's output
        for node_name, node_output in final_result.items():
            if node_output.get("messages"):
                return node_output["messages"][0].content
                
        return "Unable to extract response."
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation state."""
        return {
            "message_count": len(self.conversation_history),
            "waiting_for_clarification": self.waiting_for_clarification,
            "clarification_type": self.last_clarification_type,
            "conversation_id": id(self),
            "last_message_preview": (
                self.conversation_history[-1].content[:100] + "..." 
                if self.conversation_history and len(self.conversation_history[-1].content) > 100
                else (self.conversation_history[-1].content if self.conversation_history else "No messages")
            )
        }
    
    def get_full_final_response(self) -> str:
        """
        Get the complete final AI response from the conversation.
        
        Returns:
            Complete final AI response or empty string if none found
        """
        if not self.conversation_history:
            return ""
        
        # Find the last AI message
        for msg in reversed(self.conversation_history):
            if isinstance(msg, AIMessage):
                return msg.content
        
        return "No AI response found."
    
    def is_conversation_complete(self) -> bool:
        """
        Check if the conversation has completed successfully.
        
        Returns:
            True if conversation completed (not waiting for clarification)
        """
        return not self.waiting_for_clarification
    
    def get_conversation_status(self) -> Dict[str, Any]:
        """
        Get detailed conversation status including completion and final response.
        
        Returns:
            Dict with conversation status, completion state, and full response
        """
        status = {
            "is_complete": self.is_conversation_complete(),
            "waiting_for_clarification": self.waiting_for_clarification,
            "message_count": len(self.conversation_history),
            "conversation_id": id(self)
        }
        
        if self.is_conversation_complete():
            status["final_response"] = self.get_full_final_response()
            status["status"] = "✅ COMPLETED"
        else:
            status["status"] = "⏳ WAITING FOR CLARIFICATION"
            status["clarification_type"] = self.last_clarification_type
        
        return status
    
    def reset_conversation(self):
        """Reset the conversation to start fresh."""
        self.conversation_history = []
        self.waiting_for_clarification = False
        self.last_clarification_type = None
        print("🔄 Conversation reset.")

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
# CONVERSATION SESSION FACTORY FUNCTIONS
# ============================================================================

def create_conversation_session(agent_type: str = "basic", **kwargs) -> ConversationSession:
    """
    Create a conversation session with the specified agent type.
    
    Args:
        agent_type: Type of agent ("basic", "research", "investment")
        **kwargs: Additional arguments for agent creation
        
    Returns:
        ConversationSession instance ready for multi-turn conversations
    """
    agent_factories = {
        "basic": create_basic_agent,
        "research": create_research_agent,
        "investment": create_investment_advisor
    }
    
    if agent_type not in agent_factories:
        raise ValueError(f"Unknown agent type: {agent_type}. Choose from: {list(agent_factories.keys())}")
    
    agent = agent_factories[agent_type](**kwargs)
    return ConversationSession(agent)

def create_investment_conversation() -> ConversationSession:
    """Create a conversation session optimized for investment advice."""
    agent = create_investment_advisor()
    return ConversationSession(agent)

def create_research_conversation() -> ConversationSession:
    """Create a conversation session optimized for research tasks."""
    agent = create_research_agent()
    return ConversationSession(agent)

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

def demo_conversation_session():
    """
    🗣️ DEMO: Multi-turn conversation with clarification support.
    Shows how context is maintained across clarification cycles.
    """
    print("🗣️ CONVERSATIONAL AI DEMO")
    print("="*50)
    
    # Create a conversation session
    session = create_investment_conversation()
    
    print("1️⃣ Starting with a vague investment query...")
    print("-"*50)
    
    # Start conversation with vague query
    result1 = session.start_conversation(
        "Help me with investing", 
        verbose=True
    )
    
    print(f"\n📊 Conversation State: {session.get_conversation_summary()}")
    
    if result1.get("waiting_for_clarification"):
        print("\n2️⃣ AI asked for clarification! Providing detailed response...")
        print("-"*50)
        
        # Continue with detailed clarification
        detailed_response = """
        Thanks for the questions! Here are the details:
        
        1. Investment goals: I'm saving for retirement, planning to retire in about 20 years
        2. Risk tolerance: I prefer moderate risk - I want growth but can't afford major losses
        3. Timeline: Long-term, 15-20 years until I need the money
        4. Budget: I have about $10,000 to start with and can add $500/month
        5. Experience: I'm a beginner with basic knowledge of stocks and mutual funds
        
        I'm particularly interested in index funds and ETFs. Should I focus on US markets or diversify internationally?
        """
        
        result2 = session.continue_conversation(
            detailed_response,
            verbose=True
        )
        
        # Show detailed final status
        final_status = session.get_conversation_status()
        print(f"\n{final_status['status']}")
        print("="*60)
        
        if final_status["is_complete"]:
            print("🎯 COMPLETE FINAL RESPONSE:")
            print("-"*40)
            print(final_status["final_response"])
            print(f"\n📊 Total conversation turns: {final_status['message_count']}")
        else:
            print("⚠️ Still waiting for more clarification...")
            print(f"Last response preview: {session.get_conversation_summary()['last_message_preview']}")
        
    return session

if __name__ == "__main__":
    # Example usage patterns
    print("🤖 LangGraph Agent Demo - Choose your experience:")
    print("="*55)
    print("1. Single query (traditional)")
    print("2. Multiple queries (batch)")
    print("3. 🗣️  Conversational mode (with clarification support)")
    print("4. Custom agent configuration")
    print("5. 🚀 Automated conversation demo")
    
    choice = input("Enter choice (1-5): ").strip()
    
    if choice == "1":
        agent = create_basic_agent()
        query = input("Enter your query: ")
        agent.run_query(query)
        
    elif choice == "2":
        demo_workflow()
        
    elif choice == "3":
        print("\n🗣️ Starting interactive conversation session...")
        session = create_conversation_session("investment")
        
        query = input("Enter your initial query: ")
        result = session.start_conversation(query)
        
        # Interactive loop for clarification
        while result.get("waiting_for_clarification"):
            print(f"\n💡 The AI is waiting for more details...")
            follow_up = input("Provide additional information: ")
            result = session.continue_conversation(follow_up)
            
        # Show detailed completion status
        status = session.get_conversation_status()
        print(f"\n{status['status']}")
        print("="*60)
        
        if status["is_complete"]:
            print("🎯 FINAL AI RESPONSE:")
            print("-"*40)
            print(status["final_response"])
            print("\n📊 CONVERSATION STATISTICS:")
            print(f"   • Total messages: {status['message_count']}")
            print(f"   • Session ID: {status['conversation_id']}")
        else:
            print("⚠️ Conversation incomplete (unexpected state)")
        
    elif choice == "4":
        print("Creating custom agent...")
        agent = AIAgentWorkflow(
            temperature=0.2,
            model_name="deepseek-reasoner"
        )
        print(f"Custom agent info: {agent.get_workflow_info()}")
        
        query = input("Enter your query: ")
        agent.run_query(query)
        
    elif choice == "5":
        print("\n🚀 Running automated conversation demo...")
        demo_conversation_session()
        
    else:
        print("Running default demo...")
        demo_workflow() 