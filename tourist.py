import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Set Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBrIDfUxkk-6y48lFzjAUYYh4LfIvXwHic"

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Import required classes
from langgraph.graph import StateGraph
from langchain.schema import SystemMessage, HumanMessage

# Define agent functions
def travel_recommender(state):
    """Recommends destinations based on user preferences."""
    query = state.query
    response = llm.invoke(f"Recommend travel destinations for {query}.")
    state.recommendation = response.content
    return state

def budget_planner(state):
    """Suggests a budget based on recommendations."""
    destination = state.recommendation
    response = llm.invoke(f"Estimate a budget for a trip to {destination}.")
    state.budget = response.content
    return state

def itinerary_planner(state):
    """Creates an itinerary based on the budget."""
    budget = state.budget
    response = llm.invoke(f"Create a 5-day itinerary under {budget}.")
    state.itinerary = response.content
    return state

# Define state structure
class TravelState:
    def __init__(self, query):
        self.query = query
        self.recommendation = None
        self.budget = None
        self.itinerary = None

# Create a new state graph
workflow = StateGraph(lambda: TravelState(query))

# Add nodes (agents)
workflow.add_node("travel_recommender", travel_recommender)
workflow.add_node("budget_planner", budget_planner)
workflow.add_node("itinerary_planner", itinerary_planner)

# Define workflow connections
workflow.set_entry_point("travel_recommender")  # First step
workflow.add_edge("travel_recommender", "budget_planner")
workflow.add_edge("budget_planner", "itinerary_planner")

# Compile the graph
app = workflow.compile()

# Execute workflow with input query
query = "a budget-friendly trip to Japan"
result = app.invoke({"query": query})

# Print results
print("Recommended Destination:", result["recommendation"])
print("Estimated Budget:", result["budget"])
print("Planned Itinerary:", result["itinerary"])