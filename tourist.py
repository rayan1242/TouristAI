import google.generativeai as genai
from langgraph.graph import StateGraph

# Configure API key (Replace with your actual API key)
genai.configure(api_key="")  # Add your API key

# Initialize Gemini LLM
llm = genai.GenerativeModel("gemini-2.0-flash-001")

# Define agent functions
def travel_recommender(state):
    """Recommends destinations based on user preferences."""
    query = state["query"]
    response = llm.generate_content(f"Recommend travel destinations for {query}.")
    return {"query": query, "recommendation": response.text}  # Return dictionary

def budget_planner(state):
    """Suggests a budget based on recommendations."""
    destination = state["recommendation"]
    response = llm.generate_content(f"Estimate a budget for a trip to {destination}.")
    return {**state, "budget": response.text}  # Keep previous state and add budget

def itinerary_planner(state):
    """Creates an itinerary based on the budget."""
    budget = state["budget"]
    response = llm.generate_content(f"Create a 5-day itinerary under {budget}.")
    return {**state, "itinerary": response.text}  # Add itinerary to the state

# Define initial state
class TravelState:
    def __init__(self, query):
        self.query = query
        self.recommendation = None
        self.budget = None
        self.itinerary = None

    def to_dict(self):
        return {
            "query": self.query,
            "recommendation": self.recommendation,
            "budget": self.budget,
            "itinerary": self.itinerary
        }

# Create a new state graph
workflow = StateGraph(dict)  # Change state type to dictionary

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
result = app.invoke({"query": query})  # Pass dictionary, not object

# Print results
print("Recommended Destination:", result["recommendation"])
print("Estimated Budget:", result["budget"])
print("Planned Itinerary:", result["itinerary"])
