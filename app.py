from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate


llm=ChatGroq(model="llama3-70b-8192")

class State(TypedDict):
    candidate_application : str
    experience_level : str
    skill_match : str
    response : str    
    
    


graph = StateGraph(State)

def categorize_candidate_experience(state : State) -> State : 
    prompt=ChatPromptTemplate.from_template(template=
        "Based on the below job application, categorize the candidate as: 'Entry-level' or 'Mid-level' or 'Senior-level' "
        "Application : {application}"
        "No preamble. Just one word answer from the above category in string"
)

    chain=prompt | llm
    experience_level = chain.invoke({"application": state["candidate_application"]}).content
    return {"experience_level" : experience_level}


def assess_skillset(state: State) -> State:
  prompt = ChatPromptTemplate.from_template(
      "Based on the job application for a physics internship , assess the candidate's skillset"
      "Respond with either 'Match' or 'No Match'"
      "Application : {application}"
      "No preamble. Just a string datatype."
  )
  chain = prompt | llm
  skill_match = chain.invoke({"application": state["candidate_application"]}).content
  return {"skill_match" : skill_match}


def schedule_hr_interview(state: State) -> State:
  print("\nScheduling the interview : ")
  return {"response" : "Candidate has been shortlisted for an HR interview."}

def escalate_to_recruiter(state: State) -> State:
  print("Escalating to recruiter")
  return {"response" : "Candidate has senior-level experience but doesn't match job skills."}

def reject_application(state: State) -> State:
  print("Sending rejecting email")
  return {"response" : "Candidate doesn't meet JD and has been rejected."}

graph.add_node("categorize_experience", categorize_candidate_experience)
graph.add_node("assess_skillset", assess_skillset)
graph.add_node("schedule_hr_interview", schedule_hr_interview)
graph.add_node("escalate_to_recruiter", escalate_to_recruiter)
graph.add_node("reject_application", reject_application)

def route_app(state: State) -> str:
  if(state["skill_match"] == "Match"):
    return "schedule_hr_interview"
  elif(state["experience_level"] == "Senior-level"):
    return "escalate_to_recruiter"
  else:
    return "reject_application"


graph.add_edge("categorize_experience", "assess_skillset")
graph.add_conditional_edges("assess_skillset", route_app)


graph.add_edge(START, "categorize_experience")
graph.add_edge("assess_skillset", END)
graph.add_edge("escalate_to_recruiter", END)
graph.add_edge("reject_application", END)
graph.add_edge("schedule_hr_interview", END)


app = graph.compile()


def run_candidate_screening(application: str):
  results = app.invoke({"candidate_application" : application})
  return {
      "experience_level" : results["experience_level"],
      "skill_match" : results["skill_match"],
      "response" : results["response"]
  }
  
  
application_text="I have a experinece of 20 years in physics department."

results = run_candidate_screening(application_text)
print("\n\nComputed Results :")
print(f"Application: {application_text}")
print(f"Experience Level: {results['experience_level']}")
print(f"Skill Match: {results['skill_match']}")
print(f"Response: {results['response']}")