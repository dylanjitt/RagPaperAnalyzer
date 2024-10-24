import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai_tools import FileReadTool, PDFSearchTool, CodeDocsSearchTool
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext

# Load environment variables
load_dotenv()

llm = LLM(model="gpt-4o-mini")
current_dir = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# Create VectorStoreIndex for PDFs
# -----------------------------
def create_pdf_index(directory_path):
    documents = SimpleDirectoryReader(directory_path).load_data()
    index = VectorStoreIndex.from_documents(documents)
    # Persist the index locally for reuse
    index.storage_context.persist(persist_dir="storage")
    return index

# Check if a pre-existing index exists, otherwise create it
if os.path.exists("storage"):
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    pdf_index = VectorStoreIndex.load(storage_context=storage_context)
else:
    pdf_index = create_pdf_index(directory_path="tarea 3/data")

query_engine = pdf_index.as_query_engine()

# Example query (adjust the question for your needs)
response = query_engine.query("Summarize the key contributions of the AI paper.")
print(response)

# -----------------------------
# Define Coding Agent
# -----------------------------
coding_agent = Agent(
    role="Senior Python Developer",
    goal="Craft well-designed and thought-out code. Also perform thorough and concise code reviews.",
    backstory="""You are a senior Python developer with extensive experience 
    in software architecture, programming best practices, and code reviews.""",
    allow_code_execution=True,
    llm=llm,
)

# -----------------------------
# Define PDF RAG Task
# -----------------------------
pdf_rag_task = Task(
    description="Use the PDFSearchTool to retrieve relevant information from the AI paper.",
    expected_output="A summary of the AI paper's key points using the PDF RAG tool.",
    agent=coding_agent,
    tools=[PDFSearchTool(file_path=os.path.join(current_dir, "data/Lect-7-DM.pdf"))],
    output_file="ai_paper_summary.txt"
)

# -----------------------------
# Define FastAPI RAG Task
# -----------------------------
fastapi_rag_task = Task(
    description="Retrieve key information from FastAPI docs related to building an API using CodeDocsSearchTool.",
    expected_output="A summary of the key information needed for building the API using FastAPI.",
    agent=coding_agent,
    tools=[CodeDocsSearchTool(doc_path=os.path.join(current_dir, "data/fastapi_docs"))],
    output_file="fastapi_docs_summary.txt"
)

# -----------------------------
# Define Scipy RAG Task
# -----------------------------
scipy_rag_task = Task(
    description="Use the CodeDocsSearchTool to retrieve relevant information from the Scipy documentation.",
    expected_output="A summary of the key Scipy functionalities for mathematical operations.",
    agent=coding_agent,
    tools=[CodeDocsSearchTool(doc_path=os.path.join(current_dir, "data/scipy_docs"))],
    output_file="scipy_docs_summary.txt"
)

# -----------------------------
# Analyzer Task (Combine RAG Results)
# -----------------------------
analyze_task = Task(
    description="Combine all RAG outputs to create a system architecture based on the AI paper, FastAPI, and Scipy docs.",
    expected_output="""A combined analysis summarizing all the information 
    and outlining the architecture needed to implement the AI project.""",
    agent=coding_agent,
    tools=[FileReadTool(file_path=os.path.join(current_dir, "ai_paper_summary.txt")),
           FileReadTool(file_path=os.path.join(current_dir, "fastapi_docs_summary.txt")),
           FileReadTool(file_path=os.path.join(current_dir, "scipy_docs_summary.txt"))],
    output_file="project_architecture.txt"
)

# -----------------------------
# Coder Task (Generate Python Code)
# -----------------------------
coding_task = Task(
    description="""Using the system architecture and information from the AI paper, FastAPI, 
    and Scipy docs, generate Python code for the AI project.""",
    expected_output="A fully functional Python implementation of the AI project.",
    agent=coding_agent,
    tools=[FileReadTool(file_path=os.path.join(current_dir, "project_architecture.txt"))],
    output_file="ai_project_code.py"
)

# -----------------------------
# Define the Crew (Pipeline)
# -----------------------------
dev_crew = Crew(
    agents=[coding_agent],
    tasks=[pdf_rag_task, fastapi_rag_task, scipy_rag_task, analyze_task, coding_task],
    verbose=True
)

# Run the Crew
result = dev_crew.kickoff()

# Output the result of the crew execution
print(result)

