# --- Importing Lib's ---
import os
import tempfile
# from narwhals import col
import streamlit as st

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
from tavily import TavilyClient
from dotenv import load_dotenv

# --- Configuration & Setup ---
load_dotenv()

@st.cache_resource
def get_llm():
    return ChatGroq(
        model='llama-3.3-70b-versatile',
        temperature=0.7,
        groq_api_key=os.getenv('GROQ_API_KEY')
    )

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def get_vector_db():
    return FAISS.from_texts(['System: Vector DB Initialized'], get_embeddings())

llm = get_llm()
embeddings = get_embeddings()
vector_db = get_vector_db()

# Tavily Client is a interface for tavily api key. Allowing web access layer.
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# --- State Definition ---
class OrchestratorState(TypedDict):
    query: str
    plan: str
    research_data: str
    analysis: str
    citations: str
    final_report: str

# --- Agent Functions ---
def root_orchestrator(state: OrchestratorState):
    return {'query': state['query']}

def planner_agent(state: OrchestratorState):
    response = llm.invoke([
        SystemMessage(content="You are a research assistant. Create a plan with: 1) Research Question, 2) Research Plan, 3) Expected Outcomes."),
        HumanMessage(content=f"Research Plan Question for: {state['query']}")
    ])
    return {'plan': response.content}

def research_agent(state: OrchestratorState):
    response = tavily_client.search(query=state['query'], 
                search_depth='advanced', 
                max_results=5
            )

    if not response or 'results' not in response:
        return {"research_data": "Error: No data received."}
    
    results = [f"Title: {item.get('title')} \n Content: {item.get('content')} \n URL: {item.get('url')} \n Published Date: {item.get('published_date')}\n Source: {item.get('source')}" for item in response['results']]

    raw_content = "\n\n".join(results)
    vector_db.add_texts([raw_content])
    return {"research_data": raw_content}

def analysis_agent(state: OrchestratorState):
    response = llm.invoke([
        SystemMessage(content="Analyze research data and extract key insights."),
        HumanMessage(content=f"Data: {state['research_data']}")
    ])
    return {'analysis': response.content}

def citation_agent(state: OrchestratorState): # Fact Checker Agent. 
    response = llm.invoke([
        SystemMessage(content="Generate APA citations for the provided data."),
        HumanMessage(content=f"Data: {state['research_data']}")
    ])
    return {'citations': response.content}

def format_agent(state: OrchestratorState):

    docs = vector_db.similarity_search(state['query'], k=3)

    memory_context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    response = llm.invoke([
        SystemMessage(content="""
You are an elite AI Research Scholar Assistant.

Generate a BEAUTIFUL, visually engaging research report using markdown formatting.

STRICT FORMATTING RULES:
- Use emojis in section titles
- Use markdown headings (#, ##, ###)
- Use bullet points
- Use arrows (→, ⇒, ➜)
- Highlight important findings using bold text
- Add separators using ---
- Make sections visually clean
- Use concise paragraphs
- Include a final "Key Takeaways" section
- Include APA References section
- Use tables whenever useful

FORMAT STRUCTURE:

# 📘 Title

## 🎯 Introduction

## 🧠 Research Methodology

## 🔍 Key Analysis

## 📊 Findings & Results

## 🚀 Major Insights

## ⚠ Challenges & Limitations

## ✅ Conclusion

## 💡 Key Takeaways

## 📚 References

Make the report modern and attractive.
"""),

        HumanMessage(content=f"""
Research Query:
{state['query']}

Analysis:
{state['analysis']}

Citations:
{state['citations']}

Memory Context:
{memory_context}
""")
    ])

    return {'final_report': response.content}


# --- Workflow Setup ---
workflow = StateGraph(OrchestratorState)
workflow.add_node('root_orchestrator', root_orchestrator)
workflow.add_node('planner_agent', planner_agent)
workflow.add_node('research_agent', research_agent)
workflow.add_node('analysis_agent', analysis_agent)
workflow.add_node('citation_agent', citation_agent)
workflow.add_node('format_agent', format_agent)

workflow.add_edge(START, 'root_orchestrator')
workflow.add_edge('root_orchestrator', 'planner_agent')
workflow.add_edge('planner_agent', 'research_agent')
workflow.add_edge('research_agent', 'analysis_agent')
workflow.add_edge('analysis_agent', 'citation_agent')
workflow.add_edge('citation_agent', 'format_agent')
workflow.add_edge('format_agent', END)

app = workflow.compile(checkpointer=MemorySaver())

# --- Streamlit UI ---
st.set_page_config(page_title="Research Scholar Agent", page_icon="📚", layout='centered')
st.title("🔍 Research Scholar Agent")

with st.sidebar:
    st.header("Document Upload")
    upload_file = st.file_uploader('Upload your file here.', type=['pdf', 'txt', 'docx', 'csv'])
    if upload_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{upload_file.name.split(".")[-1]}') as tmp_file:
            tmp_file.write(upload_file.read())
            tmp_file_path = tmp_file.name
        
        loaders = {
            'pdf': PyPDFLoader,
            'txt': TextLoader, 
            'docx': Docx2txtLoader, 
            'csv': CSVLoader
        }

        loader_class = loaders.get(upload_file.name.split('.')[-1])
        if loader_class:
            pages = loader_class(tmp_file_path).load_and_split() # reads & splits the document into pages.
            vector_db.add_texts([p.page_content for p in pages])
            st.success(f"{upload_file.name} processed!")
        os.remove(tmp_file_path)
    
    st.write('Dummy Examples:')
    quick_queries = [
        'Applications of AI in healthcare',
        'Recent NLP techniques for chat bots',
        'Quantum Computing recent trends',
        'Ai applications in cyber security'
    ]

    for q in quick_queries:

        if st.button(q, use_container_width=True):

            with st.spinner('🤖 Agent is researching...'):

                progress = st.progress(0)

                progress.progress(15, text="🧠 Planning Research...")
                progress.progress(35, text="🌐 Collecting Web Data...")
                progress.progress(55, text="📊 Analyzing Research...")
                progress.progress(75, text="📚 Generating Citations...")
                progress.progress(100, text="✅ Building Final Report...")

                result = app.invoke(
                    {"query": q},
                    config={
                        "configurable": {
                            "thread_id": f"streamlit_session_{q[:5]}"
                        }
                    }
                )

                st.session_state['last_result'] = result['final_report']
                st.success("✅ Research Complete!")

if 'last_result' in st.session_state:

    st.markdown("## 📘 Final Research Report")

    st.markdown(
        st.session_state['last_result'],
        unsafe_allow_html=True
    )

query = st.text_input("Enter your research topic here:", "")

if st.button("Run Research"):
    if query:
        with st.spinner("Agent is researching..."):
            config = {"configurable": {"thread_id": "streamlit_session_001"}}
            result = app.invoke({"query": query}, config=config)
            st.success("Research Complete!")
            st.markdown("### Final Research Report")
            st.markdown(result["final_report"], unsafe_allow_html=True)
    else:
        st.warning("Please enter a query.")