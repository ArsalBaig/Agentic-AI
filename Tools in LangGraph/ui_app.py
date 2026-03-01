# --------- Importing Libs ---------
import streamlit as st
import yfinance as yf

from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

load_dotenv()

# --------- Initializing LLM ---------
@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

# --------- Defining Tools ---------

# ✅ Wrap DuckDuckGo in a proper @tool with explicit schema
@tool
def web_search(query: str) -> str:
    """Search the web for current information using DuckDuckGo. Use this for news, facts, or recent events."""
    searcher = DuckDuckGoSearchRun()
    return searcher.run(query)

@tool
def calculator(fst_no: float, scd_no: float, operator: str) -> dict:
    """Perform arithmetic on two numbers. Operator must be: add, subtract, multiply, or divide."""
    try:
        if operator == 'add':
            result = fst_no + scd_no
        elif operator == 'subtract':
            result = fst_no - scd_no
        elif operator == 'multiply':
            result = fst_no * scd_no
        elif operator == 'divide':
            if scd_no == 0:
                return {'error': 'Division by zero is infinity!'}
            result = fst_no / scd_no
        else:
            return {'error': f'Unknown operator: {operator}'}
        return {'first no': fst_no, 'second no': scd_no, 'result': result}
    except Exception as e:
        return {'error': str(e)}

@tool
def stock_price_predictor(symbol: str) -> dict:
    """Fetch latest stock price for a given symbol (e.g. 'OGDC', 'HBL', 'KEL')."""
    if not symbol.endswith('.KA'):
        symbol = symbol + '.KA'
    ticker = yf.Ticker(symbol)
    data = ticker.history(period='1d')
    if data.empty:
        return {'error': f'No data found for symbol {symbol}!'}
    latest = data.iloc[-1]
    return {
        'symbol': symbol,
        'price': round(latest['Close'], 3)
    }

# --------- Building Workflow ---------
@st.cache_resource
def build_workflow():
    model = load_llm()
    tool_list = [web_search, calculator, stock_price_predictor]  # ✅ use wrapped tool
    llm_with_tools = model.bind_tools(tool_list)

    class ChatState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    def chat_node(state: ChatState):
        system = {"role": "system", "content": (
            "You are a helpful assistant. Use tools only when needed and only relevant ones. "
            "Do not use extra tools or perform unrequested calculations. "
            "Do not use emojis in your responses."
        )}
        response = llm_with_tools.invoke([system] + state['messages'])
        return {'messages': [response]}

    tool_node = ToolNode(tool_list)

    graph = StateGraph(ChatState)
    graph.add_node('chat_node', chat_node)
    graph.add_node('tools', tool_node)

    graph.add_edge(START, 'chat_node')
    graph.add_conditional_edges('chat_node', tools_condition)
    graph.add_edge('tools', 'chat_node')

    return graph.compile()

# --------- Streamlit UI ---------
st.set_page_config(page_title='🛠️ Tools in LangGraph', layout='centered')
st.title('🛠️ Tools in LangGraph')

with st.sidebar:
    st.markdown('### Example Queries')
    st.markdown('🧮 **Calculator**')
    st.code('Divide 9600 by 4')
    st.code('What is 50 * 4')
    st.markdown('📈 **Stock Price (PSX)**')
    st.code('Stock price of OGDCL, HBL, KEL, PEL')
    st.markdown('🔍 **Web Search**')
    st.code('Latest news of Pakistan Economy?')
    st.code('Who won the T20 worldcup of 2011')

workflow = build_workflow()

if 'messages' not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg.type):
        st.markdown(msg.content)

# --------- Handling User Input ---------
if prompt := st.chat_input('How can I assist you?'):
    human_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(human_msg)

    with st.chat_message('human'):
        st.markdown(prompt)

    with st.spinner('Analyzing...'):
        try:
            response = workflow.invoke({'messages': st.session_state.messages})
            ai_msg = response['messages'][-1]
            st.session_state.messages = response['messages']
            with st.chat_message('ai'):
                st.markdown(ai_msg.content)
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")