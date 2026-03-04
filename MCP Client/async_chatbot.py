# Note: LangGraph requries async to deal with MCP Client.
import asyncio
import streamlit as st
import yfinance as yf

from langgraph.graph import StateGraph, START
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool

load_dotenv()

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

@tool
def web_search(query: str) -> str:
    """Search the web for current information using DuckDuckGo. Use this for news, facts, or recent events."""
    searcher = DuckDuckGoSearchRun()
    return searcher.run(query)

@tool
def calculator(fst_no: float, scd_no: float, operator: str) -> dict:
    """Perform arithmetic on two numbers. Operator MUST be one of: add, subtract, multiply, divide, or modulus."""
    try:
        if operator == 'add':
            result = fst_no + scd_no
        elif operator == 'subtract':
            result = fst_no - scd_no
        elif operator == 'multiply':
            result = fst_no * scd_no
        elif operator == 'modulus':
            result = fst_no % scd_no
        elif operator == 'divide':
            if scd_no == 0:
                return {'error': 'Division by zero is infinity!'}
            result = fst_no / scd_no
        else:
            return {'error': f'Unknown operator: {operator}'}
        return {'fst_no': fst_no, 'scd_no': scd_no, 'result': result}
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

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

@st.cache_resource
def build_workflow():
    model = load_llm()
    tool_list = [web_search, calculator, stock_price_predictor]
    llm_with_tools = model.bind_tools(tool_list)
    return llm_with_tools, tool_list

def build_graph():
    llm_with_tools, tool_list = build_workflow()

# Async function that acts as a node.
    async def chat_node(state: ChatState):
        system = SystemMessage(content=(
            "You are a helpful assistant. Use tools only when needed and only relevant ones. "
            "When using the calculator, ensure you use the exact parameter names: fst_no and scd_no. "
            "When you have to search using your tools. Then Just answer to user_query only not the other function(arthematic operations etc)"
            "Do not use emojis in your responses."
        ))
# 'await' & 'ainvoke' allows to run without freezing.
        response = await llm_with_tools.ainvoke([system] + state['messages'])
        return {'messages': [response]}

    tool_node = ToolNode(tool_list)

    graph = StateGraph(ChatState)
    graph.add_node('chat_node', chat_node)
    graph.add_node('tools', tool_node)

    graph.add_edge(START, 'chat_node')
    graph.add_conditional_edges('chat_node', tools_condition)
    graph.add_edge('tools', 'chat_node')

    return graph.compile()

async def main():
    chatbot = build_graph()
    result = await chatbot.ainvoke({"messages": [HumanMessage(content="Calculate the modulus of 45678 and 123.")]})
    print(result['messages'][-1].content)

if __name__ == '__main__':
    asyncio.run(main())



# Note:
# When you want to use you own file like in 'arith' or using online mcp server you have
# -to use 'expense' code. and in-order to use mcp client use the below code in async code file.

# client = MultiServerMCPClient(
#     {
#         "arith": {
#             "transport": "stdio",
#             "command": "python3",          
#             "args": ["/Users/nitish/Desktop/mcp-math-server/main.py"],
#         },
#         "expense": {
#             "transport": "streamable_http",  # if this fails, try "sse"
#             "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
#         }
#     }
# )