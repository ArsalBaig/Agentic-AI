import streamlit as st
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

load_dotenv()

@st.cache_resource
def get_workflow():
    model = ChatGroq(model='llama-3.3-70b-versatile', temperature=0)

    class ChatState(TypedDict):
        msgs: Annotated[list[BaseMessage], add_messages]

    def chatting(state: ChatState):
        msgs = model.invoke(state['msgs'])
        return {'msgs': msgs}

    graph = StateGraph(ChatState)
    graph.add_node('chatting', chatting)
    graph.add_edge(START, 'chatting')
    graph.add_edge('chatting', END)
    return graph.compile()

workflow = get_workflow()

st.set_page_config(page_title='LangGraph ChatBot', layout='centered')
st.title('LangGraph ChatBot')

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg.type): # Decides Human or Ai icon.
        st.markdown(msg.content)

# ':=' is the warlus operator.
if prompt := st.chat_input('How can I help you?'):

    human_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(human_msg)
    with st.chat_message("human"):
        st.markdown(prompt)

    with st.spinner('Thinking...'):
        response = workflow.invoke({'msgs': st.session_state.messages})
        ai_msg = response['msgs'][-1]
    st.session_state.messages.append(ai_msg)
    with st.chat_message("ai"):
        st.markdown(ai_msg.content)