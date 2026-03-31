# =================================================================
# Importing Libraries.
# =================================================================

import csv
import json
import inspect
import zipfile
import io
import streamlit as st

from typing import List, Dict
from collections import defaultdict
from groq import Groq
from dotenv import load_dotenv

# =================================================================
# 1. INITIALIZATION & DATA LOADING
# =================================================================

load_dotenv()
client = Groq()


@st.cache_resource(show_spinner="Loading dataset…")
def load_dataset(filepath : str):
    ''' Load and index the retail CSV dataset into lookup dictionaries.'''
    order_items: Dict[str, List[Dict]] = defaultdict(list)
    order_dates: Dict[str, str] = {}
    order_customer: Dict[str, str] = {}
    customer_orders: Dict[str, List[str]] = defaultdict(list) # defaultdict will automatically create an empty list if key,value pair doesn't exsist.
    customer_country: Dict[str, str] = {}
    item_prices: Dict[str, float] = {}

    with zipfile.ZipFile(filepath, 'r') as z:
        csv_filename = [name for name in z.namelist() if name.endswith('.csv')][0]
        with z.open(csv_filename) as f:
            f_text = io.TextIOWrapper(f, encoding='latin-1')
            reader = csv.DictReader(f_text)
            
            for row in reader:
                invoice = row.get('InvoiceNo', '').strip()
                customer = row.get('CustomerID', '').strip()
                desc = row.get('Description', '').strip()
                date = row.get('InvoiceDate', '').strip()
                country = row.get('Country', '').strip()
                qty_raw = row.get('Quantity', '0').strip()
                price_raw = row.get('UnitPrice', '0').strip()
                stock = row.get('StockCode', '').strip()

                if not invoice or not desc:
                    continue # skips the entire row.

                try:
                    # Tries to convert the text '10' into integer '10'.
                    qty = int(qty_raw)
                    price = float(price_raw)
                except ValueError:
                    continue

                # Single Invoice No with multiple products.
                order_items[invoice].append({
                    'item' : desc,
                    'stock' : stock,
                    'quantity' : qty,
                    'unit_price' : price
                })

                if invoice not in order_dates and date:
                    order_dates[invoice] = date
 
                if customer:
                    order_customer[invoice] = customer
                    if invoice not in customer_orders[customer]:
                        customer_orders[customer].append(invoice)
                    customer_country[customer] = country
 
                item_prices[desc] = price 
                
    return order_items, order_dates, order_customer, customer_orders, customer_country, item_prices

DATA_PATH = r'Projects of Agentic AI\Customer Supp Agent\data.csv.zip'

(ORDER_ITEMS, ORDER_DATES, ORDER_CUSTOMER, CUSTOMER_ORDERS, CUSTOMER_COUNTRY, ITEM_PRICES) = load_dataset(DATA_PATH)

# =================================================================
# 2. CORE BUSINESS LOGIC (Internal Functions)
# =================================================================

# Return Policy.
def _return_days_allowed_and_price(price : float) -> int:
    '''Based on price function tells how many days the customer has to return the item.'''
    if price <= 0:
        return 0
    elif price < 2.0:
        return 7
    elif price < 5.0:
        return 14
    elif price < 20.0:
        return 30
    else:
        return 45
    
def get_order_items(order_id : str) -> List[str]:
    '''By giving an order-item(InvoiceNo) return the list of items description purchased in that order.'''
    items = ORDER_ITEMS.get(order_id)
    if not items:
        return 'No items found!'
    else:
        return [row['item'] for row in items]
    
def get_delivery_date(order_id : str) -> str:
    '''By giving an order-item(InvoiceNo) return the dispatch date for that order.'''
    date = ORDER_DATES.get(order_id)
    if not date:
        return f'Order {order_id} not found!'
    else:
        return date
    
def get_order_total(order_id : str) -> str:
    '''By giving an order-item(InvoiceNo) return the total money for that order.'''
    items = ORDER_ITEMS.get(order_id)
    if not items:
        return f'Order {order_id} not found!'
    else:
        total =  sum(row['quantity'] * row['unit_price'] for row in items)
        return f'${total:.2f}'
    
def get_customer_orders(customer_id: str) -> List[str]:
    '''Given a CustomerID, return all order IDs (InvoiceNos) placed by
    that customer.'''
    orders = CUSTOMER_ORDERS.get(customer_id)
    if not orders:
        return [f"No orders found for customer {customer_id}."]
    return orders

def get_item_return_days(item: str) -> int:
    '''By giving an item description, return the number of days within which
    it can be returned, based on its unit price tier.'''
    
    item_upper = item.upper()
    matched_price = None
    for desc, price in ITEM_PRICES.items():
        if desc.upper() == item_upper:
            matched_price = price
            break
 
    if matched_price is None:
        return 30  # default 
    return _return_days_allowed_and_price(matched_price)

def get_order_return_policy(order_id: str) -> str:
    '''By giving an order ID, return a summary of the return eligibility for every item in that order.'''

    items = ORDER_ITEMS.get(order_id)
    if not items:
        return f"Order {order_id} not found."
 
    lines = [f"Return policy for order {order_id}:"]
# Since one order contain many items.
    for row in items:
        days = _return_days_allowed_and_price(row["unit_price"]) # Returns how many days for this price.
        if days == 0:
            eligibility = "not eligible for return"
        else:
            eligibility = f"returnable within {days} days"
        lines.append(f"  • {row['item']} (${row['unit_price']:.2f}) — {eligibility}")
    return "\n".join(lines)

# =================================================================
# 3. AGENT TOOLS & WRAPPERS
# =================================================================

class SupportQueryAgent:
    '''Simulates a LlamaIndex QueryEngine over support documentation.'''
    
    def __init__(self):
        self.support_documents = {
            'return_policy' : (
                'Return Policy: Items priced under £2 may be returned within 7 days.'
                'Items £2–£5 within 14 days. Items £5–£20 within 30 days.'
                'Items over £20 within 45 days. Postage and free items are non-returnable.'
            ),
            'contact_info' : (
                'Customer support: Phone 1-987-654-3210'
                'Email support@company.com, Hours: Mon–Fri 9AM–6PM EST.'
            ),
            'shipping': (
                "Standard shipping: 5–7 business days. "
                "Express shipping: 2–3 business days. "
                "Free standard shipping on orders over £100."
            ),
            'cancellation': (
                "Orders can be cancelled within 1 hour of placement. "
                "After dispatch, please use our returns process instead."
            ),
        }
    
    def query(self, query_input : str) -> str:
        q = query_input.lower()

        if 'return' in q or 'policy' in q:
            return self.support_documents['return_policy']
        elif 'cancel' in q or 'cancellation' in q:
            return self.support_documents['cancellation']
        elif 'contact' in q or 'support' in q or 'phone' in q or 'email' in q:
            return self.support_documents['contact_info']
        elif 'ship' in q or 'deliver' in q:
            return self.support_documents['shipping']
        return self.support_documents['contact_info']

support_query_agent = SupportQueryAgent()

# This Class Uses Python's inspect library to read your function's name and docstrings to automatically create the "Instruction Manual" the AI needs.
class FunctionTool:
    """Mimics LlamaIndex FunctionTool.from_defaults()"""
 
    def __init__(self, fn):
        self.fn = fn
        self.name = fn.__name__
        self.description = fn.__doc__
 
    @staticmethod # Calling the method without creating an instance of the class.
    def from_defaults(fn):
        return FunctionTool(fn)
 
    def to_groq_tool(self):
        sig = inspect.signature(self.fn) # It looks at your function and figures out what parameter it needs.
        properties, required = {}, []
        for param_name, param in sig.parameters.items():
            param_type = "integer" if param.annotation == int else "string" # It looks for annotation (:int) if integer then returns integer and so on.
            properties[param_name] = {
                "type": param_type,
                "description": f"The {param_name.replace('_', ' ')} to look up"  
            }
            required.append(param_name)
# The format that Groq expects 
        return {
            'type': 'function',
            'function': {
                'name': self.name,
                'description': self.description,
                'parameters': {
                    'type': 'object',
                    'properties': properties,
                    'required': required
                },
            },
        }

# This class specifically wraps the SupportQueryAgent so the AI knows it can "search" for general information not found in the database.
class QueryEngineTool:
    '''Mimics LlamaIndex QueryEngineTool.from_defaults()''' 
    def __init__(self, query_engine, description):
        self.query_engine = query_engine
        self.name = 'search_support_docs'
        self.description = description
 
    @staticmethod
    def from_defaults(query_engine, description):
        return QueryEngineTool(query_engine, description)
# Converts JSON Schema into required by Groq.
    def to_groq_tool(self):
        return {
            'type': 'function',
            'function': {
                'name': self.name, # It was an search_support_docs function.
                'description': self.description,
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'input': {
                            'type': 'string', 
                            'description': 'Query to search support documentation'
                        }
                    },
                    'required': ['input'],
                },
            },
        }

# Tool Registration.
order_item_tool = FunctionTool.from_defaults(fn=get_order_items)
delivery_date_tool   = FunctionTool.from_defaults(fn=get_delivery_date)
order_total_tool     = FunctionTool.from_defaults(fn=get_order_total)
customer_orders_tool = FunctionTool.from_defaults(fn=get_customer_orders)
return_policy_tool   = FunctionTool.from_defaults(fn=get_item_return_days)
order_return_tool    = FunctionTool.from_defaults(fn=get_order_return_policy)

support_tool = QueryEngineTool.from_defaults(
    query_engine=support_query_agent.query,
    description="Tool to query customer support documentation for return policies, shipping info, contact details, and cancellation policies."
)

# make a list of all tools for batch processing.
all_tools = [
    order_item_tool,
    delivery_date_tool,
    order_total_tool,
    customer_orders_tool,
    return_policy_tool,
    order_return_tool,
    support_tool,
]

groq_tools = [t.to_groq_tool() for t in all_tools]

# =================================================================
# 4. AGENT EXECUTION ENGINE
# =================================================================

# The class acts as a "Manager" that identifies which tool the AI is asking for and routes the arguments to the correct Python function.
class FunctionCallingAgentWorker:
    """Mimics LlamaIndex FunctionCallingAgentWorker"""
 
    def __init__(self, tools, verbose=True):
        self.tools = tools
        self.verbose = verbose # If verbose is True, it will print the tool calls and responses.
        self.groq_tools = [t.to_groq_tool() for t in tools]
 
    @staticmethod
    def from_tools(tools, verbose=True):
        return FunctionCallingAgentWorker(tools, verbose)
    
    def execute_tool(self, tool_name: str, tool_args: dict):
        for tool in self.tools:
        # 1- QueryEngine Tool.
            if isinstance(tool, QueryEngineTool) and tool.name == tool_name: # Checks if the current tool matches with QueryEngineTool & if it's name matches with the one requested.
                if self.verbose:
                    print(f'\n=== Calling Function ===')
                    print(f'{tool_name}({tool_args})')
                result = tool.query_engine.query(tool_args.get("input", "")) # This line is req when AI want to read something from documents.
                if self.verbose:
                    print(f'  → {result}')
                return json.dumps({"result": result}) # convert into json format.
        # 2- Function Tool.
            if isinstance(tool, FunctionTool) and tool.name == tool_name:
                if self.verbose:
                    print(f'\n=== Calling Function ===')
                    print(f'  {tool_name}({tool_args})')
                result = tool.fn(**tool_args)
                if self.verbose:
                    print(f'  → {result}')
                return json.dumps({"result": result})
        return json.dumps({"error": f"Tool '{tool_name}' not found."})
    
SYSTEM_PROMPT = (
    'You are a helpful customer service AI agent for an online retail store. '
    'You have access to tools that can look up order items, invoice dates, order totals, '
    'customer order history, item return windows, and full order return policies. '
    'Always be friendly, precise, and use the tools to retrieve accurate data before answering. '
    'IMPORTANT: Only call the tools necessary to answer the user\'s exact question. '  # Prevents the model from calling extra tools beyond what was asked.
    'Do not volunteer extra information like totals, dates, or policies unless explicitly asked.'  # Stops over-answering e.g. giving total when only items were requested.
)

def execute_tool(tool_name: str, tool_args: dict) -> str:
    for tool in all_tools:
        if isinstance(tool, QueryEngineTool) and tool.name == tool_name:
            return json.dumps({"result": tool.query_engine(tool_args.get("input", ""))}) 
        if isinstance(tool, FunctionTool) and tool.name == tool_name:
            return json.dumps({"result": tool.fn(**tool_args)})
    return json.dumps({"error": f"Tool '{tool_name}' not found."}) 
 
def run_agent(user_message: str, history: list) -> str:
    # Initialize the message as SystemMessage.
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        messages.append({"role": "user",      "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append({"role": "user", "content": user_message})
 
    while True:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1024,
            tools=groq_tools,       # List of tools the agent can use.
            messages=messages,      # Contains system instructions.
            parallel_tool_calls=False,  
        )
        finish = resp.choices[0].finish_reason
 
        if finish == "stop":
            return resp.choices[0].message.content
 
        elif finish == "tool_calls":
            asst = resp.choices[0].message
            messages.append({
                "role": "assistant",
                "content": asst.content,
                "tool_calls": asst.tool_calls,
            })
            for tc in asst.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.function.name,
                        "content": json.dumps({"error": "Invalid tool arguments received."}),
                    })
                    continue
                result = execute_tool(tc.function.name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": result,
                })
        else:
            return "No response generated."

# =================================================================
# 5. STREAMLIT UI LAYOUT & INTERACTION
# =================================================================

st.set_page_config(page_title="Customer Support Agent", page_icon="🛍️", layout="centered")
 
st.title("🛍️ Customer Support Agent")
st.caption("Ask about orders, returns, shipping, or contact info.")
st.divider()
 
# Session state
if 'history' not in st.session_state:
    st.session_state.history = []
 
# --------------- Quick-action buttons ----------------

st.write("**Quick actions**")
quick_queries = [
    "Items in order 536365?",
    "Total for order 536370?",
    "Return policy for order 536367?",
    "How to contact support Team?",
]
 
cols = st.columns(4)
for col, q in zip(cols, quick_queries): # This ensures each button sits in its own column space.
    if col.button(q, use_container_width=True):
        with st.spinner("Thinking…"):
            answer = run_agent(q, st.session_state.history)
        st.session_state.history.append({"user": q, "assistant": answer})
        st.rerun()
 
st.divider()
 
# --------------- Chat History ----------------
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["user"])
    with st.chat_message("assistant"):
        st.write(turn["assistant"])
 
# --------------- Chat Input ----------------
user_input = st.chat_input("Type your question here…")
if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer = run_agent(user_input, st.session_state.history) # Sends user_input & conv history to AI Agent.
        st.write(answer)
    st.session_state.history.append({"user": user_input, "assistant": answer})
 
# --------------- Clear Chat History ----------------
if st.session_state.history:
    st.divider()
    if st.button("🗑️ Clear chat", type="secondary"):
        st.session_state.history = []
        st.rerun()