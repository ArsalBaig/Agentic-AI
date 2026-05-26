# 🧠 AI Blog Generation Agent with Image Support

An advanced Agentic AI-powered Blog Generation System built using LangGraph, LLMs, and automated image planning. This project generates structured, research-oriented blog posts with AI-generated image placeholders and intelligent workflow orchestration.

---

# 🚀 Features

* ✍️ AI-powered blog generation using LLMs
* 🧩 Multi-agent workflow orchestration with LangGraph
* 🔎 Research-aware content planning and routing
* 🖼️ Automatic image planning and placeholder generation
* 📑 Structured blog section generation
* 🧠 Pydantic-based schema validation
* ⚡ Groq Llama 3.3 integration for fast inference
* 🌐 Tavily/Web research support
* 📂 Markdown-based final blog output

---

# 🏗️ Project Architecture

The system follows an Agentic AI workflow where different nodes handle specialized tasks:

1. **Router Node** → Decides whether research is required.
2. **Planner Node** → Creates the blog structure and tasks.
3. **Research Node** → Collects external evidence and references.
4. **Section Writers** → Generate individual blog sections.
5. **Reducer Node** → Merges all generated sections.
6. **Image Planner** → Creates image placeholders and prompts.
7. **Final Output Node** → Produces the complete markdown blog.

---

# 🛠️ Tech Stack

## Languages & Frameworks

* Python
* LangGraph
* LangChain
* Pydantic
* Jupyter Notebook

## AI & LLMs

* Groq API
* Llama 3.3 70B Versatile
* Google GenAI

## Utilities

* dotenv
* requests
* pathlib
* typing

---

# 📂 Project Structure

```bash
blog_agent_images.ipynb   # Main notebook containing the complete workflow
.env                       # API keys and environment variables
README.md                  # Project documentation
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

## 2. Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/Mac

```bash
source venv/bin/activate
```

---

# 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install langgraph langchain langchain-groq google-genai pydantic python-dotenv requests
```

---

# 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
```

---

# ▶️ Running the Project

Open the notebook:

```bash
jupyter notebook
```

Run:

```bash
blog_agent_images.ipynb
```

---

# 🧠 Core Components

## 📌 Task Schema

Defines blog writing tasks such as:

* Section title
* Description
* Goals
* Word limits
* Bullet coverage points
* Research requirements

## 📌 Plan Schema

Handles:

* Blog title
* Audience
* Tone
* Blog type
* Constraints
* Task breakdown

## 📌 Image Planning

The project automatically:

* Creates image placeholders
* Generates image prompts
* Assigns captions and alt text
* Supports multiple image sizes

---

# 🖼️ Example Workflow

```text
User Topic Input
       ↓
Router Decision
       ↓
Research + Planning
       ↓
Section Generation
       ↓
Image Planning
       ↓
Markdown Assembly
       ↓
Final AI Blog Output
```

---

# 📊 Use Cases

* AI Blog Writing
* Research Blog Automation
* Technical Content Generation
* Educational Blog Creation
* SEO Content Pipelines
* AI-assisted Documentation

---

# 🔮 Future Improvements

* Streamlit UI integration
* Real-time web search
* AI image generation integration
* Export to PDF/DOCX
* Multi-language support
* RAG-based contextual memory
* Deployment on cloud platforms

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

---

# 📜 License

This project is open-source and available under the MIT License.

---

# Demo Video

https://github.com/user-attachments/assets/c42696c4-f6ce-409f-ae8d-190b39b5eef2

---
# 👨‍💻 Author

**Muhammad Arsal**
AI & Machine Learning Enthusiast
Focused on Agentic AI, Automation, and Intelligent Systems.

