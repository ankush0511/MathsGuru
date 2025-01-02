import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain,LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from dotenv import load_dotenv
from langchain.callbacks import StreamlitCallbackHandler
import os
load_dotenv()

# GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
llm = ChatGroq(model='Gemma2-9b-It')

#set up the streamlit app

st.set_page_config(page_title="Maths Problem Solver", page_icon="🧮")
st.title('Text to Maths Problem Solver')


#initializing the model
wiki=WikipediaAPIWrapper()
wiki_tool=Tool(
  name='Wikipedia',
  func=wiki.run,
  description='A tool for searching the Internet to find the various information on the topic'
)
# initialize the math tool
match_chain=LLMMathChain.from_llm(llm=llm)
calcu=Tool(
  name="Calculator",
  func=match_chain.run,
  description='A tool for answering math related questions. Only input mathematical expression need to be provided'
)

prompt="""
You are a agent tasked for solving users mathemetical question. Logically arrive at the solution and provide detaled explanation
and display it point wise for the question below
Question:{question}
Answer:
"""
prompt_template=PromptTemplate(
  input_variables=['question'],
  template=prompt
)

# combine all the tools
# maths problem tool
chain=LLMChain(llm=llm,prompt=prompt_template)

reasoning_tool=Tool(
  name="reasoning",
  func=chain.run,
  description='a tool for answering logic-based and reasoning questions'
)
# initialixing the agents

assistant_agent=initialize_agent(
  tools=[wiki_tool,calcu,reasoning_tool],
  llm=llm,
  agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
  verbose=True,
  handling_parsing_errors=True
)

if "messages" not in st.session_state:
  st.session_state["messages"] = [
    {'role': 'assistant', 'content': 'Hi, I am a maths chatbot. who can answer all your maths question'}
  ]

for msg in st.session_state.messages:
  st.chat_message(msg['role']).write(msg['content'])



question=st.text_area("Enter your question")

if st.button("fina the answer"):
  if question:
    with st.spinner("Generating Response..."):
      st.session_state.messages.append({'role':'user','content':question})
      st.chat_message('user').write(question)
  
      st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
      response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb])\
  
      st.session_state.messages.append({'role':'assistant','content':response})
      st.write('response')
      st.success(response)

  else:
    st.warning("Please enter a question")