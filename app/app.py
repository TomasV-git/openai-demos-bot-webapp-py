import streamlit as st
import os
import json
from dotenv import load_dotenv
load_dotenv("../credentials.env")


os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_API_BASE"] = os.environ['AZURE_OPENAI_ENDPOINT']
os.environ["OPENAI_API_KEY"] = os.environ['AZURE_OPENAI_API_KEY']
os.environ["OPENAI_MODEL"] = os.environ['AZURE_OPENAI_MODEL_NAME']
MODEL = os.environ['AZURE_OPENAI_MODEL_NAME']

from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI

from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               MessagesPlaceholder,
                               SystemMessagePromptTemplate)

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document

from prompts import (CUSTOM_CHATBOT_PREFIX, WELCOME_MESSAGE)
from prompts import COMBINE_QUESTION_PROMPT, COMBINE_PROMPT
import uuid

def get_prompt(template):
    PROMPT = ChatPromptTemplate(
                    messages=[
                        SystemMessagePromptTemplate.from_template(
                            template
                        ),
                        # The `variable_name` here is what must align with memory
                        MessagesPlaceholder(variable_name="chat_history"),
                        HumanMessagePromptTemplate.from_template("{question}")
                    ]
                )
    return PROMPT

from chat_utils import get_search_results, process_file, generate_index, generate_doc_id, format_response

def ask_gpt(QUESTION, session_id):
    chatgpt_chain = LLMChain(
                            llm=st.session_state.llm,
                            prompt=get_prompt(st.session_state.SYSTEM_PROMPT),
                            verbose=False,
                            memory=st.session_state.memory_dict[session_id]
                            )
    answer = chatgpt_chain.run(QUESTION)                        
    return answer

# ask GPT with sources in file
def ask_gpt_with_sources(QUESTION, session_id):
    # remove the /file prefix
    # QUESTION = self.QUESTION[5:].strip()
    
    # query = "What did the president say about Ketanji Brown Jackson"
    # docs = self.db.similarity_search_with_score(query)
    vector_indexes = [generate_index()]

    ordered_results = get_search_results(QUESTION, vector_indexes, 
                                            k=10,
                                            reranker_threshold=0.1, #1
                                            vector_search=True, 
                                            similarity_k=10,
                                            #query_vector = embedder.embed_query(QUESTION)
                                            query_vector= []
                                            )
    # COMPLETION_TOKENS = 1000
    # llm = AzureChatOpenAI(deployment_name=MODEL, temperature=0.5, max_tokens=COMPLETION_TOKENS)

    top_docs = []
    for key,value in ordered_results.items():
        location = value["location"] if value["location"] is not None else ""
        # top_docs.append(Document(page_content=value["chunk"], metadata={"source": location+os.environ['BLOB_SAS_TOKEN']}))
        top_docs.append(Document(page_content=value["chunk"], metadata={"source": value["name"]}))
            
    # print("Number of chunks:",len(top_docs))

    chain_type = "stuff"
    
    if chain_type == "stuff":
        chain = load_qa_with_sources_chain(st.session_state.llm, chain_type=chain_type, 
                                        prompt=COMBINE_PROMPT)
    elif chain_type == "map_reduce":
        chain = load_qa_with_sources_chain(st.session_state.llm, chain_type=chain_type, 
                                        question_prompt=COMBINE_QUESTION_PROMPT,
                                        combine_prompt=COMBINE_PROMPT,
                                        return_intermediate_steps=True)


    response = chain({"input_documents": top_docs, "question": QUESTION, "language": "English"})
    text_output = format_response(response['output_text'])
    return text_output

# Initialize chat history
if "llm" not in st.session_state:
    st.session_state.llm = AzureChatOpenAI(deployment_name=MODEL, temperature=0.7, max_tokens=600)

if "memory_dict" not in st.session_state:
    st.session_state.memory_dict = {}

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
    st.session_state.memory_dict[st.session_state.session_id] = ConversationBufferWindowMemory(memory_key="chat_history",input_key="question", return_messages=True, k=3)

if "db" not in st.session_state:
    st.session_state.db = None

if "info" not in st.session_state:
    st.session_state.info = None
if "SYSTEM_PROMPT" not in st.session_state:
    st.session_state.SYSTEM_PROMPT = CUSTOM_CHATBOT_PREFIX
#################################################################################
# App elements

# Sidebar elements
st.sidebar.title("Menu")
with st.sidebar:
    st.markdown(WELCOME_MESSAGE + MODEL)

    # with st.expander("Settings"):
    # add upload button
    uploaded_file = st.file_uploader("Upload a file", type=["txt", "md"])
    if uploaded_file is not None:
        # store the uploaded file on disk
        msg = process_file(uploaded_file)
        st.warning(msg)
        st.session_state.info = msg
    
    st.text_area("Enter your question here", key="system_custom_prompt", value=CUSTOM_CHATBOT_PREFIX)

    # create a save button
    if st.button("Save"):
        # save the text from the text_area to CUSTOM_CHATBOT_PREFIX
        # CUSTOM_CHATBOT_PREFIX = st.session_state.SYSTEM_PROMPT
        st.session_state.SYSTEM_PROMPT = st.session_state.system_custom_prompt
        # delete memory
        st.session_state.memory_dict[st.session_state.session_id].clear()

st.title("Chat")
if st.session_state.info is not None:
    st.info(st.session_state.info)
    st.session_state.info = None

st.caption(f"session: {st.session_state.session_id}")





with st.container():
    # display messages from memory
    memory = st.session_state.memory_dict[st.session_state.session_id].load_memory_variables({})
    for message in memory["chat_history"]:
        # if message typ
        with st.chat_message(message.type):
            st.markdown(message.content)



    # Accept user input
    if prompt := st.chat_input("What is up?"):
        if prompt:
            if (st.session_state.db is not None):
                output = ask_gpt_with_sources(prompt, st.session_state.session_id)
            else:
                # Get response from GPT
                output = ask_gpt(prompt, st.session_state.session_id)            
            
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                st.markdown(output)