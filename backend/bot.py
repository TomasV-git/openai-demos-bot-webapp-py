# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import asyncio
import os
import random
import re
import requests
import json
import openai

from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import Activity, ActivityTypes, ChannelAccount
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader


from azure.search.documents import SearchClient
from azure.storage.blob import (
    AccountSasPermissions,
    BlobServiceClient,
    ResourceTypes,
    generate_account_sas,
)
from approaches.chatreadretrieveread import ChatReadRetrieveReadApproach
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient

str_to_bool = {'true': True, 'false': False}
# Replace these with your own values, either in environment variables or directly here
AZURE_BLOB_STORAGE_ACCOUNT = (
    os.environ.get("AZURE_BLOB_STORAGE_ACCOUNT") or "mystorageaccount"
)
AZURE_BLOB_STORAGE_ENDPOINT = os.environ.get("BLOB_STORAGE_ACCOUNT_ENDPOINT") 
AZURE_BLOB_STORAGE_KEY = os.environ.get("AZURE_BLOB_STORAGE_KEY")
AZURE_BLOB_STORAGE_CONTAINER = (
    os.environ.get("AZURE_BLOB_STORAGE_CONTAINER") or "content"
)
AZURE_SEARCH_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE") or "gptkb"
AZURE_SEARCH_SERVICE_ENDPOINT = os.environ.get("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_SERVICE_KEY = os.environ.get("AZURE_SEARCH_SERVICE_KEY")
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX") or "gptkbindex"
AZURE_OPENAI_SERVICE = os.environ.get("AZURE_OPENAI_SERVICE") or "myopenai"
AZURE_OPENAI_RESOURCE_GROUP = os.environ.get("AZURE_OPENAI_RESOURCE_GROUP") or ""
AZURE_OPENAI_CHATGPT_DEPLOYMENT = (
    os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT") or "gpt-35-turbo-16k"
)
AZURE_OPENAI_CHATGPT_MODEL_NAME = ( os.environ.get("AZURE_OPENAI_CHATGPT_MODEL_NAME") or "")
AZURE_OPENAI_CHATGPT_MODEL_VERSION = ( os.environ.get("AZURE_OPENAI_CHATGPT_MODEL_VERSION") or "")
USE_AZURE_OPENAI_EMBEDDINGS = str_to_bool.get(os.environ.get("USE_AZURE_OPENAI_EMBEDDINGS").lower()) or False
EMBEDDING_DEPLOYMENT_NAME = ( os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME") or "")
AZURE_OPENAI_EMBEDDINGS_MODEL_NAME = ( os.environ.get("AZURE_OPENAI_EMBEDDINGS_MODEL_NAME") or "")
AZURE_OPENAI_EMBEDDINGS_VERSION = ( os.environ.get("AZURE_OPENAI_EMBEDDINGS_VERSION") or "")

AZURE_OPENAI_SERVICE_KEY = os.environ.get("AZURE_OPENAI_SERVICE_KEY")
AZURE_SUBSCRIPTION_ID = os.environ.get("SUBSCRIPTION_ID")
IS_GOV_CLOUD_DEPLOYMENT = str_to_bool.get(os.environ.get("IS_GOV_CLOUD_DEPLOYMENT").lower()) or False
CHAT_WARNING_BANNER_TEXT = os.environ.get("CHAT_WARNING_BANNER_TEXT") or ""
APPLICATION_TITLE = os.environ.get("APPLICATION_TITLE") or "Information Assistant, built with Azure OpenAI"


KB_FIELDS_CONTENT = os.environ.get("KB_FIELDS_CONTENT") or "content"
KB_FIELDS_PAGENUMBER = os.environ.get("KB_FIELDS_PAGENUMBER") or "pages"
KB_FIELDS_SOURCEFILE = os.environ.get("KB_FIELDS_SOURCEFILE") or "file_uri"
KB_FIELDS_CHUNKFILE = os.environ.get("KB_FIELDS_CHUNKFILE") or "chunk_file"

COSMOSDB_URL = os.environ.get("COSMOSDB_URL")
COSMODB_KEY = os.environ.get("COSMOSDB_KEY")
COSMOSDB_LOG_DATABASE_NAME = os.environ.get("COSMOSDB_LOG_DATABASE_NAME") or "statusdb"
COSMOSDB_LOG_CONTAINER_NAME = os.environ.get("COSMOSDB_LOG_CONTAINER_NAME") or "statuscontainer"
COSMOSDB_TAGS_DATABASE_NAME = os.environ.get("COSMOSDB_TAGS_DATABASE_NAME") or "tagsdb"
COSMOSDB_TAGS_CONTAINER_NAME = os.environ.get("COSMOSDB_TAGS_CONTAINER_NAME") or "tagscontainer"

QUERY_TERM_LANGUAGE = os.environ.get("QUERY_TERM_LANGUAGE") or "English"

TARGET_EMBEDDING_MODEL = os.environ.get("TARGET_EMBEDDINGS_MODEL") or "BAAI/bge-small-en-v1.5"
ENRICHMENT_APPSERVICE_NAME = os.environ.get("ENRICHMENT_APPSERVICE_NAME") or "enrichment"

# Use the current user identity to authenticate with Azure OpenAI, Cognitive Search and Blob Storage (no secrets needed,
# just use 'az login' locally, and managed identity when deployed on Azure). If you need to use keys, use separate AzureKeyCredential instances with the
# keys for each service
# If you encounter a blocking error during a DefaultAzureCredntial resolution, you can exclude the problematic credential by using a parameter (ex. exclude_shared_token_cache_credential=True)
azure_credential = DefaultAzureCredential()
azure_search_key_credential = AzureKeyCredential(AZURE_SEARCH_SERVICE_KEY)

# # Used by the OpenAI SDK
# openai.api_type = "azure"
# openai.api_base = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com"
# openai.api_version = "2023-06-01-preview"

# # Setup StatusLog to allow access to CosmosDB for logging
# statusLog = StatusLog(
#     COSMOSDB_URL, COSMODB_KEY, COSMOSDB_LOG_DATABASE_NAME, COSMOSDB_LOG_CONTAINER_NAME
# )
# tagsHelper = TagsHelper(
#     COSMOSDB_URL, COSMODB_KEY, COSMOSDB_TAGS_DATABASE_NAME, COSMOSDB_TAGS_CONTAINER_NAME
# )

# Comment these two lines out if using keys, set your API key in the OPENAI_API_KEY environment variable instead
openai.api_type = "azure_ad"
# openai_token = azure_credential.get_token("https://cognitiveservices.azure.com/.default")
openai.api_key = AZURE_OPENAI_SERVICE_KEY
openai.api_type = "azure"
openai.api_base = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com"
openai.api_version = "2023-06-01-preview"
openai.api_key = AZURE_OPENAI_SERVICE_KEY

# Set up clients for Cognitive Search and Storage
search_client = SearchClient(
    endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=azure_search_key_credential,
)
blob_client = BlobServiceClient(
    account_url=AZURE_BLOB_STORAGE_ENDPOINT,
    credential=AZURE_BLOB_STORAGE_KEY,
)
blob_container = blob_client.get_container_client(AZURE_BLOB_STORAGE_CONTAINER)

model_name = ''
model_version = ''

if (IS_GOV_CLOUD_DEPLOYMENT):
    model_name = AZURE_OPENAI_CHATGPT_MODEL_NAME
    model_version = AZURE_OPENAI_CHATGPT_MODEL_VERSION
    embedding_model_name = AZURE_OPENAI_EMBEDDINGS_MODEL_NAME
    embedding_model_version = AZURE_OPENAI_EMBEDDINGS_VERSION
else:
    # Set up OpenAI management client
    openai_mgmt_client = CognitiveServicesManagementClient(
        credential=azure_credential,
        subscription_id=AZURE_SUBSCRIPTION_ID)

    deployment = openai_mgmt_client.deployments.get(
        resource_group_name=AZURE_OPENAI_RESOURCE_GROUP,
        account_name=AZURE_OPENAI_SERVICE,
        deployment_name=AZURE_OPENAI_CHATGPT_DEPLOYMENT)

    model_name = deployment.properties.model.name
    model_version = deployment.properties.model.version

    if USE_AZURE_OPENAI_EMBEDDINGS:
        embedding_deployment = openai_mgmt_client.deployments.get(
            resource_group_name=AZURE_OPENAI_RESOURCE_GROUP,
            account_name=AZURE_OPENAI_SERVICE,
            deployment_name=EMBEDDING_DEPLOYMENT_NAME)

        embedding_model_name = embedding_deployment.properties.model.name
        embedding_model_version = embedding_deployment.properties.model.version
    else:
        embedding_model_name = ""
        embedding_model_version = ""

chat_approaches = {
    "rrr": ChatReadRetrieveReadApproach(
        search_client,
        AZURE_OPENAI_SERVICE,
        AZURE_OPENAI_SERVICE_KEY,
        AZURE_OPENAI_CHATGPT_DEPLOYMENT,
        KB_FIELDS_SOURCEFILE,
        KB_FIELDS_CONTENT,
        KB_FIELDS_PAGENUMBER,
        KB_FIELDS_CHUNKFILE,
        AZURE_BLOB_STORAGE_CONTAINER,
        blob_client,
        QUERY_TERM_LANGUAGE,
        model_name,
        model_version,
        IS_GOV_CLOUD_DEPLOYMENT,
        TARGET_EMBEDDING_MODEL,
        ENRICHMENT_APPSERVICE_NAME
    )
}

from botbuilder.schema import (
    ConversationAccount,
    Attachment,
)
from botbuilder.schema.teams import (
    FileDownloadInfo,
    FileConsentCard,
    FileConsentCardResponse,
    FileInfoCard,
)
from botbuilder.schema.teams.additional_properties import ContentType


from langchain.memory import (ConversationBufferMemory,
                              ConversationBufferWindowMemory,
                              CosmosDBChatMessageHistory)
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               MessagesPlaceholder,
                               SystemMessagePromptTemplate)

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document

from prompts import (CUSTOM_CHATBOT_PREFIX, WELCOME_MESSAGE)
from prompts import COMBINE_QUESTION_PROMPT, COMBINE_PROMPT


from utils import get_search_results

from typing import Any, Dict, List, Optional, Awaitable, Callable, Tuple, Type, Union
from collections import OrderedDict
import uuid

import markdownify

#####################


# Env variables needed by langchain
os.environ["OPENAI_API_BASE"] = os.environ.get("AZURE_OPENAI_ENDPOINT")
os.environ["OPENAI_API_KEY"] = os.environ.get("AZURE_OPENAI_API_KEY")
os.environ["OPENAI_API_VERSION"] = os.environ.get("AZURE_OPENAI_API_VERSION")
os.environ["OPENAI_API_TYPE"] = "azure"

      
# Bot Class
class MyBot(ActivityHandler):
    memory = None
    prompt = ChatPromptTemplate(
                messages=[
                    SystemMessagePromptTemplate.from_template(
                        CUSTOM_CHATBOT_PREFIX
                    ),
                    # The `variable_name` here is what must align with memory
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template("{question}")
                ]
            )
    
    memory_dict = {}
    memory_cleared_messages = [
        "Historie byla smazána.",
        "Historie byla vynulována.",
        "Historie byla obnovena na výchozí hodnoty.",
        "Došlo k resetování historie.",
        "Historie byla resetována na počáteční stav."
    ]

    # allowed content types for file upload
    ALLOWED_CONTENT_TYPES = [
        "text/plain",  # ".txt"
        "text/markdown",  # ".md"
    ]

    # FAISS db 
    db = None


    def __init__(self):
        self.model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME") 
        self.llm = AzureChatOpenAI(deployment_name=self.model_name, temperature=0.7, max_tokens=600)
        self.embedding_model = "text-embedding-ada-002" # TODO: make this configurable

    def get_search_results(self, query: str, indexes: list,
                        k: int = 5,
                        reranker_threshold: int = 1,
                        sas_token: str = "",
                        vector_search: bool = False,
                        similarity_k: int = 3,
                        query_vector: list = None) -> List[dict]:
        """
        This function is responsible for getting search results from Azure Search.

        Parameters:
        query (str): The search query.
        indexes (list): The list of indexes to search in.
        k (int, optional): The number of top results to return. Defaults to 5.
        reranker_threshold (int, optional): The threshold for reranking the results. Defaults to 1.
        sas_token (str, optional): The SAS token for accessing Azure resources. Defaults to "".
        vector_search (bool, optional): Whether to perform vector search or not. Defaults to False.
        similarity_k (int, optional): The number of similar results to return in case of vector search. Defaults to 3.
        query_vector (list, optional): The vector representation of the query for vector search. Defaults to None.

        Returns:
        ordered_content (dict): The ordered search results.

        Raises:
        Exception: If there is an error in the search request.
       
        Example:
        >>> get_search_results()
        Expected output
        """

        headers = {'Content-Type': 'application/json','api-key': os.environ["AZURE_SEARCH_KEY"]}
        params = {'api-version': os.environ['AZURE_SEARCH_API_VERSION']}

        agg_search_results = dict()

        for index in indexes:
            search_payload = {
                "search": query,
                "queryType": "semantic",
                "semanticConfiguration": "default",
                "count": "true",
                # "speller": "lexicon",
                # "queryLanguage": "cs-CZ",
                "captions": "extractive",
                "answers": "extractive",
                "top": k
            }
            if vector_search:
                # search_payload["vectors"]= [{"value": query_vector, "fields": "chunkVector","k": k}]
                # search_payload["select"]= "id, title, chunk, name, location"
                search_payload["vectors"]= [{"value": query_vector, "fields": "contentVector","k": k}]
                search_payload["select"]= "id, title, content, file_name, file_uri"
            else:
                search_payload["select"]= "id, title, chunks, language, name, location, vectorized" # TODO Change according to the index schema
            

            resp = requests.post(os.environ['AZURE_SEARCH_ENDPOINT'] + "/indexes/" + index + "/docs/search",
                            data=json.dumps(search_payload), headers=headers, params=params, timeout=30)

            search_results = resp.json()
            agg_search_results[index] = search_results

        content = dict()
        ordered_content = OrderedDict()

        for index,search_results in agg_search_results.items():
            for result in search_results['value']:
                if result['@search.rerankerScore'] > reranker_threshold: # Show results that are at least N% of the max possible score=4
                    content[result['id']]={
                                            "title": result['title'], 
                                            "name": result['file_name'], 
                                            "location": result['file_uri'] + sas_token if result['file_uri'] else "",
                                            "caption": result['@search.captions'][0]['text'],
                                            "index": index
                                        }
                    if vector_search:
                        content[result['id']]["chunk"]= result['content']
                        content[result['id']]["score"]= result['@search.score'] # Uses the Hybrid RRF score
                
                    else:
                        content[result['id']]["chunks"]= result['chunks']
                        content[result['id']]["language"]= result['language']
                        content[result['id']]["score"]= result['@search.rerankerScore'] # Uses the reranker score
                        content[result['id']]["vectorized"]= result['vectorized']
                    
        # After results have been filtered, sort and add the top k to the ordered_content
        if vector_search:
            topk = similarity_k
        else:
            topk = k*len(indexes)
            
        count = 0  # To keep track of the number of results added
        for id in sorted(content, key=lambda x: content[x]["score"], reverse=True):
            ordered_content[id] = content[id]
            count += 1
            if count >= topk:  # Stop after adding 5 results
                break

        return ordered_content

    # format the response (convert html to markdown)
    def format_response(self, response):
        # return re.sub(r"(\n\s*)+\n+", "\n\n", response).strip()

        # convert html tags to markdown
        response = markdownify.markdownify(response, heading_style="ATX")
    
        return response.strip()
    
    # ask GPT    
    def ask_gpt(self, session_id):
        chatgpt_chain = LLMChain(
                                llm=self.llm,
                                prompt=self.prompt,
                                verbose=False,
                                memory=self.memory_dict[session_id]
                                )
        answer = chatgpt_chain.run(self.QUESTION)                        
        return answer
    
    # ask GPT with sources in file
    def ask_gpt_with_sources(self, session_id):
        # remove the /file prefix
        QUESTION = self.QUESTION.strip()
        
        # query = "What did the president say about Ketanji Brown Jackson"
        # docs = self.db.similarity_search_with_score(query)
        # vector_indexes = [self.generate_index()]
        vector_indexes = ["vector-index"]

        embedder = OpenAIEmbeddings(deployment=self.embedding_model, chunk_size=1)

        ordered_results = self.get_search_results(QUESTION, vector_indexes, 
                                                k=10,
                                                reranker_threshold=0.1, #1
                                                vector_search=True, 
                                                similarity_k=10,
                                                query_vector = embedder.embed_query(QUESTION)
                                                # query_vector= []
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
            chain = load_qa_with_sources_chain(self.llm, chain_type=chain_type, 
                                            prompt=COMBINE_PROMPT)
        # elif chain_type == "map_reduce":
        #     chain = load_qa_with_sources_chain(self.llm, chain_type=chain_type, 
        #                                     question_prompt=COMBINE_QUESTION_PROMPT,
        #                                     combine_prompt=COMBINE_PROMPT,
        #                                     return_intermediate_steps=True)


        response = chain({"input_documents": top_docs, "question": QUESTION, "language": "Czech"}) # TODO: make language configurable
        text_output = self.format_response(response['output_text'])
        return text_output
    
    def ask_gpt_with_sources_new(self, session_id):

        import openai, os, requests

        QUESTION = self.QUESTION.strip()

        openai.api_type = "azure"
        # Azure OpenAI on your own data is only supported by the 2023-08-01-preview API version
        openai.api_version = "2023-08-01-preview"

        # Env variables needed by langchain
        # os.environ["OPENAI_API_BASE"] = os.environ.get("AZURE_OPENAI_ENDPOINT")
        # os.environ["OPENAI_API_KEY"] = os.environ.get("AZURE_OPENAI_API_KEY")
        # os.environ["OPENAI_API_VERSION"] = os.environ.get("AZURE_OPENAI_API_VERSION")
        # os.environ["OPENAI_API_TYPE"] = "azure"

        # Azure OpenAI setup
        openai.api_base = "https://openaimmaswe.openai.azure.com/" # Add your endpoint here
        openai.api_key = os.getenv("OPENAI_API_KEY") # Add your OpenAI API key here
        deployment_id = "gpt-4-turbo" # Add your deployment ID here

        # Azure AI Search setup
        search_endpoint = "https://infoasst-search-hqpv5.search.windows.net"; # Add your Azure AI Search endpoint here
        search_key = "VNsbFBp8TvEZziPQEZHcb0YV3LydpEwLVm9XRg5iIDAzSeAEO3gC"; # Add your Azure AI Search admin key here
        search_index_name = "vector-index"; # Add your Azure AI Search index name here

        def setup_byod(deployment_id: str) -> None:
            """Sets up the OpenAI Python SDK to use your own data for the chat endpoint.

            :param deployment_id: The deployment ID for the model to use with your own data.

            To remove this configuration, simply set openai.requestssession to None.
            """

            class BringYourOwnDataAdapter(requests.adapters.HTTPAdapter):

                def send(self, request, **kwargs):
                    request.url = f"{openai.api_base}/openai/deployments/{deployment_id}/extensions/chat/completions?api-version={openai.api_version}"
                    return super().send(request, **kwargs)

            session = requests.Session()

            # Mount a custom adapter which will use the extensions endpoint for any call using the given `deployment_id`
            session.mount(
                prefix=f"{openai.api_base}/openai/deployments/{deployment_id}",
                adapter=BringYourOwnDataAdapter()
            )

            openai.requestssession = session

        setup_byod(deployment_id)


        message_text = [{"role": "user", "content": QUESTION}]

        completion = openai.ChatCompletion.create(
            messages=message_text,
            deployment_id=deployment_id,
            dataSources=[  # camelCase is intentional, as this is the format the API expects
                {
                    "type": "AzureCognitiveSearch",
                    "parameters": {
                        "endpoint": search_endpoint,
                        "key": search_key,
                        "indexName": search_index_name,
                    }
                }
            ]
        )
        print(completion)
        text_output = self.format_response(completion.choices[0]["message"]["content"])
        # text_output = self.format_response(response['output_text'])
        return text_output


    # Function to show welcome message to new users
    async def on_members_added_activity(self, members_added: ChannelAccount, turn_context: TurnContext):
        for member_added in members_added:
            if member_added.id != turn_context.activity.recipient.id:
                await turn_context.send_activity(WELCOME_MESSAGE + "\n\n" + self.model_name)
    
    
    # See https://aka.ms/about-bot-activity-message to learn more about the message and other activity types.
    async def on_message_activity(self, turn_context: TurnContext):
             
        # Extract info from TurnContext - You can change this to whatever , this is just one option 
        session_id = turn_context.activity.conversation.id
        user_id = turn_context.activity.from_property.id + "-" + turn_context.activity.channel_id

        self.QUESTION = turn_context.activity.text

        if session_id not in self.memory_dict:
            self.memory_dict[session_id] = ConversationBufferWindowMemory(memory_key="chat_history",input_key="question", return_messages=True, k=3)

        if (self.QUESTION == "/reset"):
            # self.memory.clear()
            self.memory_dict[session_id].clear()
            self.db = None # reset the db
            # randomly pick one of the memory_cleared_messages
            await turn_context.send_activity(random.choice(self.memory_cleared_messages))
            # await turn_context.send_activity("Memory cleared")
        elif (self.QUESTION == "/help"):
            await turn_context.send_activity(WELCOME_MESSAGE + "\n\n" + self.model_name)
        else:
            # # check if there is uploaded file and initialized FAISS db
            # if (self.db):
            #     answer = self.ask_gpt_with_sources(session_id)
            # else:
            #     answer = self.ask_gpt(session_id)
            
            # answer = self.ask_gpt_with_sources_new(session_id)
            
            
            # answer = self.ask_gpt_with_sources(session_id)
            
            approach = "rrr"
            try:
                impl = chat_approaches.get(approach)
                if not impl:
                    return json.loads({"error": "unknown approach"}), 400
                # r = impl.run(request.json["history"], request.json.get("overrides") or {})
                self.memory_dict[session_id] = [{"user":self.QUESTION}]
                r = impl.run(self.memory_dict[session_id], {})

                # return jsonify(r)
                # To fix citation bug,below code is added.aparmar
                
                answer_full  = {
                            "data_points": r["data_points"],
                            "answer": r["answer"],
                            "thoughts": r["thoughts"],
                            "citation_lookup": r["citation_lookup"],
                        }
                answer = answer_full["answer"]
            except Exception as ex:
                print(ex)
                answer = "ERORR" + str(ex)

            await turn_context.send_activity(answer)





