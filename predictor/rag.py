import requests
import json
import os
from dotenv import load_dotenv
from langchain.schema import HumanMessage, AIMessage
from langchain_google_genai  import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from .vector_store import VectorStore


load_dotenv()

class Rag:
    def __init__(self):
        print("Initializing RAG...")
        self.ollama_url = os.getenv("OLLAMA_API_URL")  # e.g., http://localhost:11434/api/chat
        self.gemini_api_key = os.getenv("GOOGLE_API_KEY")
        self.vector_store= VectorStore()
        print("RAG initialized")
    def _format_context(self, result_json):
        return json.dumps(result_json, indent=2)

    def _build_initial_prompt(self, context):
        return f"""
You are a compassionate and knowledgeable dermatologist assistant.
You have analyzed a patient's face and detected various dermatological conditions. Below is the segmentation result across different face angles:
{context}

Based on this, introduce yourself to the user, briefly explain what you observed, give general advice and ask if the patient wants more information or has a followup question.
"""



    def _get_chat_model(self):
        # Add ollama here later
        return  ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
        )
    
    def start_converstaion(self, result_json) -> str:
        print("Starting conversation with the model...")
        context = self._format_context(result_json)
        initial_prompt = self._build_initial_prompt(context)
        llm = self.get_chat_model()
        response = llm(initial_prompt)
        return response.content
    
    def _search_vector_store(self, query):
        return self.vector_store.search(query, namespace="general")
    
    def _build_conversation_prompt(self, initail_res,history,query):
        prompt = PromptTemplate(
            template=""""
        You are a compassionate and knowledgeable dermatologist assistant.
You have analyzed a patient's face and detected various dermatological conditions. Below is the segmentation result across different face angles:
{initail_res}

You and the user are having a conversation about the dermatological conditions. Here is a history of the conversation so far:
{history}

The user has asked the following question:
{query}

You can use the following context to answer the user's question:
{context}
        """,
        input_variables=["initail_res", "history", "query","context"],
        )
        context = self._search_vector_store(query)
        prompt = prompt.format(
            initail_res=initail_res,
            history=history,
            query=query,
            context=context
        )
        return prompt
    def followup_conversation(self,initail_res, history,query):
        initail_res = self._format_context(initail_res)
        prompt =self._build_conversation_prompt(initail_res,history,query)

        llm = self.get_chat_model()
        llm_response = llm(prompt)
        return llm_response.content
