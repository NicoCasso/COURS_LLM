#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain import hub
import requests
import feedparser

# === OUTILS ULTRA-SIMPLES ===

@tool
def veille_ia() -> str:
    """Fait une veille complète sur l'intelligence artificielle avec actualités et tendances."""
    print("tool Veille IA")
    try:
        # Actualités
        feed = feedparser.parse("https://news.google.com/rss/search?q=intelligence+artificielle&hl=fr")
        news = [f"• {entry.title}" for entry in feed.entries[:3]]
        
        # GitHub trending
        github_api = "https://api.github.com/search/repositories?q=topic:artificial-intelligence&sort=stars&order=desc&per_page=3"
        response = requests.get(github_api, timeout=10)
        repos = [f"• {repo['full_name']} ({repo['stargazers_count']:,})" for repo in response.json().get('items', [])]
        
        result = "VEILLE INTELLIGENCE ARTIFICIELLE\n\n"
        result += "ACTUALITÉS:\n" + "\n".join(news) + "\n\n"
        result += "PROJETS POPULAIRES:\n" + "\n".join(repos)
        
        return result
    except Exception as e:
        return f"Erreur veille: {e}"

@tool  
def calculer(expression: str) -> str:
    """Effectue des calculs mathématiques."""
    print("Tool calculer")
    try:
        return f"{expression} = {eval(expression)}"
    except Exception:
        return "Expression invalide"

@tool
def info_date() -> str:
    """Donne la date et l'heure actuelles."""
    print("tool Date")
    from datetime import datetime
    return datetime.now().strftime("%d/%m/%Y à %H:%M")

# === SETUP ULTRA-RAPIDE ===

# LLM
llm = ChatOllama(model="llama3.2:latest")

# Outils
tools = [veille_ia, calculer, info_date]

# Prompt du hub (ou prompt simple)
try:
    prompt = hub.pull("hwchase17/openai-tools-agent")
except Exception:
    # Fallback si pas de connexion au hub
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tu es un assistant qui utilise des outils. Réponds de manière structurée."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

# Agent
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# === INTERFACE SIMPLE ===

def ask(question: str) -> str:
    """Interface ultra-simple - comme OpenAI"""
    return executor.invoke({"input": question})["output"]

# === EXEMPLES D'UTILISATION ===

if __name__ == "__main__":
    # Mode interactif
    print("\n Mode interactif (tapez 'exit' pour quitter):")
    while True:
        question = input("\n Votre question: ").strip()
        if question.lower() == 'exit':
            break
        if question:
            print(f"Réponse: {ask(question)}")

# === VERSION ENCORE PLUS SIMPLE ===

class SimpleAI:
    """Classe ultra-simple pour remplacer OpenAI"""
    
    def __init__(self):
        self.executor = executor
    
    def responses_create(self, input: str, tools: list = None) -> str:
        """Méthode compatible avec votre code OpenAI original"""
        return self.executor.invoke({"input": input})["output"]