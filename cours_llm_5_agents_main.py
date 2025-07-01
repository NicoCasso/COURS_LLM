import json

import requests

# -- CONFIGURATION --
OLLAMA_API_URL = "http://localhost:11434/api/chat"
MCP_SERVER_URL = "http://127.0.0.1:5001"  # L'adresse de notre serveur d'outils
MODEL_NAME = "llama3.2:3b"


def query_ollama_for_tool_choice(prompt: str):
    """Interroge le LLM pour qu'il choisisse un outil et retourne un JSON."""
    try:
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "format": "json",
            "stream": False,
        }
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        return json.loads(response.json()["message"]["content"])
    
    except Exception as e:
        print(f"Erreur lors de l'appel à Ollama : {e}")
        return None


def run_agent_workflow(user_question: str):
    print(f"\n--- Début du flux pour la question : '{user_question}' ---")

    # Étape 1 : Demander au LLM de choisir un outil
    prompt_for_decision = f"""
    Tu es un routeur intelligent. Ton rôle est de diriger la question de l'utilisateur vers le bon outil.
    Voici les outils disponibles et comment les appeler :
    
    1. Outil "wikipedia": Pour obtenir un résumé sur un sujet encyclopédique bien défini (personne, lieu, concept...).
       - JSON à générer : {{"tool_name": "wikipedia", "args": {{"query": "SUJET_DE_RECHERCHE"}}}}
    
    2. Outil "weather": Pour obtenir la météo d'une ville.
       - JSON à générer : {{"tool_name": "weather", "args": {{"city": "NOM_DE_LA_VILLE"}}}}
       
    3. Outil "search": Pour répondre à des questions sur l'actualité, des événements récents, ou des sujets généraux.
       - JSON à générer : {{"tool_name": "search", "args": {{"query": "QUESTION_OU_SUJET"}}}}

    Si aucun outil ne correspond, génère : {{"tool_name": "none", "args": {{}}}}

    Analyse la question de l'utilisateur et génère UNIQUEMENT l'objet JSON correspondant à l'outil à utiliser.
    Question: "{user_question}"
    """

    print("\n[Agent] 1. Envoi de la requête de décision au LLM...")
    tool_choice = query_ollama_for_tool_choice(
        prompt_for_decision
    )  # La fonction reste la même

    if not tool_choice or "tool_name" not in tool_choice:
        print("[Agent] Le LLM n'a pas retourné une décision valide.")
        return

    print(
        f"[Agent] Le LLM a décidé d'utiliser l'outil : '{tool_choice.get('tool_name')}' avec les arguments {tool_choice.get('args')}"
    )

    # Étape 2 : Exécuter l'outil via un appel HTTP au MCP Server
    tool_result = None
    tool_name = tool_choice.get("tool_name")

    # CORRECTION : On gère maintenant 'search' en plus des autres
    if tool_name in ["wikipedia", "weather", "search"]:
        endpoint = f"{MCP_SERVER_URL}/tools/{tool_name}"
        try:
            print(f" [Agent] Appel du MCP Server à l'adresse : {endpoint}")
            response = requests.get(endpoint, params=tool_choice.get("args", {}))
            response.raise_for_status()
            tool_result = response.json()
            print(f"[Agent] Résultat de l'outil reçu du MCP : {tool_result}")
            
        except requests.exceptions.RequestException as e:
            print(f"[Agent] Erreur lors de l'appel au MCP Server : {e}")
            tool_result = {"error": str(e)}

    # ... (Étape 3 : Synthèse, inchangée) ...
    if tool_result and "error" not in tool_result:
        prompt_for_synthesis = f"""
        La question initiale était : "{user_question}".
        L'outil '{tool_name}' a retourné les informations suivantes : {json.dumps(tool_result)}.
        Formule une réponse naturelle et utile à l'utilisateur en te basant sur ces informations.
        """
        print("\n[Agent] 2. Envoi de la requête de synthèse au LLM...")
        # ... la suite est identique ...
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt_for_synthesis}],
            "stream": False,
        }
        response = requests.post(OLLAMA_API_URL, json=payload)
        final_answer = response.json()["message"]["content"]
    elif tool_name == "none":
        final_answer = "Je suis désolé, je ne peux pas répondre à cette question avec les outils dont je dispose."
    else:
        final_answer = (
            "Je suis désolé, une erreur est survenue lors de l'utilisation de l'outil."
        )

    print("\n--- Fin du flux ---")
    print(f"\nRéponse finale à l'utilisateur :\n {final_answer.strip()}")


if __name__ == "__main__":
    run_agent_workflow("quel est le dernier de tom cruise ?")
    print("\n" + "=" * 50 + "\n")
    run_agent_workflow("Quel temps fait-il à Tokyo ?")