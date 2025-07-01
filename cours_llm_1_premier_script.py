import requests
import json

# L'URL de l'API locale d'Ollama
OLLAMA_API_URL = "http://localhost:11434/api/chat"

def query_llm(prompt):
    """
    Envoie une requête au LLM via l'API d'Ollama et retourne la réponse.
    """
    try:
        # Les données à envoyer dans la requête POST
        data = {
            "model": "llama3.2:3b",  # Le modèle que nous voulons utiliser
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False  # Pour recevoir la réponse en une seule fois
        }

        # Envoi de la requête POST
        response = requests.post(OLLAMA_API_URL, json=data)
        response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP

        # Extraction et affichage de la réponse
        response_json = response.json()
        print("Réponse du LLM :", response_json['message']['content'])

    except requests.exceptions.RequestException as e:
        print(f"Erreur de connexion à l'API d'Ollama : {e}")
        print("Veuillez vous assurer qu'Ollama est en cours d'exécution.")

if __name__ == "__main__":
    user_prompt = "Explique le concept de trou noir de manière simple."
    query_llm(user_prompt)