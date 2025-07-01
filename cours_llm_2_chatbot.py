import requests
import json

OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2:3b"

def chat_with_history(messages):
    """
    Envoie une conversation avec historique à l'API d'Ollama.
    """
    try:
        data = {
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False
        }
        response = requests.post(OLLAMA_API_URL, json=data)
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"Erreur de connexion : {e}")
        return None

if __name__ == "__main__":
    # Initialisation de l'historique de la conversation
    conversation_history = []
    print("🤖 Votre chatbot est prêt ! Tapez 'exit' ou 'quit' pour arrêter.")

    while True:
        user_input = input("Vous : ")
        if user_input.lower() in ["exit", "quit"]:
            print("Au revoir !")
            break

        # Ajoute le message de l'utilisateur à l'historique
        conversation_history.append({"role": "user", "content": user_input})

        # Envoie l'historique complet et récupère la réponse
        response_json = chat_with_history(conversation_history)

        if response_json:
            # Ajoute la réponse de l'assistant à l'historique
            assistant_response = response_json['message']
            conversation_history.append(assistant_response)
            print(f"IA : {assistant_response['content']}")