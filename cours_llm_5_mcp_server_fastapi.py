import os

import requests
import uvicorn
import wikipediaapi
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

load_dotenv(override=True)

# -----------------------------------------------------------------------------
# Initialisation
# -----------------------------------------------------------------------------
# Crée une instance de l'application FastAPI
app = FastAPI(
    title="Serveur d'Outils (MCP)",
    description="Un serveur qui expose des outils comme Wikipedia et la météo pour être utilisé par un agent LLM.",
    version="1.0.0",
)

# Initialise l'API Wikipedia
wiki_api = wikipediaapi.Wikipedia(
    # AJOUT REQUIS : Le user_agent est maintenant obligatoire.
    user_agent="MonAppDeCoursLLM/1.0 (contact@example.com)",
    language="fr",
    extract_format=wikipediaapi.ExtractFormat.WIKI,
)

# Récupère la clé API Météo depuis les variables d'environnement
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
# -----------------------------------------------------------------------------
# Définition des points d'API (nos outils)
# -----------------------------------------------------------------------------


@app.get("/tools/wikipedia", summary="Recherche un sujet sur Wikipedia")
async def search_wikipedia(query: str):
    """
    Recherche un sujet sur Wikipedia et retourne un court résumé.
    - **query**: Le sujet à rechercher (ex: "Intelligence_artificielle").
    """
    if not query:
        raise HTTPException(status_code=400, detail="Le paramètre 'query' est manquant")

    print(f" [MCP Server] Reçu une demande pour l'outil Wikipedia. Sujet : {query}")
    page = wiki_api.page(query)

    if not page.exists():
        raise HTTPException(
            status_code=404,
            detail=f"La page '{query}' n'a pas été trouvée sur Wikipedia.",
        )

    summary = page.summary[:300]
    return {"tool": "wikipedia", "query": query, "summary": f"{summary}..."}


@app.get("/tools/weather", summary="Obtient la météo actuelle pour une ville")
async def get_weather(city: str):
    """
    Obtient la météo actuelle pour une ville donnée.
    """
    if not city:
        raise HTTPException(status_code=400, detail="Le paramètre 'city' est manquant")

    if not OPENWEATHER_API_KEY:
        print(
            "[MCP Server] ERREUR : La variable d'environnement OPENWEATHER_API_KEY n'est pas définie."
        )
        raise HTTPException(
            status_code=500,
            detail="La clé API OpenWeatherMap n'est pas configurée côté serveur.",
        )

    print(f" [MCP Server] Reçu une demande pour l'outil Météo. Ville : {city}")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric&lang=fr"

    try:
        response = requests.get(url)

        # --- AMÉLIORATION DU DIAGNOSTIC ---
        if response.status_code == 401:
            print(
                "[MCP Server] ERREUR 401 : La clé API OpenWeatherMap est invalide ou non autorisée."
            )
            raise HTTPException(
                status_code=500,
                detail="La clé API pour le service météo semble être invalide.",
            )

        response.raise_for_status()  # Gère les autres erreurs HTTP (4xx, 5xx)

        data = response.json()
        description = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        return {
            "tool": "weather",
            "city": city,
            "temperature": temp,
            "description": description,
        }
    except requests.exceptions.RequestException as e:
        print(
            f"[MCP Server] ERREUR : Impossible de contacter l'API OpenWeatherMap. Détails : {e}"
        )
        raise HTTPException(
            status_code=500,
            detail="Erreur de communication avec le service météo externe.",
        )


# --- NOUVEL OUTIL : BRAVE SEARCH ---
@app.get("/tools/search", summary="Effectue une recherche sur le web avec Brave Search")
async def brave_search(query: str):
    """
    Recherche sur le web et retourne les 3 premiers résultats.
    - **query**: La question ou le terme à rechercher.
    """
    if not query:
        raise HTTPException(status_code=400, detail="Le paramètre 'query' est manquant")

    if not BRAVE_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="La clé API Brave Search n'est pas configurée côté serveur.",
        )

    print(
        f" [MCP Server] Reçu une demande pour l'outil Brave Search. Requête : {query}"
    )

    headers = {"X-Subscription-Token": BRAVE_API_KEY}
    url = f"https://api.search.brave.com/res/v1/web/search?q={query.replace(' ', '+')}&count=3"

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        # On formate les résultats pour être plus utiles au LLM
        results = []
        if "web" in data and "results" in data["web"]:
            for item in data["web"]["results"]:
                results.append(
                    {
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "snippet": item.get("description"),
                    }
                )

        return {"tool": "search", "query": query, "results": results}
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur de communication avec l'API Brave Search: {e}",
        )


# -----------------------------------------------------------------------------
# Point d'entrée pour lancer le serveur
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(
        "[MCP Server] Démarrage du serveur d'outils (FastAPI) sur http://127.0.0.1:5001"
    )
    # On utilise uvicorn pour lancer l'application FastAPI
    uvicorn.run(app, host="127.0.0.1", port=5001)