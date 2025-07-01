# -*- coding: utf-8 -*-
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import requests


def get_embedding(text: str, model: str = "mxbai-embed-large") -> List[float]:
    """
    Génère l'embedding d'un texte via l'API Ollama

    Args:
        text: Le texte à convertir en embedding
        model: Le modèle d'embedding à utiliser

    Returns:
        Liste de nombres (vecteur) représentant le texte
    """
    url = "http://localhost:11434/api/embeddings"

    data = {"model": model, "prompt": text}

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        return result["embedding"]
    except Exception as e:
        print(f"❌ Erreur lors de la génération d'embedding: {e}")
        return []


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calcule la similarité cosinus entre deux vecteurs

    Résultat entre -1 et 1 :
    - 1.0 = Identiques
    - 0.0 = Orthogonaux (pas de relation)
    - -1.0 = Opposés

    En pratique pour les embeddings : entre 0.0 et 1.0
    """
    # Convertir en arrays numpy pour les calculs
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    # Formule de similarité cosinus
    dot_product = np.dot(v1, v2)
    magnitude1 = np.linalg.norm(v1)
    magnitude2 = np.linalg.norm(v2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    """
    Calcule la distance euclidienne entre deux vecteurs

    Plus la distance est petite, plus les vecteurs sont similaires
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.linalg.norm(v1 - v2)


def demo_embeddings():
    """
    Démonstration des embeddings et de la similarité
    """
    print("🔬 === DÉMONSTRATION DES EMBEDDINGS ===\n")

    # Phrases de test
    phrases = [
        "Je mange une pomme",
        "Je consomme un fruit rouge",  # Similaire à la première
        "Il pleut dehors",  # Différent
        "La pluie tombe",  # Similaire à la précédente
        "Python est un langage de programmation",  # Complètement différent
    ]

    print("📝 Phrases à analyser :")
    for i, phrase in enumerate(phrases):
        print(f"  {i + 1}. {phrase}")

    print("\n🔄 Génération des embeddings...")

    # Générer les embeddings
    embeddings = []
    for phrase in phrases:
        embedding = get_embedding(phrase)
        if embedding:
            embeddings.append(embedding)
            print(f"  ✅ Embedding généré pour : '{phrase[:30]}...'")
        else:
            print(f"  ❌ Échec pour : '{phrase}'")
            return

    print(f"\n📊 Dimension des embeddings : {len(embeddings[0])}")

    # Calculer les similarités
    print("\n🔍 === MATRICE DE SIMILARITÉ ===")
    print("(Plus proche de 1.0 = plus similaire)\n")

    # En-tête
    print("     ", end="")
    for i in range(len(phrases)):
        print(f"  {i + 1:4}", end="")
    print()

    # Matrice
    for i in range(len(embeddings)):
        print(f"{i + 1:2}. ", end="")
        for j in range(len(embeddings)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"{sim:6.3f}", end="")
        print(f"  {phrases[i][:25]}...")

    # Analyse des résultats
    print("\n🧐 === ANALYSE ===")

    # Trouver les paires les plus similaires (hors diagonale)
    max_sim = 0
    best_pair = (0, 0)

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim > max_sim:
                max_sim = sim
                best_pair = (i, j)

    print(f"🏆 Paire la plus similaire (score: {max_sim:.3f}) :")
    print(f"   • '{phrases[best_pair[0]]}'")
    print(f"   • '{phrases[best_pair[1]]}'")

    # Distance euclidienne pour comparaison
    print(
        f"\n📏 Distance euclidienne entre ces phrases : {euclidean_distance(embeddings[best_pair[0]], embeddings[best_pair[1]]):.2f}"
    )


def semantic_search(query: str, documents: List[str], top_k: int = 3):
    """
    Recherche sémantique simple : trouve les documents les plus similaires à la requête

    Args:
        query: La requête de recherche
        documents: Liste de documents dans lesquels chercher
        top_k: Nombre de résultats à retourner
    """
    print(f"🔍 Recherche pour : '{query}'\n")

    # Générer l'embedding de la requête
    query_embedding = get_embedding(query)
    if not query_embedding:
        print("❌ Impossible de générer l'embedding de la requête")
        return

    # Générer les embeddings des documents
    doc_embeddings = []
    for doc in documents:
        embedding = get_embedding(doc)
        doc_embeddings.append(embedding)

    # Calculer les similarités
    similarities = []
    for i, doc_embedding in enumerate(doc_embeddings):
        if doc_embedding:
            sim = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, sim, documents[i]))
        else:
            similarities.append((i, 0.0, documents[i]))

    # Trier par similarité décroissante
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Afficher les résultats
    print("📋 Résultats (triés par pertinence) :")
    for rank, (doc_idx, similarity, document) in enumerate(similarities[:top_k], 1):
        print(f"{rank}. Score: {similarity:.3f} | {document}")

    return similarities[:top_k]


def visualize_embeddings_2d(texts: List[str], embeddings: List[List[float]]):
    """
    Visualise les embeddings en 2D en utilisant PCA
    (Nécessite scikit-learn : pip install scikit-learn)
    """
    try:
        from sklearn.decomposition import PCA

        # Réduire la dimensionnalité à 2D
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)

        # Créer le graphique
        plt.figure(figsize=(12, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100, alpha=0.7)

        # Ajouter les labels
        for i, txt in enumerate(texts):
            plt.annotate(
                txt[:20],
                (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
            )

        plt.title("Visualisation 2D des Embeddings")
        plt.xlabel("Composante 1")
        plt.ylabel("Composante 2")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        print(f"📊 Variance expliquée : {pca.explained_variance_ratio_.sum():.2%}")

    except ImportError:
        print(
            "⚠️  Pour la visualisation, installez scikit-learn : pip install scikit-learn"
        )


# ============================================================================
# 7. EXERCICES PRATIQUES
# ============================================================================


def exercice_classification():
    """
    Exercice : Classifier des phrases par thème en utilisant les embeddings
    """
    print("🎯 === EXERCICE : CLASSIFICATION PAR THÈME ===\n")

    # Phrases de référence pour chaque thème
    themes = {
        "cuisine": "recette de cuisine gastronomie plat délicieux",
        "sport": "football tennis course entraînement compétition",
        "technologie": "ordinateur programmation intelligence artificielle",
    }

    # Phrases à classifier
    test_phrases = [
        "J'ai fait un gâteau au chocolat",
        "L'équipe a gagné le match",
        "Python est un excellent langage",
        "Cette pizza est délicieuse",
        "Il court un marathon",
        "L'IA révolutionne le monde",
    ]

    print("🏷️  Thèmes disponibles :")
    for theme, description in themes.items():
        print(f"   • {theme}: {description}")

    print("\n📝 Phrases à classifier :")
    for phrase in test_phrases:
        print(f"   • {phrase}")

    print("\n🔄 Classification en cours...\n")

    # Générer les embeddings des thèmes
    theme_embeddings = {}
    for theme, description in themes.items():
        theme_embeddings[theme] = get_embedding(description)

    # Classifier chaque phrase
    for phrase in test_phrases:
        phrase_embedding = get_embedding(phrase)

        if phrase_embedding:
            best_theme = None
            best_score = -1

            for theme, theme_embedding in theme_embeddings.items():
                if theme_embedding:
                    score = cosine_similarity(phrase_embedding, theme_embedding)
                    if score > best_score:
                        best_score = score
                        best_theme = theme

            print(f"📊 '{phrase}' → {best_theme} (score: {best_score:.3f})")


# ============================================================================
# 8. PROGRAMME PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    print("🚀 Module 4 : Embeddings et Similarité\n")
    print(
        "⚠️  Assurez-vous qu'Ollama est démarré et que le modèle 'mxbai-embed-large' est installé"
    )
    print("   Commande : ollama pull mxbai-embed-large\n")

    while True:
        print("=" * 60)
        print("MENU :")
        print("1. 🔬 Démonstration des embeddings")
        print("2. 🔍 Recherche sémantique")
        print("3. 📊 Visualisation 2D")
        print("4. 🎯 Exercice classification")
        print("5. ❌ Quitter")
        print("=" * 60)

        choice = input("Votre choix (1-5) : ").strip()

        if choice == "1":
            demo_embeddings()

        elif choice == "2":
            documents = [
                "Python est un langage de programmation populaire",
                "L'intelligence artificielle transforme notre société",
                "La cuisine française est réputée dans le monde entier",
                "Le machine learning utilise des algorithmes complexes",
                "Les pandas sont des animaux adorables",
                "NumPy est une bibliothèque Python pour le calcul scientifique",
            ]
            query = input("🔍 Votre requête de recherche : ")
            semantic_search(query, documents)

        elif choice == "3":
            phrases = [
                "Je programme en Python",
                "Python est un serpent",
                "J'aime cuisiner",
                "La recette est délicieuse",
                "Le sport c'est la santé",
                "Je cours tous les matins",
            ]
            print("📊 Génération des embeddings pour la visualisation...")
            embeddings = [get_embedding(phrase) for phrase in phrases]
            if all(embeddings):
                visualize_embeddings_2d(phrases, embeddings)
            else:
                print("❌ Erreur lors de la génération des embeddings")

        elif choice == "4":
            exercice_classification()

        elif choice == "5":
            print("👋 Au revoir ! Direction le Module 5 pour découvrir le RAG !")
            break

        else:
            print("❌ Choix invalide")

        print("\n" + "=" * 60 + "\n")