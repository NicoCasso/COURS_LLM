# -*- coding: utf-8 -*-
"""
11. Collaboration Multi-Agents (MCA) avec les LLM

Exemple simple d'un système où plusieurs agents spécialisés collaborent
pour analyser un sujet et produire un rapport complet.

Architecture: Collaboration Séquentielle + Parallèle + Validation
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List

from langchain_community.chat_models import ChatOllama

# Configuration
LLM_MODEL = "llama3.2:latest"


@dataclass
class AgentMessage:
    """Structure pour les messages échangés entre agents"""

    sender: str
    content: str
    timestamp: float
    message_type: str  # "task", "result", "feedback"


class BaseAgent:
    """Classe de base pour tous les agents"""

    def __init__(self, name: str, role: str, specialty: str):
        self.name = name
        self.role = role
        self.specialty = specialty
        self.llm = ChatOllama(model=LLM_MODEL, temperature=0.7)
        self.memory: List[AgentMessage] = []

    def add_to_memory(self, message: AgentMessage):
        """Ajoute un message à la mémoire de l'agent"""
        self.memory.append(message)

    def get_context(self) -> str:
        """Récupère le contexte des messages précédents"""
        if not self.memory:
            return ""

        context = "Contexte des échanges précédents:\n"
        for msg in self.memory[-3:]:  # Garde les 3 derniers messages
            context += f"- {msg.sender}: {msg.content[:100]}...\n"
        return context

    def process(self, task: str, context: str = "") -> str:
        """Méthode de base pour traiter une tâche"""
        raise NotImplementedError("Chaque agent doit implémenter sa méthode process")


class ResearcherAgent(BaseAgent):
    """Agent spécialisé dans la recherche et collecte d'informations"""

    def __init__(self):
        super().__init__(
            "Chercheur", "Researcher", "Collecte et analyse d'informations"
        )

    def process(self, topic: str, context: str = "") -> str:
        """Recherche des informations sur un sujet"""

        prompt = f"""
Tu es un agent CHERCHEUR expert. Ton rôle est de collecter et organiser des informations.

{context}

MISSION: Rechercher des informations complètes sur le sujet: "{topic}"

Instructions:
1. Identifie les aspects clés du sujet
2. Propose des points d'analyse importants
3. Suggère des angles d'approche intéressants
4. Structure tes findings de manière claire

Format de réponse:
## Informations clés sur {topic}

### Points principaux:
- [Point 1]
- [Point 2]
- [Point 3]

### Aspects à analyser:
- [Aspect 1]
- [Aspect 2]

### Angles d'approche:
- [Angle 1]
- [Angle 2]

Réponds de manière factuelle et structurée.
"""

        print(f"{self.name} recherche des informations sur '{topic}'...")
        response = self.llm.invoke(prompt)
        return response.content


class AnalystAgent(BaseAgent):
    """Agent spécialisé dans l'analyse critique et l'évaluation"""

    def __init__(self):
        super().__init__("Analyste", "Analyst", "Analyse critique et évaluation")

    def process(self, research_data: str, context: str = "") -> str:
        """Analyse les données de recherche"""

        prompt = f"""
Tu es un agent ANALYSTE critique. Ton rôle est d'analyser et d'évaluer les informations.

{context}

MISSION: Analyser les informations suivantes de manière critique:

{research_data}

Instructions:
1. Évalue la pertinence de chaque point
2. Identifie les forces et faiblesses
3. Propose des insights et des connexions
4. Signale les points qui nécessitent plus d'attention

Format de réponse:
## Analyse Critique

### Points forts identifiés:
- [Force 1 avec justification]
- [Force 2 avec justification]

### Points faibles ou manques:
- [Faiblesse 1]
- [Faiblesse 2]

### Insights et connexions:
- [Insight 1]
- [Insight 2]

### Recommandations:
- [Recommandation 1]
- [Recommandation 2]

Sois objectif et constructif dans ton analyse.
"""

        print(f"{self.name} analyse les données...")
        response = self.llm.invoke(prompt)
        return response.content


class WriterAgent(BaseAgent):
    """Agent spécialisé dans la rédaction et la synthèse"""

    def __init__(self):
        super().__init__("Rédacteur", "Writer", "Rédaction et synthèse")

    def process(self, research_data: str, analysis_data: str, context: str = "") -> str:
        """Rédige un rapport final basé sur les données"""

        prompt = f"""
Tu es un agent RÉDACTEUR expert. Ton rôle est de synthétiser et rédiger un rapport final.

{context}

MISSION: Créer un rapport synthétique basé sur les éléments suivants:

RECHERCHE:
{research_data}

ANALYSE:
{analysis_data}

Instructions:
1. Synthétise les informations des deux sources
2. Crée un rapport cohérent et bien structuré
3. Utilise un ton professionnel mais accessible
4. Inclus des conclusions actionnables

Format de réponse:
# Rapport de Synthèse

## Résumé Exécutif
[Synthèse en 2-3 phrases]

## Findings Principaux
[Points clés organisés]

## Analyse et Insights
[Analyse synthétique]

## Conclusions et Recommandations
[Conclusions pratiques]

## Points d'Action Suggérés
[Actions concrètes]

Crée un rapport professionnel et actionnable.
"""

        print(f"{self.name} rédige le rapport final...")
        response = self.llm.invoke(prompt)
        return response.content


class ValidatorAgent(BaseAgent):
    """Agent spécialisé dans la validation et l'assurance qualité"""

    def __init__(self):
        super().__init__("Validateur", "Validator", "Validation et assurance qualité")

    def process(self, final_report: str, context: str = "") -> str:
        """Valide et améliore le rapport final"""

        prompt = f"""
Tu es un agent VALIDATEUR expert. Ton rôle est de valider et améliorer la qualité.

{context}

MISSION: Valider et améliorer le rapport suivant:

{final_report}

Instructions:
1. Vérifie la cohérence et la clarté
2. Identifie les améliorations possibles
3. Suggère des corrections ou ajouts
4. Évalue la qualité globale

Format de réponse:
## Validation du Rapport

### Évaluation Qualité (sur 10): [Note]

### Points Positifs:
- [Point positif 1]
- [Point positif 2]

### Améliorations Suggérées:
- [Amélioration 1]
- [Amélioration 2]

### Corrections Recommandées:
- [Correction 1 si nécessaire]
- [Correction 2 si nécessaire]

### Verdict Final:
[Approuvé/À réviser avec justification]

### Version Améliorée (si nécessaire):
[Version corrigée du rapport si des améliorations majeures sont nécessaires]

Sois constructif et précis dans tes retours.
"""

        print(f"{self.name} valide le rapport...")
        response = self.llm.invoke(prompt)
        return response.content


class MultiAgentOrchestrator:
    """Orchestrateur qui coordonne la collaboration entre agents"""

    def __init__(self):
        self.agents = {
            "researcher": ResearcherAgent(),
            "analyst": AnalystAgent(),
            "writer": WriterAgent(),
            "validator": ValidatorAgent(),
        }
        self.conversation_log: List[AgentMessage] = []

    def log_message(self, sender: str, content: str, msg_type: str = "result"):
        """Enregistre un message dans le log de conversation"""
        message = AgentMessage(
            sender=sender, content=content, timestamp=time.time(), message_type=msg_type
        )
        self.conversation_log.append(message)

        # Ajoute le message à la mémoire de tous les agents
        for agent in self.agents.values():
            agent.add_to_memory(message)

    def process_topic(self, topic: str) -> Dict[str, Any]:
        """Traite un sujet en coordination multi-agents"""

        print(f"\nMISSION: Analyse complète du sujet '{topic}'")
        print("=" * 60)

        results = {}

        # Étape 1: Recherche (Agent Chercheur)
        print("\nÉTAPE 1: Recherche d'informations")
        research_result = self.agents["researcher"].process(topic)
        self.log_message("Chercheur", research_result)
        results["research"] = research_result

        # Étape 2: Analyse (Agent Analyste)
        print("\nÉTAPE 2: Analyse critique")
        context = self.agents["analyst"].get_context()
        analysis_result = self.agents["analyst"].process(research_result, context)
        self.log_message("Analyste", analysis_result)
        results["analysis"] = analysis_result

        # Étape 3: Rédaction (Agent Rédacteur)
        print("\nÉTAPE 3: Rédaction du rapport")
        context = self.agents["writer"].get_context()
        report_result = self.agents["writer"].process(
            research_result, analysis_result, context
        )
        self.log_message("Rédacteur", report_result)
        results["report"] = report_result

        # Étape 4: Validation (Agent Validateur)
        print("\nÉTAPE 4: Validation et amélioration")
        context = self.agents["validator"].get_context()
        validation_result = self.agents["validator"].process(report_result, context)
        self.log_message("Validateur", validation_result)
        results["validation"] = validation_result

        return results

    def print_results(self, results: Dict[str, Any]):
        """Affiche les résultats de manière structurée"""

        print("\n" + "=" * 80)
        print("RÉSULTATS DE LA COLLABORATION MULTI-AGENTS")
        print("=" * 80)

        sections = [
            ("RECHERCHE", results["research"]),
            ("ANALYSE", results["analysis"]),
            ("RAPPORT FINAL", results["report"]),
            ("VALIDATION", results["validation"]),
        ]

        for title, content in sections:
            print(f"\n{title}")
            print("-" * 40)
            print(content)
            print()


def main():
    """Fonction principale pour tester le système multi-agents"""

    print("SYSTÈME DE COLLABORATION MULTI-AGENTS")
    print("Architecture: Pipeline Séquentiel avec Validation")
    print("\nAgents disponibles:")
    print("- Chercheur: Collecte d'informations")
    print("- Analyste: Analyse critique")
    print("- Rédacteur: Synthèse et rédaction")
    print("- Validateur: Contrôle qualité")

    # Initialisation de l'orchestrateur
    orchestrator = MultiAgentOrchestrator()

    while True:
        print("\n" + "=" * 50)
        topic = input("Entrez un sujet à analyser (ou 'exit' pour quitter): ").strip()

        if topic.lower() in ["exit", "quit", "sortir"]:
            print("Au revoir!")
            break

        if not topic:
            print("Veuillez entrer un sujet valide.")
            continue

        try:
            # Traitement par les agents
            start_time = time.time()
            results = orchestrator.process_topic(topic)
            end_time = time.time()

            # Affichage des résultats
            orchestrator.print_results(results)

            print(
                f"\nTemps total de traitement: {end_time - start_time:.2f} secondes"
            )
            print(f"Messages échangés: {len(orchestrator.conversation_log)}")

        except Exception as e:
            print(f"Erreur lors du traitement: {e}")


if __name__ == "__main__":
    main()