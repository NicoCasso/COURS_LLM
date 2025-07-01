# -*- coding: utf-8 -*-

import re

import psycopg2
from langchain_community.chat_models import ChatOllama

# Configuration
LLM_MODEL = "llama3.2:latest"
DB_CONFIG = {
    "host": "localhost",
    "port": 55432,
    "database": "northwind",
    "user": "postgres",
    "password": "postgres",
}


class CustomSQLAssistant:
    def __init__(self):
        self.llm = ChatOllama(model=LLM_MODEL, temperature=0)
        self.db_schema = self._get_database_schema()

    def _get_database_schema(self):
        """Récupère le schéma complet de la base"""
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        schema = {}

        # Récupérer toutes les tables
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name;
        """)
        tables = [row[0] for row in cursor.fetchall()]

        # Pour chaque table, récupérer sa structure
        for table in tables:
            cursor.execute(f"""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = '{table}' AND table_schema = 'public'
                ORDER BY ordinal_position;
            """)
            columns = cursor.fetchall()

            # Récupérer quelques exemples de données
            cursor.execute(f"SELECT COUNT(*) FROM {table};")
            count = cursor.fetchone()[0]

            schema[table] = {"columns": columns, "count": count}

        cursor.close()
        conn.close()
        return schema

    def _create_schema_prompt(self):
        """Crée un prompt avec le schéma complet"""
        schema_text = "SCHÉMA COMPLET DE LA BASE NORTHWIND:\n\n"

        for table_name, table_info in self.db_schema.items():
            schema_text += (
                f"Table: {table_name} ({table_info['count']} enregistrements)\n"
            )
            for col_name, data_type, nullable, default in table_info["columns"]:
                schema_text += f"  - {col_name}: {data_type}\n"
            schema_text += "\n"

        return schema_text

    def _execute_sql(self, query):
        """Exécute une requête SQL en toute sécurité"""
        # Vérifications de sécurité basiques
        forbidden_keywords = [
            "DROP",
            "DELETE",
            "UPDATE",
            "INSERT",
            "ALTER",
            "CREATE",
            "TRUNCATE",
        ]
        query_upper = query.upper()

        for keyword in forbidden_keywords:
            if keyword in query_upper:
                return f"❌ Requête interdite (contient {keyword})"

        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            return results
        except Exception as e:
            return f"❌ Erreur SQL: {e}"

    def ask_question(self, question):
        """Traite une question utilisateur"""

        schema_prompt = self._create_schema_prompt()

        full_prompt = f"""
Tu es un expert SQL. Voici le schéma COMPLET et RÉEL de la base de données Northwind:

{schema_prompt}

Question de l'utilisateur: {question}

Instructions:
1. Analyse la question pour comprendre l'information demandée
2. Génère UNE SEULE requête SQL SELECT qui répond exactement à la question
3. Utilise UNIQUEMENT les tables et colonnes listées ci-dessus
4. Pour les questions de comptage, utilise COUNT(*)
5. Ne génère JAMAIS de requêtes de modification (INSERT, UPDATE, DELETE, etc.)
6. Réponds UNIQUEMENT avec la requête SQL, sans explication

Exemple de format de réponse:
SELECT COUNT(*) FROM employees;

Maintenant, génère la requête SQL pour: {question}
"""

        try:
            # Demander la requête au LLM
            response = self.llm.invoke(full_prompt)
            sql_query = response.content.strip()

            # Nettoyer la réponse (enlever les backticks, etc.)
            sql_query = re.sub(r"```sql\n?", "", sql_query)
            sql_query = re.sub(r"```\n?", "", sql_query)
            sql_query = sql_query.strip()

            print(f"🔍 Requête générée: {sql_query}")

            # Exécuter la requête
            results = self._execute_sql(sql_query)

            if isinstance(results, str):  # Erreur
                return results

            # Formatter le résultat
            if len(results) == 1 and len(results[0]) == 1:
                # Résultat simple (COUNT, etc.)
                return f"Résultat: {results[0][0]}"
            else:
                # Résultats multiples
                return f"Résultats: {results}"

        except Exception as e:
            return f"❌ Erreur: {e}"


# === UTILISATION ===
def main():
    print("=== Système SQL Personnalisé ===")
    print("(Contourne l'agent LangChain défaillant)")

    assistant = CustomSQLAssistant()

    # Afficher le schéma pour info
    print("\n📋 Tables disponibles:")
    for table_name, table_info in assistant.db_schema.items():
        print(f"  - {table_name}: {table_info['count']} enregistrements")

    print("\nTapez 'exit' pour quitter, 'schema' pour voir le schéma complet")

    while True:
        question = input("\nVous: ")

        if question.lower() == "exit":
            break
        elif question.lower() == "schema":
            print(assistant._create_schema_prompt())
            continue

        print("Assistant: ...")
        response = assistant.ask_question(question)
        print(f"Assistant: {response}")


if __name__ == "__main__":
    main()