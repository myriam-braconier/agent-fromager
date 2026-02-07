# app.py - LIGNES 1-10
import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

AUTH_USERNAME = os.environ.get("AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.environ.get("AUTH_PASSWORD", "defaultpassword")

print("=" * 50)
print("🧪 MODE LOCAL - Chargement .env")
print("=" * 50)

import requests
import random
import gradio as gr
import json
import os
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download
import pandas as pd

# AJOUTER CES IMPORTS POUR LE CHAT
import time
from typing import List, Dict, Optional

# ===== VARIABLES GLOBALES =====
fallback_cache = None
recipe_map = {}


class AgentFromagerHF:
    """Agent fromager avec persistance HF Dataset"""

    def __init__(self):
        self.rng = random.Random()
        self.knowledge_base = self._init_knowledge()
        self.recipes_file = "recipes_history.json"
        self.hf_repo = "volubyl/fromager-recipes"
        self.hf_token = os.environ.get("HF_TOKEN")
        self.api = HfApi(token=self.hf_token) if self.hf_token else None
        self.http = requests.Session()

        # Configuration HTTP
        self.http.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/121.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "fr-FR,fr;q=0.9",
                "Referer": "https://duckduckgo.com/",
                "Connection": "keep-alive",
            }
        )

        # Variables d'environnement
        self.serpapi_key = os.environ.get("SERPAPI_KEY")
        self.hf_token = os.environ.get("HF_TOKEN")

        # ===== SECTION DIAGNOSTIC ORIGINALE =====
        print("=" * 50)
        print("🧪 DIAGNOSTIC SYSTÈME")
        print("=" * 50)
        print(f"   SerpAPI: {'✅ PRÉSENTE' if self.serpapi_key else '❌ ABSENTE'}")
        print(
            f"🔍 HF_TOKEN détecté : {'✅ OUI' if os.environ.get('HF_TOKEN') else '❌ NON'}"
        )
        print(f"🔍 Repo cible : {self.hf_repo}")
        print(f"🔍 API initialisée : {'✅ OUI' if self.api else '❌ NON'}")
        print("=" * 50)

        # ===== CONFIGURATION CHAT LLM =====
        print("\n" + "=" * 50)
        print("🤖 CONFIGURATION CHAT LLM")
        print("=" * 50)

        # Initialiser tous les attributs
        self.deepseek_enabled = False
        self.ollama_enabled = False
        self.hf_inference_enabled = False
        self.lmstudio_enabled = False
        self.google_ai_enabled = False
        self.openrouter_enabled = False
        self.together_enabled = False  # Ajouté pour Together AI

        # ===== OPENROUTER (PRIORITÉ HAUTE - GRATUIT AVEC QUOTAS) =====
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        if self.openrouter_api_key and self.openrouter_api_key.strip():
            self.openrouter_enabled = True
            print("✅ OpenRouter: CONFIGURÉ (gratuit avec quotas)")
            print(
                f"   📝 Clé: {self.openrouter_api_key[:10]}...{self.openrouter_api_key[-4:]}"
            )
        else:
            print("❌ OpenRouter: PAS DE CLÉ - https://openrouter.ai/ (gratuit)")

        # ===== GOOGLE AI / GEMINI (PRIORITÉ MOYENNE - TRÈS GÉNÉREUX) =====
        self.google_ai_api_key = os.environ.get("GOOGLE_AI_API_KEY")
        if self.google_ai_api_key:
            self.google_ai_enabled = True
            print("✅ Google AI (Gemini): CONFIGURÉ (gratuit)")
        else:
            print("ℹ️ Google AI: PAS DE CLÉ - https://makersuite.google.com/")

        # ===== TOGETHER AI (PRIORITÉ MOYENNE - 25$ GRATUIT) =====
        self.together_api_key = os.environ.get("TOGETHER_API_KEY")
        if self.together_api_key:
            self.together_enabled = True
            print("✅ Together AI: CONFIGURÉ (25$ gratuit)")
        else:
            print("ℹ️ Together AI: PAS DE CLÉ - https://api.together.xyz/")

        # ===== DEEPSEEK (PRIORITÉ BASSE - VOUS AVEZ DIT QUE ÇA NE FONCTIONNE PAS) =====
        self.deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
        if self.deepseek_api_key and self.deepseek_api_key != "sk-xxx":
            self.deepseek_enabled = True
            print("✅ DeepSeek: CONFIGURÉ")
        else:
            print("❌ DeepSeek: NON CONFIGURÉ")

        # ===== SOLUTIONS LOCALES =====

        # OLLAMA (local)
        self.ollama_url = "http://localhost:11434/api/generate"
        self.ollama_model = "qwen2.5:7b"  # Meilleur que llama2 pour le français

        try:
            response = requests.post(
                self.ollama_url,
                json={"model": self.ollama_model, "prompt": "test", "stream": False},
                timeout=2,
            )
            self.ollama_enabled = response.status_code == 200
        except:
            self.ollama_enabled = False

        if self.ollama_enabled:
            print(f"✅ Ollama: CONNECTÉ ({self.ollama_model})")
        else:
            print("ℹ️ Ollama: NON DÉTECTÉ")

        # LM STUDIO (local)
        try:
            response = requests.get("http://localhost:1234/v1/models", timeout=2)
            self.lmstudio_enabled = response.status_code == 200
        except:
            self.lmstudio_enabled = False

        if self.lmstudio_enabled:
            print("✅ LM Studio: CONNECTÉ")
        else:
            print("ℹ️ LM Studio: NON DÉTECTÉ")

        # HUGGING FACE INFERENCE
        if self.hf_token:
            self.hf_inference_enabled = True
            print("✅ Hugging Face Inference: DISPONIBLE")
        else:
            print("ℹ️ Hugging Face Inference: PAS DE TOKEN")

        # FALLBACK LOCAL (TOUJOURS DISPONIBLE)
        print("✅ Base de connaissances: PRÊTE (fallback intelligent)")

        # ===== RÉSUMÉ DES OPTIONS DISPONIBLES =====
        print("\n" + "=" * 50)
        print("🎯 OPTIONS DISPONIBLES (par ordre de priorité)")
        print("=" * 50)

        options = []
        if self.openrouter_enabled:
            options.append("1. OpenRouter 🌐 (cloud, gratuit)")
        if self.google_ai_enabled:
            options.append("2. Google AI 🌐 (cloud, gratuit)")
        if self.together_enabled:
            options.append("3. Together AI 🌐 (cloud, 25$ gratuit)")
        if self.ollama_enabled:
            options.append("4. Ollama 💻 (local, 100% gratuit)")
        if self.lmstudio_enabled:
            options.append("5. LM Studio 💻 (local, 100% gratuit)")
        if self.hf_inference_enabled:
            options.append("6. Hugging Face 🌐 (cloud, gratuit)")
        if self.deepseek_enabled:
            options.append("7. DeepSeek 🌐 (cloud)")

        for option in options:
            print(f"   {option}")

        if not options:
            print("   ⚠️ AUCUN LLM externe - mode fallback uniquement")
        else:
            print(f"\n   Total: {len(options)} option(s) disponible(s)")

        print("=" * 50 + "\n")
        # ===== FIN CONFIGURATION CHAT =====

        # Charger l'historique depuis HF au démarrage
        self._download_history_from_hf()

        # Charger l'historique en mémoire
        self.history = self._load_history()

        # Configuration de retry pour les requêtes HTTP
        self._setup_retry_session()

    def adapt_recipe_to_profile(self, recipe: str, profile: str) -> str:
        """Adapte la recette selon le profil utilisateur"""

        profiles_config = {
            "🧀 Amateur": {
                "tone": "accessible et encourageant",
                "details": "explications simples, astuces pratiques",
                "vocabulary": "termes courants, équivalences faciles",
            },
            "🏭 Producteur": {
                "tone": "technique et précis",
                "details": "températures exactes, timing précis, rendement",
                "vocabulary": "termes professionnels, normes sanitaires",
            },
            "🎓 Formateur": {
                "tone": "pédagogique et structuré",
                "details": "points d'attention, erreurs courantes, variantes",
                "vocabulary": "objectifs pédagogiques, progression",
            },
        }

        if profile not in profiles_config:
            return recipe

        config = profiles_config[profile]

        # Ajouter un préambule adapté au profil
        if profile == "🧀 Amateur":
            prefix = f"🏠 **RECETTE POUR AMATEUR**\n\n"
            prefix += "✨ *Conseils débutant :*\n"
            prefix += "- Prenez votre temps, la fromagerie demande de la patience\n"
            prefix += "- Suivez les températures indiquées avec un thermomètre\n"
            prefix += "- N'hésitez pas à adapter selon vos goûts\n\n"

        elif profile == "🏭 Producteur":
            prefix = f"🏭 **FICHE TECHNIQUE PRODUCTION**\n\n"
            prefix += "📊 *Points de contrôle qualité :*\n"
            prefix += "- Respect strict des températures et temps\n"
            prefix += "- Traçabilité des matières premières\n"
            prefix += "- Conditions d'hygiène professionnelles\n\n"

        else:  # Formateur
            prefix = f"🎓 **SUPPORT PÉDAGOGIQUE**\n\n"
            prefix += "📚 *Objectifs d'apprentissage :*\n"
            prefix += "- Maîtriser les étapes clés de la transformation\n"
            prefix += "- Comprendre les réactions biochimiques\n"
            prefix += "- Identifier les points critiques\n\n"

        return prefix + recipe

    def _setup_retry_session(self):
        """Configure la session avec retry automatique"""
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.http.mount("https://", adapter)
        self.http.mount("http://", adapter)

    def _test_ollama_connection(self):
        """Teste la connexion à Ollama (local)"""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama2", "prompt": "test", "stream": False},
                timeout=3,
            )
            return response.status_code == 200
        except:
            return False

    # ===== FONCTION PRINCIPALE MISE À JOUR =====
    def search_web_recipes(
        self, ingredients: str, cheese_type: str, max_results: int = 6
    ) -> list:
        """Recherche web - GARANTIT au moins 6 résultats"""

        all_recipes = []
        min_required = max_results  # On veut AU MOINS 6 résultats

        try:
            from urllib.parse import quote
            from bs4 import BeautifulSoup
            import time
            import random

            query = f"recette fromage {ingredients}"
            if cheese_type and cheese_type != "Laissez l'IA choisir":
                query = f"recette {cheese_type} {ingredients}"

            print(f"🔍 Recherche garantie: {query} (minimum {min_required} résultats)")

            # ===== PHASE 1: MOTEURS PRINCIPAUX (rapides) =====
            primary_engines = [
                ("Google", self._search_google),
                ("Bing", self._search_bing),
                ("Ecosia", self._search_ecosia),
            ]

            for engine_name, engine_func in primary_engines:
                if len(all_recipes) >= min_required * 2:  # On veut du choix
                    break

                try:
                    print(f"  🔎 {engine_name}...")
                    recipes = engine_func(query, min_required)

                    if recipes:
                        # Ajouter avec vérification des doublons
                        for recipe in recipes:
                            norm_url = self._normalize_url(recipe["url"])
                            if norm_url not in [
                                self._normalize_url(r["url"]) for r in all_recipes
                            ]:
                                all_recipes.append(recipe)

                        print(
                            f"    ✅ {len(recipes)} nouveaux, total: {len(all_recipes)}"
                        )

                    time.sleep(random.uniform(1, 1.5))

                except Exception as e:
                    print(f"    ⚠️ {engine_name} échoué: {e}")
                    continue

            # ===== PHASE 2: VÉRIFICATION SI ON A ASSEZ =====
            if len(all_recipes) >= min_required:
                # On a assez, on trie et on retourne les meilleurs
                unique_recipes = self._deduplicate_recipes(all_recipes)
                unique_recipes.sort(key=lambda x: x.get("score", 0), reverse=True)
                final = unique_recipes[:min_required]
                print(f"🎯 Phase 1 suffisante: {len(final)} résultats uniques")
                return final

            # ===== PHASE 3: MOTEURS SECONDAIRES (si besoin) =====
            print(f"⚠️ Seulement {len(all_recipes)} résultats, Phase 2...")

            secondary_engines = [
                ("Qwant", self._search_qwant),
                ("DuckDuckGo Lite", self._search_duckduckgo_lite),
                ("Yandex", self._search_yandex),
            ]

            for engine_name, engine_func in secondary_engines:
                if len(all_recipes) >= min_required * 2:
                    break

                try:
                    print(f"  🔎 {engine_name} (secondaire)...")
                    recipes = engine_func(query, min_required)

                    if recipes:
                        for recipe in recipes:
                            norm_url = self._normalize_url(recipe["url"])
                            if norm_url not in [
                                self._normalize_url(r["url"]) for r in all_recipes
                            ]:
                                all_recipes.append(recipe)

                        print(
                            f"    ✅ {len(recipes)} nouveaux, total: {len(all_recipes)}"
                        )

                    time.sleep(random.uniform(0.8, 1.2))

                except Exception as e:
                    print(f"    ⚠️ {engine_name} échoué: {e}")
                    continue

            # ===== PHASE 4: GARANTIE MINIMUM =====
            print(f"📊 Après Phase 2: {len(all_recipes)} résultats")

            if len(all_recipes) >= min_required:
                # On a assez maintenant
                unique_recipes = self._deduplicate_recipes(all_recipes)
                unique_recipes.sort(key=lambda x: x.get("score", 0), reverse=True)
                final = unique_recipes[:min_required]
                print(f"🎯 Suffisant après Phase 2: {len(final)} résultats")
                return final

            # ===== PHASE 5: BACKUP HYBRIDE (force d'avoir 6 résultats) =====
            print(f"🚨 BACKUP: Seulement {len(all_recipes)} résultats, on complète...")

            # 1. D'abord les résultats web qu'on a
            final_recipes = self._deduplicate_recipes(all_recipes)

            # 2. Ensuite le fallback enrichi
            needed = min_required - len(final_recipes)
            if needed > 0:
                print(f"   📥 Besoin de {needed} résultats supplémentaires")

                # Fallback statique
                fallback = self._get_enriched_fallback_recipes(
                    ingredients, cheese_type, needed + 3
                )

                # Ajouter ceux qu'on n'a pas déjà
                for recipe in fallback:
                    if len(final_recipes) >= min_required:
                        break

                    norm_url = self._normalize_url(recipe["url"])
                    if norm_url not in [
                        self._normalize_url(r["url"]) for r in final_recipes
                    ]:
                        final_recipes.append(recipe)

                print(
                    f"   ✅ Ajouté {len(final_recipes) - len(all_recipes)} du fallback"
                )

            # 3. Si TOUJOURS pas assez, on génère des recettes "similaires"
            if len(final_recipes) < min_required:
                print(
                    f"   🚨 CRITIQUE: Encore {min_required - len(final_recipes)} manquants"
                )
                generated = self._generate_similar_recipes(
                    ingredients, cheese_type, min_required - len(final_recipes)
                )
                final_recipes.extend(generated)

            # 4. Finalisation
            final_recipes = final_recipes[:min_required]
            final_recipes.sort(key=lambda x: x.get("score", 0), reverse=True)

            print(
                f"🎯 FINAL: Garanti {len(final_recipes)} résultats (dont {len(all_recipes)} du web)"
            )
            return final_recipes

        except Exception as e:
            print(f"❌ Erreur recherche garantie: {e}")
            import traceback

            traceback.print_exc()

            # Fallback absolu
            return self._get_absolute_fallback(ingredients, cheese_type, min_required)

    def _deduplicate_recipes(self, recipes):
        """Élimine les doublons tout en gardant les meilleures versions"""
        unique_recipes = []
        seen_urls = set()

        # Trier d'abord par score pour garder les meilleures versions
        recipes.sort(key=lambda x: x.get("score", 0), reverse=True)

        for recipe in recipes:
            norm_url = self._normalize_url(recipe["url"])

            if not norm_url:
                # Recette sans URL valide, on garde quand même
                unique_recipes.append(recipe)
            elif norm_url not in seen_urls:
                seen_urls.add(norm_url)
                unique_recipes.append(recipe)

        return unique_recipes

    def _generate_similar_recipes(self, ingredients, cheese_type, count):
        """Génère des recettes similaires basées sur la base de connaissances"""
        print(f"   🧠 Génération de {count} recettes similaires...")

        similar_recipes = []
        base_url = "https://fromage-maison.com/recettes/"

        # Extraire des mots-clés des ingrédients
        ingredients_lower = ingredients.lower()
        keywords = []

        for word in ingredients_lower.split(","):
            word = word.strip()
            if len(word) > 3 and word not in ["lait", "de", "et", "avec"]:
                keywords.append(word)

        # Types de fromage courants pour suggestions
        cheese_types = [
            "fromage frais",
            "chèvre",
            "brebis",
            "pâte molle",
            "camembert",
            "brie",
            "tomme",
            "bleu",
        ]

        for i in range(count):
            # Choisir un type aléatoire ou utiliser celui spécifié
            if cheese_type and cheese_type != "Laissez l'IA choisir":
                chosen_type = cheese_type.lower()
            else:
                chosen_type = self.rng.choice(cheese_types)

            # Construire un titre crédible
            if "chèvre" in ingredients_lower or "chevre" in ingredients_lower:
                titles = [
                    "Fromage de chèvre artisanal",
                    "Crottin de chèvre maison",
                    "Bûche de chèvre à l'herbe",
                ]
            elif "brebis" in ingredients_lower:
                titles = [
                    "Fromage de brebis affiné",
                    "Brebis des Pyrénées maison",
                    "Fromage de brebis à pâte pressée",
                ]
            elif "frais" in ingredients_lower or "blanc" in ingredients_lower:
                titles = [
                    "Fromage frais maison",
                    "Faisselle artisanale",
                    "Fromage blanc crémeux",
                ]
            else:
                titles = [
                    f"Fromage {chosen_type} artisanal",
                    f"Recette de {chosen_type} maison",
                    f"{chosen_type.title()} fait maison",
                ]

            title = self.rng.choice(titles)
            url_slug = (
                title.lower().replace(" ", "-").replace("é", "e").replace("è", "e")
            )

            similar_recipes.append(
                {
                    "title": title,
                    "url": f"{base_url}{url_slug}-{i+1}",
                    "description": f"Recette similaire à base de {ingredients.split(',')[0].strip()}",
                    "source": "fromage-maison.com",
                    "score": 4,  # Score bas car généré
                    "generated": True,
                }
            )

        return similar_recipes

    def _get_absolute_fallback(self, ingredients, cheese_type, min_required):
        """Fallback NEUTRE - respecte le type de lait demandé"""
        print(f"🚨 FALLBACK ABSOLU activé pour {min_required} résultats")

        # Détecter le type de lait demandé (si spécifié)
        lait_demande = self._detect_lait_from_ingredients(ingredients)
        if lait_demande:
            print(f"   🥛 Lait demandé détecté: {lait_demande}")

        # ===== 1. BASE DE RECETTES NEUTRES (sans mention de lait spécifique) =====
        neutral_recipes = [
            {
                "title": "Fromage frais maison facile",
                "url": "https://www.marmiton.org/recettes/recette_fromage-frais-maison_337338.aspx",
                "description": "Recette de fromage frais basique",
                "source": "marmiton.org",
                "score": 8,
                "lait": None,  # Neutre, peut être adapté
            },
            {
                "title": "Recette de mozzarella maison",
                "url": "https://www.regal.fr/produit/fromage/recette-mozzarella-maison-100305",
                "description": "Mozzarella fraîche en quelques heures",
                "source": "regal.fr",
                "score": 7,
                "lait": "bufflonne",  # Spécifique mais différent
            },
            {
                "title": "Brie maison traditionnel",
                "url": "https://www.femmeactuelle.fr/cuisine/guides-cuisine/fromage-maison-213130",
                "description": "Brie à croûte fleurie fait maison",
                "source": "femmeactuelle.fr",
                "score": 6,
                "lait": "vache",  # Brie est toujours au lait de vache
            },
            {
                "title": "Fromage à pâte pressée",
                "url": "https://www.750g.com/recette-fromage-pate-pressee_452189.htm",
                "description": "Techniques de pressage pour fromages durs",
                "source": "750g.com",
                "score": 6,
                "lait": None,  # Technique générique
            },
            {
                "title": "Ricotta maison au petit-lait",
                "url": "https://cuisine.journaldesfemmes.fr/recette/415921-ricotta-maison",
                "description": "Ricotta crémeuse à partir de petit-lait",
                "source": "cuisine.journaldesfemmes.fr",
                "score": 7,
                "lait": None,  # Peut être fait avec n'importe quel petit-lait
            },
            {
                "title": "Faisselle maison en 24h",
                "url": "https://www.marmiton.org/recettes/recette_faisselle-maison_537338.aspx",
                "description": "Faisselle crémeuse à déguster nature",
                "source": "marmiton.org",
                "score": 7,
                "lait": None,  # Neutre
            },
        ]

        # ===== 2. RECETTES SPÉCIFIQUES PAR TYPE DE LAIT =====
        lait_specific_recipes = {
            "brebis": [
                {
                    "title": "Fromage de brebis des Pyrénées",
                    "url": "https://www.marmiton.org/recettes/recette_fromage-brebis-pyrenees_441229.aspx",
                    "description": "Fromage à pâte pressée de brebis façon Ossau-Iraty",
                    "source": "marmiton.org",
                    "score": 9,
                    "lait": "brebis",
                },
                {
                    "title": "Recette de Manchego maison",
                    "url": "https://cuisine.journaldesfemmes.fr/recette/412345-manchego-maison",
                    "description": "Fromage espagnol de brebis à pâte pressée",
                    "source": "cuisine.journaldesfemmes.fr",
                    "score": 8,
                    "lait": "brebis",
                },
                {
                    "title": "Pecorino romano artisanal",
                    "url": "https://www.750g.com/pecorino-romano-maison-r352700.htm",
                    "description": "Fromage de brebis italien à pâte dure",
                    "source": "750g.com",
                    "score": 8,
                    "lait": "brebis",
                },
            ],
            "chèvre": [
                {
                    "title": "Fromage de chèvre frais maison",
                    "url": "https://www.marmiton.org/recettes/recette_fromage-chevre-frais_337339.aspx",
                    "description": "Chèvre frais à déguster rapidement",
                    "source": "marmiton.org",
                    "score": 9,
                    "lait": "chèvre",
                },
                {
                    "title": "Crottin de Chavignol artisanal",
                    "url": "https://cuisine.journaldesfemmes.fr/recette/315922-crottin-chavignol",
                    "description": "Crottin de chèvre affiné à la cendre",
                    "source": "cuisine.journaldesfemmes.fr",
                    "score": 8,
                    "lait": "chèvre",
                },
                {
                    "title": "Bûche de chèvre aux herbes",
                    "url": "https://www.750g.com/buche-chevre-herbes-r252701.htm",
                    "description": "Bûche de chèvre roulée dans des herbes",
                    "source": "750g.com",
                    "score": 8,
                    "lait": "chèvre",
                },
            ],
            "vache": [
                {
                    "title": "Camembert normand maison",
                    "url": "https://www.marmiton.org/recettes/recette_camembert-maison_551229.aspx",
                    "description": "Camembert à croûte fleurie",
                    "source": "marmiton.org",
                    "score": 9,
                    "lait": "vache",
                },
                {
                    "title": "Comté affiné 6 mois maison",
                    "url": "https://cuisine.journaldesfemmes.fr/recette/512345-comte-maison",
                    "description": "Fromage à pâte pressée cuite",
                    "source": "cuisine.journaldesfemmes.fr",
                    "score": 8,
                    "lait": "vache",
                },
                {
                    "title": "Reblochon de Savoie maison",
                    "url": "https://www.750g.com/reblochon-maison-r552700.htm",
                    "description": "Fromage à pâte pressée non cuite",
                    "source": "750g.com",
                    "score": 7,
                    "lait": "vache",
                },
            ],
        }

        # ===== 3. SÉLECTION INTELLIGENTE =====
        selected_recipes = []

        # A. Si un lait est spécifiquement demandé → prendre les recettes spécifiques
        if lait_demande and lait_demande in lait_specific_recipes:
            print(f"   🎯 Sélection spécifique pour lait de {lait_demande}")
            selected_recipes = lait_specific_recipes[lait_demande][:min_required]

        # B. Sinon, ou si pas assez → ajouter des recettes neutres
        if len(selected_recipes) < min_required:
            needed = min_required - len(selected_recipes)
            print(f"   📥 Besoin de {needed} recettes supplémentaires (neutres)")

            # Filtrer les neutres pour éviter les incohérences
            for recipe in neutral_recipes:
                if len(selected_recipes) >= min_required:
                    break

                # Vérifier la cohérence
                is_coherent = True

                if lait_demande and recipe["lait"]:
                    # Si on demande un lait spécifique, éviter les recettes avec d'autres laits
                    if lait_demande == "brebis" and recipe["lait"] in [
                        "chèvre",
                        "vache",
                    ]:
                        is_coherent = False
                    elif lait_demande == "chèvre" and recipe["lait"] in [
                        "brebis",
                        "vache",
                    ]:
                        is_coherent = False
                    elif lait_demande == "vache" and recipe["lait"] in [
                        "brebis",
                        "chèvre",
                    ]:
                        is_coherent = False

                if is_coherent and recipe["url"] not in [
                    r["url"] for r in selected_recipes
                ]:
                    selected_recipes.append(recipe)

        # C. Si TOUJOURS pas assez → dernier recours (très neutre)
        if len(selected_recipes) < min_required:
            print(
                f"   🚨 Dernier recours: {min_required - len(selected_recipes)} manquants"
            )

            ultra_neutral = [
                {
                    "title": "Guide du fromage maison",
                    "url": "https://www.lerustique.fr/guide-fromage-maison",
                    "description": "Toutes les techniques pour faire son fromage",
                    "source": "lerustique.fr",
                    "score": 6,
                    "lait": None,
                },
                {
                    "title": "Matériel pour fromager amateur",
                    "url": "https://www.tompress.fr/fromagerie-amateur",
                    "description": "Guide d'équipement pour débuter",
                    "source": "tompress.fr",
                    "score": 5,
                    "lait": None,
                },
            ]

            for recipe in ultra_neutral:
                if len(selected_recipes) >= min_required:
                    break
                selected_recipes.append(recipe)

        # ===== 4. FINALISATION =====
        # Garantir le nombre exact
        selected_recipes = selected_recipes[:min_required]

        # Vérifier la cohérence finale
        lait_trouves = set()
        for r in selected_recipes:
            if r["lait"]:
                lait_trouves.add(r["lait"])

        print(f"✅ Fallback: {len(selected_recipes)} résultats")

        if len(lait_trouves) == 1:
            print(f"   🎯 Tous au lait de: {list(lait_trouves)[0]}")
        elif len(lait_trouves) > 1:
            print(f"   ⚠️ Mélange de laits: {lait_trouves}")
        else:
            print(f"   ✅ Recettes neutres (pas de lait spécifique)")

        return selected_recipes

    def _detect_lait_from_ingredients(self, ingredients):
        """Détecte le type de lait depuis les ingrédients"""
        if not ingredients:
            return None

        ingredients_lower = ingredients.lower()

        # Mots-clés pour chaque type de lait
        lait_patterns = {
            "brebis": [
                "brebis",
                "mouton",
                "ovin",
                "sheep",
                "manchego",
                "pecorino",
                "roquefort",
            ],
            "chèvre": [
                "chèvre",
                "chevre",
                "caprin",
                "goat",
                "crottin",
                "sainte-maure",
                "bûche",
            ],
            "vache": [
                "vache",
                "bovin",
                "cow",
                "lait de vache",
                "camembert",
                "brie",
                "comté",
            ],
            "bufflonne": ["bufflonne", "buffle", "buffalo", "mozzarella di bufala"],
        }

        # Priorité aux patterns les plus spécifiques
        for lait_type, patterns in lait_patterns.items():
            for pattern in patterns:
                if pattern in ingredients_lower:
                    return lait_type

        # Vérifier "lait de X"
        if "lait de brebis" in ingredients_lower:
            return "brebis"
        elif (
            "lait de chèvre" in ingredients_lower
            or "lait de chevre" in ingredients_lower
        ):
            return "chèvre"
        elif "lait de vache" in ingredients_lower:
            return "vache"

        return None

    # ===== FONCTIONS AUXILIAIRES =====

    def search_web_recipes_fallback(self, ingredients, cheese_type, max_results=6):
        """Fallback robuste avec différentes stratégies"""
        print("🔄 Activation du mode fallback")

        try:
            # Stratégie 1: Recherche très simple
            simple_results = self._search_simple(ingredients, cheese_type, max_results)
            if simple_results:
                print(f"✅ Fallback simple: {len(simple_results)} résultats")
                return simple_results

            # Stratégie 2: Retourner des recettes statiques de la base
            print("⚠️ Utilisation de la base statique")
            return self._get_static_fallback_recipes(ingredients, cheese_type)

        except Exception as e:
            print(f"❌ Erreur fallback: {e}")
            return []

    def _search_simple(self, ingredients, cheese_type, max_results):
        """Recherche HTML très simple"""
        try:
            from urllib.parse import quote
            import requests

            query = f"fromage {ingredients} recette"
            url = f"https://duckduckgo.com/html/?q={quote(query)}&kl=fr-fr"

            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
            }

            response = requests.get(url, headers=headers, timeout=15)

            if response.status_code == 200:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(response.text, "html.parser")

                recipes = []
                # Chercher tous les liens pertinents
                for a in soup.find_all("a", href=True)[:30]:
                    url = a.get("href", "")
                    title = a.get_text(strip=True)

                    # Filtrer les liens pertinents
                    if (
                        (
                            "fromage" in title.lower()
                            or "cheese" in title.lower()
                            or "recette" in title.lower()
                        )
                        and "http" in url
                        and len(title) > 10
                    ):

                        # Extraire le domaine
                        domain = self._extract_domain(url)

                        recipes.append(
                            {
                                "title": title[:80],
                                "url": url,
                                "description": f"Recette de {ingredients.split(',')[0]}",
                                "source": domain,
                                "score": 5,
                            }
                        )

                        if len(recipes) >= max_results:
                            break

                return recipes
        except Exception as e:
            print(f"⚠️ Erreur recherche simple: {e}")

        return []

    def _get_static_fallback_recipes(self, ingredients, cheese_type):
        """Recettes statiques de fallback"""
        static_recipes = [
            {
                "title": "Recette de fromage frais maison",
                "url": "https://www.marmiton.org/recettes/recette_fromage-frais-maison_337338.aspx",
                "description": "Recette simple de fromage frais avec lait et présure",
                "source": "marmiton.org",
                "score": 8,
            },
            {
                "title": "Fromage blanc maison en 24h",
                "url": "https://cuisine.journaldesfemmes.fr/recette/315921-fromage-blanc-maison",
                "description": "Fromage blanc crémeux fait maison avec ferments lactiques",
                "source": "journaldesfemmes.fr",
                "score": 7,
            },
            {
                "title": "Faire son fromage de chèvre maison",
                "url": "https://www.750g.com/faire-son-fromage-de-chevre-maison-r152700.htm",
                "description": "Guide complet pour fabriquer du fromage de chèvre à la maison",
                "source": "750g.com",
                "score": 6,
            },
            {
                "title": "Recette de mozzarella maison",
                "url": "https://www.regal.fr/produit/fromage/recette-mozzarella-maison-100305",
                "description": "Mozzarella fraîche faite maison en quelques heures",
                "source": "regal.fr",
                "score": 7,
            },
            {
                "title": "Fromage à pâte pressée maison",
                "url": "https://www.femmeactuelle.fr/cuisine/guides-cuisine/fromage-maison-213130",
                "description": "Techniques pour réaliser des fromages à pâte pressée",
                "source": "femmeactuelle.fr",
                "score": 6,
            },
        ]

        # Filtrer par ingrédients si possible
        filtered = []
        ingredients_lower = ingredients.lower()
        cheese_type_lower = cheese_type.lower() if cheese_type else ""

        for recipe in static_recipes:
            score = recipe["score"]

            # Bonus pour correspondance avec ingrédients
            if "chèvre" in ingredients_lower and "chèvre" in recipe["title"].lower():
                score += 3
            elif "frais" in ingredients_lower and "frais" in recipe["title"].lower():
                score += 2
            elif (
                "mozzarella" in ingredients_lower
                and "mozzarella" in recipe["title"].lower()
            ):
                score += 3

            # Bonus pour correspondance avec type
            if (
                "pâte pressée" in cheese_type_lower
                and "pâte pressée" in recipe["title"].lower()
            ):
                score += 2
            elif (
                "fromage frais" in cheese_type_lower
                and "frais" in recipe["title"].lower()
            ):
                score += 2

            filtered.append(
                {**recipe, "score": min(10, score)}  # Limiter le score à 10
            )

        # Trier par score et limiter
        filtered.sort(key=lambda x: x["score"], reverse=True)
        return filtered[:3]

    def _clean_description(self, description: str) -> str:
        """Nettoie et formate la description"""
        if not description:
            return "Description non disponible"

        # Limiter la longueur
        if len(description) > 200:
            description = description[:200] + "..."

        # Supprimer les caractères bizarres
        description = description.replace("\n", " ").replace("\r", " ")
        description = " ".join(description.split())  # Nettoyer espaces multiples

        return description

    def _extract_domain(self, url: str) -> str:
        """Extrait le nom de domaine d'une URL"""
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            domain = parsed.netloc
            # Retirer 'www.' et garder le domaine principal
            domain = domain.replace("www.", "")
            # Prendre seulement le domaine de base
            if "." in domain:
                parts = domain.split(".")
                if len(parts) >= 2:
                    domain = f"{parts[-2]}.{parts[-1]}"
            return domain
        except:
            return "web"

    def _init_knowledge(self):
        """Base de connaissances fromage intégrée"""
        return {
            "types_pate": {
                "Fromage frais": {
                    "description": "Non affiné, humide, à consommer rapidement",
                    "exemples": "Fromage blanc, faisselle, ricotta, cottage cheese",
                    "duree": "0-3 jours",
                    "difficulte": "Facile - Idéal débutants",
                },
                "Pâte molle": {
                    "description": "Croûte fleurie ou lavée, texture crémeuse",
                    "exemples": "Camembert, brie, munster, reblochon",
                    "duree": "2-8 semaines",
                    "difficulte": "Moyenne - Nécessite une cave",
                },
                "Pâte pressée non cuite": {
                    "description": "Pressée sans cuisson, texture ferme",
                    "exemples": "Cantal, saint-nectaire, morbier, tomme",
                    "duree": "1-6 mois",
                    "difficulte": "Moyenne - Matériel spécifique",
                },
                "Pâte pressée cuite": {
                    "description": "Caillé chauffé puis pressé, longue conservation",
                    "exemples": "Comté, gruyère, beaufort, parmesan",
                    "duree": "3-36 mois",
                    "difficulte": "Difficile - Expertise requise",
                },
                "Pâte persillée": {
                    "description": "Avec moisissures bleues, goût prononcé",
                    "exemples": "Roquefort, bleu d'Auvergne, gorgonzola, stilton",
                    "duree": "2-6 mois",
                    "difficulte": "Difficile - Contrôle précis",
                },
            },
            "ingredients_base": {
                "Lait": [
                    "Vache (doux)",
                    "Chèvre (acidulé)",
                    "Brebis (riche)",
                    "Bufflonne (crémeux)",
                    "Mélange",
                ],
                "Coagulant": [
                    "Présure animale",
                    "Présure végétale",
                    "Jus de citron",
                    "Vinaigre blanc",
                ],
                "Ferments": [
                    "Lactiques (yaourt)",
                    "Mésophiles (température ambiante)",
                    "Thermophiles (haute température)",
                ],
                "Sel": ["Sel fin", "Gros sel", "Sel de mer", "Saumure (eau + sel)"],
                "Affinage": [
                    "Penicillium roqueforti (bleu)",
                    "Geotrichum (croûte)",
                    "Herbes",
                    "Cendres",
                ],
            },
            "epices_et_aromates": {
                "Herbes fraîches": [
                    "Basilic (doux, fromages frais)",
                    "Ciboulette (léger, fromages de chèvre)",
                    "Thym (robuste, tommes)",
                    "Romarin (puissant, pâtes pressées)",
                    "Persil (neutre, universel)",
                    "Aneth (anisé, fromages nordiques)",
                    "Menthe (rafraîchissant, fromages méditerranéens)",
                    "Coriandre (exotique, fromages épicés)",
                ],
                "Herbes séchées": [
                    "Herbes de Provence (mélange classique)",
                    "Origan (italien, fromages à pizza)",
                    "Sarriette (poivrée, fromages de montagne)",
                    "Estragon (anisé, fromages frais)",
                    "Laurier (dans saumure)",
                    "Sauge (forte, pâtes dures)",
                ],
                "Épices chaudes": [
                    "Poivre noir (concassé ou moulu)",
                    "Poivre rouge (Espelette, piment doux)",
                    "Paprika (fumé ou doux)",
                    "Cumin (terreux, fromages orientaux)",
                    "Curry (mélange, fromages fusion)",
                    "Piment de Cayenne (fort, avec modération)",
                    "Ras el hanout (complexe, fromages marocains)",
                ],
                "Épices douces": [
                    "Nigelle (sésame noir, fromages levantins)",
                    "Graines de fenouil (anisées)",
                    "Graines de carvi (pain, fromages nordiques)",
                    "Fenugrec (sirop d'érable, rare)",
                    "Coriandre en graines (agrumes)",
                ],
                "Fleurs et pollen": [
                    "Lavande (Provence, délicat)",
                    "Safran (luxueux, fromages d'exception)",
                    "Pétales de rose (persan, subtil)",
                    "Bleuet (visuel, doux)",
                    "Pollen de fleurs (sauvage)",
                ],
                "Aromates spéciaux": [
                    "Ail frais (haché ou confit)",
                    "Échalote (finement ciselée)",
                    "Oignon rouge (mariné)",
                    "Gingembre (frais râpé, fusion)",
                    "Citronnelle (asiatique, rare)",
                    "Zeste d'agrumes (citron, orange, bergamote)",
                ],
                "Cendres et croûtes": [
                    "Cendres végétales (charbon de bois alimentaire)",
                    "Cendres de sarment de vigne",
                    "Charbon actif alimentaire (noir intense)",
                    "Foin séché (affinage sur foin)",
                    "Paille (affinage traditionnel)",
                ],
                "Accompagnements dans la pâte": [
                    "Noix concassées (texture)",
                    "Noisettes (doux, chèvre)",
                    "Pistaches (vert, raffiné)",
                    "Fruits secs (abricots, figues)",
                    "Olives (noires ou vertes)",
                    "Tomates séchées (umami)",
                    "Truffe (luxe absolu)",
                    "Champignons séchés (boisé)",
                ],
            },
            "techniques_aromatisation": {
                "Incorporation dans le caillé": "Ajouter les épices au moment du moulage pour distribution homogène",
                "Enrobage externe": "Rouler le fromage dans les épices après salage",
                "Affinage aromatisé": "Placer herbes/épices dans la cave d'affinage",
                "Saumure parfumée": "Infuser la saumure avec aromates",
                "Huile aromatisée": "Badigeonner la croûte d'huile aux herbes",
                "Couche intermédiaire": "Saupoudrer entre deux couches de caillé",
            },
            "dosages_recommandes": {
                "Herbes fraîches": "2-3 cuillères à soupe pour 1kg de fromage",
                "Herbes séchées": "1-2 cuillères à soupe pour 1kg",
                "Épices moulues": "1-2 cuillères à café pour 1kg",
                "Épices en grains": "1 cuillère à soupe concassée pour 1kg",
                "Ail/gingembre": "1-2 gousses/morceaux pour 1kg",
                "Zestes": "1 agrume entier pour 1kg",
                "Cendres": "Fine couche sur la croûte",
            },
            "associations_classiques": {
                "Fromage de chèvre": "Herbes de Provence, miel, lavande",
                "Brebis": "Piment d'Espelette, romarin, olives",
                "Pâte molle": "Ail, fines herbes, poivre",
                "Pâte pressée": "Cumin, fenugrec, noix",
                "Fromage frais": "Ciboulette, aneth, menthe fraîche",
                "Bleu": "Noix, figues, porto (pas dans le fromage)",
            },
            "temperatures_affinage": {
                "Fromage frais": "4-6°C (réfrigérateur)",
                "Pâte molle croûte fleurie": "10-12°C, 90-95% humidité",
                "Pâte molle croûte lavée": "12-14°C, 90-95% humidité",
                "Pâte pressée non cuite": "12-14°C, 85-90% humidité",
                "Pâte pressée cuite": "14-18°C, 85-90% humidité",
                "Pâte persillée": "8-10°C, 95% humidité",
                "Chèvre": "10-12°C, 80-85% humidité",
            },
            "problemes_courants": {
                "Caillé trop dur": "Trop de présure ou température trop haute. Solution : Réduire la dose de présure de 20%",
                "Pas de caillage": "Lait UHT (stérilisé) ou présure périmée. Solution : Utiliser du lait cru ou pasteurisé",
                "Caillé trop mou": "Pas assez de présure ou temps insuffisant. Solution : Attendre 15-30 min de plus",
                "Fromage trop acide": "Fermentation trop longue ou trop chaud. Solution : Réduire température ou temps d'affinage",
                "Fromage trop salé": "Excès de sel ou salage trop long. Solution : Utiliser 1,5% du poids au lieu de 2%",
                "Moisissures indésirables": "Humidité excessive ou mauvaise hygiène. Solution : Nettoyer la cave, réduire humidité",
                "Croûte craquelée": "Air trop sec. Solution : Augmenter humidité à 85-90%",
                "Fromage trop sec": "Égouttage excessif. Solution : Réduire temps d'égouttage de moitié",
                "Texture granuleuse": "Caillage incomplet ou découpe trop brutale. Solution : Attendre caillage complet",
                "Goût amer": "Sur-affinage ou contamination bactérienne. Solution : Réduire durée d'affinage",
                "Fromage coule": "Température trop élevée pendant affinage. Solution : Cave à 10-12°C maximum",
                "Yeux (trous) non désirés": "Fermentation gazeuse. Solution : Presser davantage pour éliminer l'air",
            },
            "conservation": {
                "Fromage frais": "3-5 jours au frigo (4°C) dans boîte hermétique",
                "Pâte molle jeune": "1-2 semaines au frigo dans papier fromagerie",
                "Pâte molle affinée": "2-3 semaines, sortir 1h avant dégustation",
                "Pâte pressée non cuite": "1-2 mois au frigo, bien emballer",
                "Pâte pressée cuite": "3-6 mois au frais (10-12°C), croûte protégée",
                "Pâte persillée": "3-4 semaines, papier alu pour limiter moisissures",
                "Chèvre frais": "1 semaine maximum au frigo",
                "Chèvre affiné": "2-3 semaines en cave ou frigo",
                "Conseil général": "Ne jamais congeler (texture détruite), emballer dans papier respirant",
            },
            "accords_vins": {
                "Fromage frais nature": "Vin blanc sec et vif (Muscadet, Picpoul de Pinet)",
                "Fromage frais aux herbes": "Blanc aromatique (Sauvignon, Riesling)",
                "Chèvre frais": "Sancerre, Pouilly-Fumé, Sauvignon blanc",
                "Chèvre sec": "Blanc minéral (Chablis) ou rouge léger (Pinot Noir)",
                "Brie, Camembert": "Champagne, Crémant, ou rouge léger (Beaujolais)",
                "Munster, Maroilles": "Blanc puissant (Gewurztraminer) ou bière",
                "Comté jeune": "Vin jaune du Jura, Chardonnay",
                "Comté vieux": "Vin jaune, Porto Tawny",
                "Cantal, Salers": "Rouge charpenté (Cahors, Madiran)",
                "Roquefort": "Blanc doux (Sauternes, Monbazillac) ou Porto",
                "Bleu d'Auvergne": "Rouge puissant (Côtes du Rhône) ou blanc moelleux",
                "Brebis des Pyrénées": "Rouge du Sud-Ouest (Irouléguy, Madiran)",
                "Morbier": "Vin blanc du Jura (Chardonnay)",
                "Reblochon": "Blanc de Savoie (Apremont, Chignin)",
                "Règle d'or": "Accord régional : fromage et vin de la même région",
            },
            "accords_mets": {
                "Fromage frais": "Pain complet, fruits rouges, miel, concombre",
                "Pâte molle": "Baguette fraîche, pommes, raisins, confiture de figues",
                "Pâte pressée": "Pain de campagne, noix, cornichons, charcuterie",
                "Pâte persillée": "Pain aux noix, poire, miel de châtaignier, céleri",
                "Chèvre": "Pain grillé, miel, salade verte, betterave",
                "Fromages forts": "Pain de seigle, oignon confit, pomme de terre",
            },
            "regles_compatibilite": {
                "lait_x_type_pate": {
                    "description": "Associations valides entre types de lait et types de pâte",
                    "combinaisons_valides": [
                        {
                            "lait": "vache",
                            "types_pate_compatibles": [
                                "Fromage frais",
                                "Pâte molle",
                                "Pâte pressée non cuite",
                                "Pâte pressée cuite",
                                "Pâte persillée",
                            ],
                            "exemples": ["camembert", "brie", "comté", "roquefort"],
                        },
                        {
                            "lait": "chevre",
                            "types_pate_compatibles": [
                                "Fromage frais",
                                "Pâte pressée non cuite",
                            ],
                            "types_pate_incompatibles": ["Pâte molle"],
                            "raison": "Le lait de chèvre donne naturellement une croûte cendrée/naturelle, pas de croûte fleurie",
                            "exemples": [
                                "crottin de Chavignol",
                                "sainte-maure",
                                "tomme de chèvre",
                            ],
                        },
                        {
                            "lait": "brebis",
                            "types_pate_compatibles": [
                                "Fromage frais",
                                "Pâte pressée non cuite",
                                "Pâte pressée cuite",
                                "Pâte persillée",
                            ],
                            "types_pate_incompatibles": ["Pâte molle"],
                            "raison": "La brebis est traditionnellement utilisée pour fromages pressés ou bleus, pas pour croûtes fleuries",
                            "exemples": [
                                "roquefort",
                                "ossau-iraty",
                                "manchego",
                                "pecorino",
                            ],
                        },
                        {
                            "lait": "bufflonne",
                            "types_pate_compatibles": ["Fromage frais"],
                            "types_pate_incompatibles": [
                                "Pâte molle",
                                "Pâte pressée cuite",
                            ],
                            "raison": "Lait très riche utilisé principalement pour fromages frais italiens",
                            "exemples": ["mozzarella di bufala", "burrata"],
                        },
                    ],
                },
                "lait_x_aromates": {
                    "description": "Associations classiques et harmonieuses",
                    "affinites": [
                        {
                            "lait": "chevre",
                            "aromates_recommandes": [
                                "herbes de Provence",
                                "miel",
                                "lavande",
                                "thym",
                                "cendre",
                            ],
                            "aromates_deconseilles": ["curry fort", "cumin intense"],
                            "raison": "Le chèvre a un goût délicat qui peut être écrasé par épices trop fortes",
                        },
                        {
                            "lait": "brebis",
                            "aromates_recommandes": [
                                "piment d'Espelette",
                                "romarin",
                                "olives",
                                "tomates séchées",
                            ],
                            "aromates_deconseilles": [],
                            "raison": "Goût prononcé de brebis supporte bien épices méditerranéennes fortes",
                        },
                        {
                            "lait": "vache",
                            "aromates_recommandes": [
                                "ail",
                                "fines herbes",
                                "poivre",
                                "noix",
                                "cumin",
                            ],
                            "aromates_deconseilles": [],
                            "raison": "Neutre, s'accommode de presque tout",
                        },
                    ],
                },
                "type_pate_x_aromates": {
                    "Fromage frais": {
                        "aromates_compatibles": [
                            "herbes fraîches",
                            "ail frais",
                            "ciboulette",
                            "aneth",
                            "menthe",
                        ],
                        "aromates_incompatibles": [
                            "épices chaudes fortes",
                            "curry",
                            "piment de Cayenne",
                        ],
                        "raison": "Goût délicat, consommation rapide : herbes fraîches idéales",
                    },
                    "Pâte molle": {
                        "aromates_compatibles": [
                            "herbes séchées",
                            "poivre",
                            "ail confit",
                        ],
                        "aromates_incompatibles": ["herbes fraîches"],
                        "raison": "Affinage humide : herbes fraîches peuvent pourrir, préférer séchées",
                    },
                    "Pâte pressée non cuite": {
                        "aromates_compatibles": [
                            "cumin",
                            "fenugrec",
                            "noix",
                            "fruits secs",
                            "épices en grains",
                        ],
                        "aromates_incompatibles": ["herbes fraîches délicates"],
                        "raison": "Longue conservation : épices robustes et séchées résistent mieux",
                    },
                    "Pâte pressée cuite": {
                        "aromates_compatibles": ["cumin", "noix", "fruits secs"],
                        "aromates_incompatibles": ["herbes fraîches"],
                        "raison": "Très long affinage : seules épices robustes survivent",
                    },
                    "Pâte persillée": {
                        "aromates_compatibles": ["noix", "miel", "fruits secs"],
                        "aromates_incompatibles": [
                            "herbes fortes",
                            "épices puissantes",
                        ],
                        "raison": "Goût déjà très prononcé : accompagnements doux uniquement",
                    },
                },
                "exclusions_absolues": [
                    {
                        "combinaison": "lait:brebis + type_pate:Pâte molle",
                        "raison": "Incompatibilité traditionnelle et technique. La brebis ne développe pas bien le Penicillium camemberti",
                        "severite": "haute",
                        "alternatives": ["Pâte pressée non cuite", "Pâte persillée"],
                    },
                    {
                        "combinaison": "lait:chevre + type_pate:Pâte molle",
                        "raison": "Chèvre développe naturellement croûte cendrée, pas fleurie comme camembert",
                        "severite": "haute",
                        "alternatives": ["Fromage frais", "Pâte pressée non cuite"],
                    },
                    {
                        "combinaison": "type_pate:Fromage frais + aromate:herbes séchées fortes",
                        "raison": "Déséquilibre gustatif - fromage frais trop délicat",
                        "severite": "moyenne",
                        "alternatives": ["Herbes fraîches", "herbes séchées douces"],
                    },
                    {
                        "combinaison": "affinage:long + aromate:herbes fraîches",
                        "raison": "Risque sanitaire - les herbes fraîches moisissent pendant affinage humide",
                        "severite": "haute",
                        "alternatives": ["Herbes séchées", "aromates après affinage"],
                    },
                ],
            },
            "materiel_indispensable": {
                "Pour débuter": [
                    "Thermomètre de cuisson (précision ±1°C) - 10-15€",
                    "Grande casserole inox 3-5L - 20-30€",
                    "Moule à fromage perforé 500g - 5-10€",
                    "Étamine/mousseline (toile à fromage) - 5€",
                    "Louche et couteau long - 10€",
                ],
                "Pour progresser": [
                    "Hygromètre pour cave (mesure humidité) - 15-20€",
                    "Presse à fromage - 50-100€",
                    "Set de moules variés - 30-50€",
                    "pH-mètre - 30-50€",
                    "Claie d'affinage en bois - 20-40€",
                ],
                "Pour expert": [
                    "Cave d'affinage électrique - 300-800€",
                    "Trancheuse à caillé professionnelle - 100€",
                    "Balance de précision 0.1g - 30€",
                    "Kit de cultures spécifiques - 50€/an",
                ],
            },
            "fournisseurs_recommandes": {
                "Présure et ferments": "Tom Press, Ferments-et-vous.com, Fromage-maison.com",
                "Matériel": "Tom Press (FR), Fromag'Home, Le Parfait",
                "Moules": "Amazon, Tom Press, magasins cuisine spécialisés",
                "Lait cru": "Producteurs locaux, AMAP, marchés fermiers",
                "Livres": '"Fromages et laitages naturels faits maison" de Marie-Claire Frédéric',
            },
            "calendrier_fromager": {
                "Printemps (Mars-Mai)": "Saison idéale pour chèvre (lait riche). Fromages frais, chèvre frais",
                "Été (Juin-Août)": "Éviter pâtes molles (chaleur). Privilégier fromages frais, ricotta",
                "Automne (Sept-Nov)": "Excellente période pour tous types. Lancer affinage pour Noël",
                "Hiver (Déc-Fév)": "Fromages d'affinage, pâtes pressées. Cave naturellement fraîche",
            },
            "profils_utilisateurs": {
                "🧀 Amateur": {
                    "description": "Débutant, usage familial, matériel limité",
                    "niveau": "débutant",
                    "objectifs": [
                        "Apprendre les bases",
                        "Réussir simplement",
                        "Goûter rapidement",
                    ],
                    "contraintes": ["Matériel basique", "Temps limité", "Budget serré"],
                    "ton": "Encourageant, pédagogique, rassurant",
                    "termes": "Vocabulaire simple, explications détaillées",
                    "equipement": [
                        "Casserole standard",
                        "Thermomètre basique",
                        "Moule simple",
                    ],
                    "complexite": "Recettes en 3-5 étapes max",
                    "duree_max": "24-48h maximum",
                    "budget": "Économique (moins de 20€)",
                    "quantites": "Petites quantités (500g-1kg)",
                    "focus": "Succès rapide, plaisir immédiat",
                },
                "🏭 Producteur": {
                    "description": "Professionnel ou semi-pro, recherche de qualité",
                    "niveau": "expert",
                    "objectifs": [
                        "Rendement optimal",
                        "Qualité constante",
                        "Commercialisation",
                    ],
                    "contraintes": ["Normes sanitaires", "Traçabilité", "Rentabilité"],
                    "ton": "Technique, précis, professionnel",
                    "termes": "Vocabulaire professionnel, normes, certifications",
                    "equipement": [
                        "Matériel pro",
                        "Hygromètre",
                        "pH-mètre",
                        "Cave d'affinage",
                    ],
                    "complexite": "Recettes détaillées avec paramètres précis",
                    "duree_max": "Plusieurs semaines/mois",
                    "budget": "Investissement justifié",
                    "quantites": "Grandes quantités (5-20kg)",
                    "focus": "Qualité optimale, reproductibilité",
                },
                "🎓 Formateur": {
                    "description": "Enseignant, animateur, partage de savoir",
                    "niveau": "intermédiaire",
                    "objectifs": [
                        "Transmettre",
                        "Expliquer les concepts",
                        "Anticiper les erreurs",
                    ],
                    "contraintes": ["Pédagogie", "Clarté", "Sécurité"],
                    "ton": "Pédagogique, structuré, anticipatif",
                    "termes": "Explications conceptuelles, métaphores, illustrations",
                    "equipement": ["Matériel pédagogique", "Supports visuels"],
                    "complexite": "Étapes décomposées, points d'attention",
                    "duree_max": "Adaptable aux sessions",
                    "budget": "Variable selon public",
                    "quantites": "Quantités adaptées à la démonstration",
                    "focus": "Compréhension, expérimentation, apprentissage",
                },
            },
            "adaptations_par_profil": {
                "🧀 Amateur": {
                    "introduction": "✨ **RECETTE SIMPLIFIÉE POUR DÉBUTANT** ✨\n\n*Conseil du chef : Commencez simple, la fromagerie s'apprend en douceur !*",
                    "etapes": [
                        "Explications très détaillées",
                        "Astuces anti-échec",
                        "Photos mentales",
                    ],
                    "materiel": "🔧 **Matériel vraiment indispensable :**\n- Une grande casserole\n- Un thermomètre\n- Un torchon propre\n- Un moule (un saladier percé peut faire l'affaire !)",
                    "ingredients": "🥛 **Ingrédients faciles à trouver :**\nEn grande surface ou chez votre producteur local",
                    "conseils": [
                        "Ne vous précipitez pas !",
                        "Si ça ne marche pas du premier coup, c'est normal.",
                        "Goûtez à chaque étape pour comprendre l'évolution.",
                    ],
                },
                "🏭 Producteur": {
                    "introduction": "📊 **FICHE TECHNIQUE PROFESSIONNELLE**\n\n*Pour une production de qualité constante*",
                    "etapes": [
                        "Procédures standardisées",
                        "Points de contrôle qualité",
                        "Mesures précises",
                    ],
                    "materiel": "🏭 **Équipement recommandé :**\n- Thermomètre de précision ±0.5°C\n- pH-mètre\n- Balance 0.1g\n- Cave à affinage contrôlée\n- Cahier de suivi de production",
                    "ingredients": "📦 **Spécifications techniques :**\n- Lait cru ou microfiltré\n- Ferments sélectionnés\n- Présure certifiée",
                    "conseils": [
                        "Documentez chaque batch",
                        "Calibrez vos instruments régulièrement",
                        "Formalisez vos procédures",
                    ],
                },
                "🎓 Formateur": {
                    "introduction": "📚 **SUPPORT PÉDAGOGIQUE COMPLET**\n\n*Pour animer un atelier fromager réussi*",
                    "etapes": [
                        "Objectifs pédagogiques",
                        "Erreurs courantes anticipées",
                        "Questions pour le groupe",
                    ],
                    "materiel": "🎓 **Matériel pédagogique :**\n- Tableau ou paperboard\n- Échantillons visuels\n- Fiches participants\n- Chronomètre pour les temps",
                    "ingredients": "🧪 **Pour la démonstration :**\n- Quantités adaptées au groupe\n- Variétés pour comparer\n- Échantillons d'étapes intermédiaires",
                    "conseils": [
                        "Préparez les questions à l'avance",
                        "Anticipez les blocages",
                        "Variez les supports (visuel, pratique, théorique)",
                    ],
                },
            },
        }

    def _download_history_from_hf(self):
        """Télécharge l'historique depuis HF Dataset"""
        if not self.api:
            print("⚠️  Pas de token HF - historique local uniquement")
            return

        try:
            downloaded_path = hf_hub_download(
                repo_id=self.hf_repo,
                filename=self.recipes_file,
                repo_type="dataset",
                token=self.hf_token,
            )

            with open(downloaded_path, "r", encoding="utf-8") as src:
                history = json.load(src)

            with open(self.recipes_file, "w", encoding="utf-8") as dst:
                json.dump(history, dst, indent=2, ensure_ascii=False)

            print(f"✅ Historique chargé : {len(history)} recettes")

        except Exception as e:
            print(f"ℹ️  Pas d'historique existant: {e}")
            with open(self.recipes_file, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _upload_history_to_hf(self):
        """Upload l'historique vers HF Dataset"""
        if not self.api:
            print("⚠️  Pas de token HF - sauvegarde locale uniquement")
            return False

        try:
            self.api.upload_file(
                path_or_fileobj=self.recipes_file,
                path_in_repo=self.recipes_file,
                repo_id=self.hf_repo,
                repo_type="dataset",
                commit_message=f"Update: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            )
            print("✅ Historique synchronisé avec HF")
            return True
        except Exception as e:
            print(f"❌ Erreur upload HF: {e}")
            return False

    def get_history(self):
        """Retourne l'historique complet"""
        try:
            if os.path.exists(self.recipes_file):
                with open(self.recipes_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"❌ Erreur get_history: {e}")
            return []

    def _save_to_history(self, ingredients, cheese_type, constraints, recipe):
        """Sauvegarde une recette dans l'historique"""
        try:
            history = self._load_history()

            recipe_lines = recipe.split("\n")
            cheese_name = "Fromage personnalisé"
            for line in recipe_lines:
                if "🧀" in line and len(line) < 100:
                    cheese_name = (
                        line.replace("🧀", "").replace("═", "").replace("║", "").strip()
                    )
                    break

            entry = {
                "id": len(history) + 1,
                "date": datetime.now().isoformat(),
                "cheese_name": cheese_name,
                "ingredients": ingredients,
                "type": cheese_type,
                "constraints": constraints,
                "recipe_complete": recipe,
                "recipe_preview": recipe[:300] + "..." if len(recipe) > 300 else recipe,
            }

            history.append(entry)

            # Sauvegarder localement
            with open(self.recipes_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

            # ✅ AJOUTER CETTE LIGNE : Mettre à jour l'historique en mémoire
            self.history = history

            # Upload vers HF
            sync_success = self._upload_history_to_hf()

            if sync_success:
                print(f"✅ Recette #{entry['id']} sauvegardée et synchronisée")
            else:
                print(f"⚠️  Recette #{entry['id']} sauvegardée localement")

            return True

        except Exception as e:
            print(f"❌ Erreur sauvegarde: {e}")
            return False

    def get_knowledge_summary(self):
        """Retourne un résumé complet de la base de connaissances"""
        summary = "📚 BASE DE CONNAISSANCES FROMAGE COMPLÈTE\n\n"

        # Types de pâte
        summary += "🧀 TYPES DE PÂTE :\n"
        summary += "=" * 70 + "\n\n"

        for name, info in self.knowledge_base["types_pate"].items():
            summary += f"• {name.upper()}\n"
            summary += f"  {info['description']}\n"
            summary += f"  Exemples : {info['exemples']}\n"
            summary += (
                f"  Durée : {info['duree']} | Difficulté : {info['difficulte']}\n\n"
            )

        # Ingrédients de base
        summary += "\n" + "=" * 70 + "\n"
        summary += "🥛 INGRÉDIENTS ESSENTIELS :\n"
        summary += "=" * 70 + "\n\n"

        for category, items in self.knowledge_base["ingredients_base"].items():
            summary += f"\n• {category.upper()} :\n"
            for item in items:
                summary += f"  - {item}\n"

        # Épices et aromates
        if "epices_et_aromates" in self.knowledge_base:
            summary += "\n" + "=" * 70 + "\n"
            summary += "🌶️ ÉPICES ET AROMATES :\n"
            summary += "=" * 70 + "\n\n"

            for category, items in self.knowledge_base["epices_et_aromates"].items():
                summary += f"• {category.upper()} :\n"
                for item in items[:5]:
                    summary += f"  - {item}\n"
                if len(items) > 5:
                    summary += f"  ... et {len(items)-5} autres\n"
                summary += "\n"

        # Techniques d'aromatisation
        if "techniques_aromatisation" in self.knowledge_base:
            summary += "\n" + "=" * 70 + "\n"
            summary += "🎨 TECHNIQUES D'AROMATISATION :\n"
            summary += "=" * 70 + "\n\n"

            for tech, desc in self.knowledge_base["techniques_aromatisation"].items():
                summary += f"• {tech} :\n  {desc}\n\n"

        # Dosages recommandés
        if "dosages_recommandes" in self.knowledge_base:
            summary += "\n" + "=" * 70 + "\n"
            summary += "📐 DOSAGES RECOMMANDÉS :\n"
            summary += "=" * 70 + "\n\n"

            for ingredient, dosage in self.knowledge_base[
                "dosages_recommandes"
            ].items():
                summary += f"• {ingredient} : {dosage}\n"

        # Associations classiques
        if "associations_classiques" in self.knowledge_base:
            summary += "\n" + "=" * 70 + "\n"
            summary += "🎯 ASSOCIATIONS CLASSIQUES :\n"
            summary += "=" * 70 + "\n\n"

            for fromage, assoc in self.knowledge_base[
                "associations_classiques"
            ].items():
                summary += f"• {fromage} : {assoc}\n"

        # Températures d'affinage
        if "temperatures_affinage" in self.knowledge_base:
            summary += "\n" + "=" * 70 + "\n"
            summary += "🌡️ TEMPÉRATURES D'AFFINAGE :\n"
            summary += "=" * 70 + "\n\n"

            for fromage_type, temp in self.knowledge_base[
                "temperatures_affinage"
            ].items():
                summary += f"• {fromage_type} : {temp}\n"

        # Problèmes courants
        if "problemes_courants" in self.knowledge_base:
            summary += "\n" + "=" * 70 + "\n"
            summary += "🚨 PROBLÈMES COURANTS ET SOLUTIONS :\n"
            summary += "=" * 70 + "\n\n"

            for probleme, solution in list(
                self.knowledge_base["problemes_courants"].items()
            )[:8]:
                summary += f"❌ {probleme}\n"
                summary += f"   ✅ {solution}\n\n"

            remaining = len(self.knowledge_base["problemes_courants"]) - 8
            if remaining > 0:
                summary += f"... et {remaining} autres problèmes documentés\n"

        # Conservation
        if "conservation" in self.knowledge_base:
            summary += "\n" + "=" * 70 + "\n"
            summary += "📦 CONSERVATION :\n"
            summary += "=" * 70 + "\n\n"

            for fromage_type, duree in self.knowledge_base["conservation"].items():
                summary += f"• {fromage_type} : {duree}\n"

        # Accords vins
        if "accords_vins" in self.knowledge_base:
            summary += "\n" + "=" * 70 + "\n"
            summary += "🍷 ACCORDS VINS :\n"
            summary += "=" * 70 + "\n\n"

            for fromage_type, vin in list(self.knowledge_base["accords_vins"].items())[
                :12
            ]:
                summary += f"• {fromage_type} → {vin}\n"

            remaining = len(self.knowledge_base["accords_vins"]) - 12
            if remaining > 0:
                summary += f"\n... et {remaining} autres accords\n"

        # Accords mets
        if "accords_mets" in self.knowledge_base:
            summary += "\n" + "=" * 70 + "\n"
            summary += "🍽️ ACCORDS METS :\n"
            summary += "=" * 70 + "\n\n"

            for fromage_type, mets in self.knowledge_base["accords_mets"].items():
                summary += f"• {fromage_type} : {mets}\n"

        # Matériel indispensable
        if "materiel_indispensable" in self.knowledge_base:
            summary += "\n" + "=" * 70 + "\n"
            summary += "🛠️ MATÉRIEL RECOMMANDÉ :\n"
            summary += "=" * 70 + "\n\n"

            for niveau, items in self.knowledge_base["materiel_indispensable"].items():
                summary += f"\n📌 {niveau.upper()} :\n"
                for item in items:
                    summary += f"  - {item}\n"

        # Fournisseurs recommandés
        if "fournisseurs_recommandes" in self.knowledge_base:
            summary += "\n" + "=" * 70 + "\n"
            summary += "🏪 FOURNISSEURS RECOMMANDÉS :\n"
            summary += "=" * 70 + "\n\n"

            for cat, fournisseurs in self.knowledge_base[
                "fournisseurs_recommandes"
            ].items():
                summary += f"• {cat} : {fournisseurs}\n"

        # Calendrier fromager
        if "calendrier_fromager" in self.knowledge_base:
            summary += "\n" + "=" * 70 + "\n"
            summary += "📅 CALENDRIER FROMAGER :\n"
            summary += "=" * 70 + "\n\n"

            for saison, conseil in self.knowledge_base["calendrier_fromager"].items():
                summary += f"• {saison} :\n  {conseil}\n\n"

        # Conseils généraux
        summary += "\n" + "=" * 70 + "\n"
        summary += "💡 CONSEILS GÉNÉRAUX DU MAÎTRE FROMAGER :\n"
        summary += "=" * 70 + "\n\n"
        summary += "✨ Hygiène irréprochable : stériliser tout le matériel à l'eau bouillante\n"
        summary += "✨ Température précise : ±2°C peut totalement changer le résultat\n"
        summary += (
            "✨ Patience : un bon fromage ne se précipite pas, respecter les temps\n"
        )
        summary += "✨ Qualité du lait : préférer lait cru ou pasteurisé (JAMAIS UHT)\n"
        summary += "✨ Tenir un carnet : noter températures, durées et résultats\n"
        summary += "✨ Commencer simple : fromage frais avant pâtes pressées\n"
        summary += (
            "✨ Cave d'affinage DIY : Une glacière + bol d'eau + hygromètre suffit\n"
        )
        summary += "✨ Le petit-lait est précieux : pain, ricotta, plantes\n\n"

        # Statistiques
        summary += "=" * 70 + "\n"
        summary += "📊 STATISTIQUES DE LA BASE DE CONNAISSANCES :\n"
        summary += "=" * 70 + "\n"
        summary += f"• Types de pâte documentés : {len(self.knowledge_base.get('types_pate', {}))}\n"
        summary += f"• Catégories d'ingrédients : {len(self.knowledge_base.get('ingredients_base', {}))}\n"
        if "epices_et_aromates" in self.knowledge_base:
            summary += f"• Catégories d'épices : {len(self.knowledge_base['epices_et_aromates'])}\n"
            total_epices = sum(
                len(items)
                for items in self.knowledge_base["epices_et_aromates"].values()
            )
            summary += f"• Total épices/aromates : {total_epices}\n"
        summary += f"• Températures d'affinage : {len(self.knowledge_base.get('temperatures_affinage', {}))}\n"
        summary += f"• Problèmes documentés : {len(self.knowledge_base.get('problemes_courants', {}))}\n"
        summary += f"• Infos conservation : {len(self.knowledge_base.get('conservation', {}))}\n"
        summary += (
            f"• Accords vins : {len(self.knowledge_base.get('accords_vins', {}))}\n"
        )
        summary += (
            f"• Accords mets : {len(self.knowledge_base.get('accords_mets', {}))}\n"
        )
        summary += f"• Techniques d'aromatisation : {len(self.knowledge_base.get('techniques_aromatisation', {}))}\n"
        summary += (
            "\n🎉 Base de connaissances très complète pour devenir maître fromager !\n"
        )

        return summary

    def get_history_display(self):
        """Retourne l'historique formaté pour affichage"""
        try:
            history = self._load_history()

            if not history:
                return "📭 Aucune recette sauvegardée pour le moment."

            display = f"📚 **{len(history)} recette(s) sauvegardée(s)**\n\n"
            display += "---\n\n"

            for entry in reversed(history[-10:]):  # 10 dernières recettes
                display += f"**#{entry['id']}** | 📅 {entry['timestamp']}\n"
                display += f"🧀 Type: {entry['cheese_type']}\n"

                ing = entry["ingredients"]
                if isinstance(ing, list):
                    ing = ", ".join(str(i) for i in ing)  # ✅ CORRECT !
                elif isinstance(ing, str):
                    ing = ing[:50]  # Limite si déjà string

                display += f"🥛 Ingrédients: {ing[:50]}...\n"

                if entry.get("constraints"):
                    display += f"⚙️ Contraintes: {entry['constraints']}\n"

                display += "\n---\n\n"

            return display

        except Exception as e:
            return f"❌ Erreur lecture historique: {e}"

    def clear_history(self):
        """Efface tout l'historique"""
        try:
            with open(self.recipes_file, "w", encoding="utf-8") as f:
                json.dump([], f)

            # ✅ AJOUTER CETTE LIGNE
            self.history = []

            if self.api:
                self._upload_history_to_hf()
                return "✅ Historique effacé (local + HF) !"
            else:
                return "✅ Historique local effacé"

        except Exception as e:
            return f"❌ Erreur: {e}"

    # vérification connexion internet
    def test_internet(self):
        """Test si Internet fonctionne"""
        try:
            import requests

            response = requests.get("https://httpbin.org/get", timeout=10)
            return f"✅ Internet fonctionne !\n\nStatus: {response.status_code}\nURL testée: https://httpbin.org/get"
        except Exception as e:
            return f"❌ Erreur d'accès Internet:\n{str(e)}"

    def search_web_recipes(
        self, ingredients: str, cheese_type: str, max_results: int = 6
    ) -> list:
        """Recherche RÉELLE sur le web pour des recettes de fromage"""

        print(f"🔍 RECHERCHE RÉELLE WEB: {ingredients}")

        all_recipes = []

        try:
            from urllib.parse import quote

            # Construire une requête optimisée
            query = self._build_search_query(ingredients, cheese_type)
            print(f"📝 Requête: {query}")

            # ===== 1. ESSAYER SERPAPI (si clé disponible) =====
            serpapi_results = self._try_serpapi_search(query, max_results)
            if serpapi_results:
                all_recipes.extend(serpapi_results)
                print(f"✅ SerpAPI: {len(serpapi_results)} résultats")

            # ===== 2. ESSAYER CUSTOM SEARCH JSON API (Google) =====
            google_results = self._try_google_custom_search(query, max_results)
            if google_results:
                all_recipes.extend(google_results)
                print(f"✅ Google Custom Search: {len(google_results)} résultats")

            # ===== 3. ESSAYER DUCKDUCKGO HTML (fallback) =====
            if len(all_recipes) < max_results:
                ddg_results = self._try_duckduckgo_html(
                    query, max_results - len(all_recipes)
                )
                if ddg_results:
                    all_recipes.extend(ddg_results)
                    print(f"✅ DuckDuckGo HTML: {len(ddg_results)} résultats")

            # ===== 4. TRAITEMENT DES RÉSULTATS =====
            if all_recipes:
                # Filtrer et nettoyer
                cleaned = self._clean_web_results(all_recipes, ingredients)

                # Prendre les meilleurs
                final = cleaned[:max_results]

                print(f"🎯 TOTAL: {len(final)} résultats RÉELS du web")

                # Afficher pour debug
                for i, r in enumerate(final, 1):
                    print(
                        f"   {i}. {r.get('title', '')[:60]}... ({r.get('source', '?')})"
                    )

                return final

            # ===== 5. SI AUCUN RÉSULTAT =====
            print("⚠️ Aucun résultat web trouvé")
            return self._get_fallback_with_real_urls(
                ingredients, cheese_type, max_results
            )

        except Exception as e:
            print(f"❌ Erreur recherche web: {e}")
            import traceback

            traceback.print_exc()
            return self._get_fallback_with_real_urls(
                ingredients, cheese_type, max_results
            )

    def _build_search_query(self, ingredients, cheese_type):
        """Construit une requête SIMPLE et EFFICACE pour DuckDuckGo"""

        # 1. Détecter le lait
        lait_detecte = self._detect_lait_from_ingredients(ingredients)

        # 2. Extraire les aromates principaux
        ing_list = [i.strip().lower() for i in ingredients.split(",")]
        aromates = []
        aromates_list = [
            "thym",
            "romarin",
            "basilic",
            "origan",
            "ail",
            "poivre",
            "cumin",
            "herbes",
        ]

        for ing in ing_list:
            for aromate in aromates_list:
                if aromate in ing:
                    aromates.append(aromate)

        # 3. Construire requête SIMPLE comme un humain
        query_parts = []

        # Type de fromage basique
        if cheese_type and cheese_type != "Laissez l'IA choisir":
            if "frais" in cheese_type.lower():
                query_parts.append("fromage frais")
            elif "pressée" in cheese_type.lower():
                query_parts.append("fromage à pâte pressée")
            else:
                query_parts.append("fromage")
        else:
            query_parts.append("fromage")

        # Ajouter lait si détecté
        if lait_detecte:
            query_parts.append(lait_detecte)

        # Ajouter aromates (max 2)
        for aromate in aromates[:2]:
            query_parts.append(aromate)

        # Ajouter "recette" ou "faire maison"
        query_parts.append("recette")

        query = " ".join(query_parts)

        # 4. Log pour debug
        print(f"🔍 Requête construite: '{query}'")
        print(f"   Détails: lait={lait_detecte}, aromates={aromates}")

        return query

    def _detect_lait_from_ingredients(self, ingredients):
        """Détecte SIMPLEMENT le type de lait"""
        if not ingredients:
            return None

        ingredients_lower = ingredients.lower()

        # Recherche directe
        if "brebis" in ingredients_lower:
            return "brebis"
        elif "chèvre" in ingredients_lower or "chevre" in ingredients_lower:
            return "chèvre"
        elif "vache" in ingredients_lower:
            return "vache"
        elif "bufflonne" in ingredients_lower:
            return "bufflonne"

        # Recherche dans "lait de X"
        if "lait de brebis" in ingredients_lower:
            return "brebis"
        elif "lait de chèvre" in ingredients_lower:
            return "chèvre"
        elif "lait de vache" in ingredients_lower:
            return "vache"

        return None

    def _try_serpapi_search(self, query, max_results):
        """Utilise SerpAPI (nécessite clé API)"""
        try:
            serpapi_key = os.environ.get("SERPAPI_KEY")
            if not serpapi_key:
                print("   ⚠️ SerpAPI: pas de clé API définie")
                return []

            import requests

            params = {
                "engine": "google",
                "q": query,
                "api_key": serpapi_key,
                "hl": "fr",
                "gl": "fr",
                "num": max_results,
            }

            response = requests.get(
                "https://serpapi.com/search", params=params, timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                recipes = []

                if "organic_results" in data:
                    for result in data["organic_results"][:max_results]:
                        recipes.append(
                            {
                                "title": result.get("title", ""),
                                "url": result.get("link", ""),
                                "description": result.get("snippet", ""),
                                "source": self._extract_domain(result.get("link", "")),
                                "score": 9,
                                "engine": "serpapi",
                            }
                        )

                return recipes

        except Exception as e:
            print(f"   ⚠️ SerpAPI error: {e}")

        return []

    def _try_google_custom_search(self, query, max_results):
        """Utilise Google Custom Search JSON API"""
        try:
            google_api_key = os.environ.get("GOOGLE_API_KEY")
            google_cse_id = os.environ.get("GOOGLE_CSE_ID")

            if not google_api_key or not google_cse_id:
                print("   ⚠️ Google CSE: pas de clés API définies")
                return []

            import requests
            from urllib.parse import quote

            url = f"https://www.googleapis.com/customsearch/v1"
            params = {
                "key": google_api_key,
                "cx": google_cse_id,
                "q": query,
                "num": max_results,
                "hl": "fr",
                "gl": "fr",
            }

            response = requests.get(url, params=params, timeout=15)

            if response.status_code == 200:
                data = response.json()
                recipes = []

                if "items" in data:
                    for item in data["items"][:max_results]:
                        recipes.append(
                            {
                                "title": item.get("title", ""),
                                "url": item.get("link", ""),
                                "description": item.get("snippet", ""),
                                "source": self._extract_domain(item.get("link", "")),
                                "score": 9,
                                "engine": "google_cse",
                            }
                        )

                return recipes

        except Exception as e:
            print(f"   ⚠️ Google CSE error: {e}")

        return []

    def _try_duckduckgo_html(self, query, max_results):
        """Fallback: DuckDuckGo HTML scraping"""
        try:
            import requests
            from bs4 import BeautifulSoup
            from urllib.parse import quote
            import time

            url = f"https://html.duckduckgo.com/html/?q={quote(query)}"

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html",
                "Accept-Language": "fr-FR,fr;q=0.9",
            }

            # Attendre pour paraître humain
            time.sleep(2)

            response = requests.get(url, headers=headers, timeout=15)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                recipes = []

                # Chercher les résultats DDG
                results = soup.find_all("div", class_="result")

                for result in results[: max_results * 2]:
                    try:
                        # Titre
                        title_elem = result.find("a", class_="result__a")
                        if not title_elem:
                            continue

                        title = title_elem.get_text(strip=True)

                        # URL (DDG utilise des redirections)
                        url_elem = result.find("a", class_="result__url")
                        if not url_elem:
                            continue

                        ddg_url = url_elem.get("href", "")
                        if not ddg_url:
                            continue

                        # Nettoyer l'URL DDG
                        import re

                        if "uddg=" in ddg_url:
                            match = re.search(r"uddg=([^&]+)", ddg_url)
                            if match:
                                from urllib.parse import unquote

                                real_url = unquote(match.group(1))
                            else:
                                continue
                        else:
                            real_url = ddg_url

                        # Description
                        desc_elem = result.find("a", class_="result__snippet")
                        description = (
                            desc_elem.get_text(strip=True) if desc_elem else ""
                        )

                        # Filtrer par pertinence
                        if not any(
                            kw in title.lower()
                            for kw in ["fromage", "cheese", "recette"]
                        ):
                            continue

                        recipes.append(
                            {
                                "title": title[:100],
                                "url": real_url,
                                "description": description[:200],
                                "source": self._extract_domain(real_url),
                                "score": 6,
                                "engine": "ddg_html",
                            }
                        )

                    except Exception as e:
                        print(f"      ⚠️ DDG parse error: {e}")
                        continue

                return recipes

        except Exception as e:
            print(f"   ⚠️ DuckDuckGo error: {e}")

        return []

    def _clean_web_results(self, recipes, ingredients):
        """Nettoie et filtre les résultats web"""
        cleaned = []
        seen_urls = set()

        for recipe in recipes:
            try:
                # Vérifier les champs obligatoires
                if not recipe.get("title") or not recipe.get("url"):
                    continue

                # Normaliser URL
                norm_url = self._normalize_url(recipe["url"])
                if not norm_url:
                    continue

                # Éviter doublons
                if norm_url in seen_urls:
                    continue
                seen_urls.add(norm_url)

                # Vérifier pertinence avec les ingrédients
                recipe_text = (
                    recipe["title"] + " " + recipe.get("description", "")
                ).lower()
                ingredients_lower = ingredients.lower()

                score = recipe.get("score", 5)

                # Bonus pour correspondance
                for ing in ingredients_lower.split(","):
                    ing = ing.strip()
                    if len(ing) > 3 and ing in recipe_text:
                        score += 1

                recipe["score"] = min(10, score)

                cleaned.append(recipe)

            except Exception as e:
                print(f"⚠️ Clean error: {e}")
                continue

        # Trier par score
        cleaned.sort(key=lambda x: x.get("score", 0), reverse=True)

        return cleaned

    def _get_fallback_with_real_urls(self, ingredients, cheese_type, max_results):
        """Fallback avec de VRAIES URLs de sites de recettes"""
        print("🔄 Fallback avec URLs réelles...")

        # Sites réels de recettes de fromage
        real_recipes = [
            {
                "title": "Fromage frais maison facile",
                "url": "https://www.marmiton.org/recettes/recette_fromage-frais-maison_337338.aspx",
                "description": "Recette simple de fromage frais avec lait et présure",
                "source": "marmiton.org",
                "score": 8,
                "real": True,
            },
            {
                "title": "Recette de mozzarella maison",
                "url": "https://www.regal.fr/produit/fromage/recette-mozzarella-maison-100305",
                "description": "Mozzarella fraîche faite maison en quelques heures",
                "source": "regal.fr",
                "score": 8,
                "real": True,
            },
            {
                "title": "Fromage de chèvre débutant",
                "url": "https://www.750g.com/faire-son-fromage-de-chevre-maison-r152700.htm",
                "description": "Premiers pas dans la fabrication fromagère",
                "source": "750g.com",
                "score": 7,
                "real": True,
            },
            {
                "title": "Brie maison traditionnel",
                "url": "https://www.femmeactuelle.fr/cuisine/guides-cuisine/fromage-maison-213130",
                "description": "Brie à croûte fleurie fait maison",
                "source": "femmeactuelle.fr",
                "score": 7,
                "real": True,
            },
            {
                "title": "Fromage à pâte pressée",
                "url": "https://cuisine.journaldesfemmes.fr/recette/332154-fromage-pate-pressee",
                "description": "Techniques de pressage pour fromages durs",
                "source": "journaldesfemmes.fr",
                "score": 6,
                "real": True,
            },
            {
                "title": "Roquefort maison",
                "url": "https://www.lerustique.fr/recette-roquefort-maison",
                "description": "Fromage bleu de brebis persillé",
                "source": "lerustique.fr",
                "score": 6,
                "real": True,
            },
        ]

        # Filtrer par ingrédients si possible
        filtered = []
        ingredients_lower = ingredients.lower()

        for recipe in real_recipes:
            score = recipe["score"]
            title_lower = recipe["title"].lower()

            # Bonus pour correspondance
            if "brebis" in ingredients_lower and "brebis" in title_lower:
                score += 2
            elif "chèvre" in ingredients_lower and "chèvre" in title_lower:
                score += 2
            elif "vache" in ingredients_lower and any(
                x in title_lower for x in ["brie", "camembert", "comté"]
            ):
                score += 1

            filtered.append({**recipe, "score": min(10, score)})

        # Trier et limiter
        filtered.sort(key=lambda x: x["score"], reverse=True)

        return filtered[:max_results]

    def _get_smart_fallback(self, ingredients, cheese_type, max_results):
        """Fallback intelligent qui FILTRE par type de lait"""
        print(f"🧠 Fallback PERSONNALISÉ pour: {ingredients}")

        # Analyser PRÉCISÉMENT les ingrédients
        ing_list = [i.strip().lower() for i in ingredients.split(",")]

        # Détecter le type de lait EXACT
        lait_detecte = None
        lait_mots_cles = {
            "chèvre": ["chèvre", "chevre", "caprin", "goat"],
            "brebis": ["brebis", "mouton", "ovin", "sheep", "pecorino", "manchego"],
            "vache": ["vache", "bovin", "cow", "lait de vache", "comté", "camembert"],
            "bufflonne": ["bufflonne", "buffle", "buffalo", "mozzarella di bufala"],
        }

        for lait_type, mots_cles in lait_mots_cles.items():
            for mot in mots_cles:
                if any(mot in ing for ing in ing_list):
                    lait_detecte = lait_type
                    break
            if lait_detecte:
                break

        if not lait_detecte:
            # Par défaut, chercher "lait" dans la liste
            for ing in ing_list:
                if "lait" in ing:
                    if "chèvre" in ing or "chevre" in ing:
                        lait_detecte = "chèvre"
                    elif "brebis" in ing:
                        lait_detecte = "brebis"
                    elif "vache" in ing:
                        lait_detecte = "vache"
                    elif "bufflonne" in ing:
                        lait_detecte = "bufflonne"
                    break

        print(f"   🥛 Lait détecté: {lait_detecte or 'non spécifié'}")

        # Base de recettes ADAPTÉES par type de lait
        lait_specific_recipes = {
            "brebis": [
                {
                    "title": "Fromage de brebis des Pyrénées",
                    "url": "https://www.marmiton.org/recettes/recette_fromage-brebis-pyrenees_441229.aspx",
                    "description": "Fromage à pâte pressée de brebis façon Ossau-Iraty",
                    "source": "marmiton.org",
                    "score": 9,
                    "type": "brebis",
                },
                {
                    "title": "Recette de Manchego maison",
                    "url": "https://cuisine.journaldesfemmes.fr/recette/412345-manchego-maison",
                    "description": "Fromage espagnol de brebis à pâte pressée",
                    "source": "cuisine.journaldesfemmes.fr",
                    "score": 8,
                    "type": "brebis",
                },
                {
                    "title": "Pecorino romano artisanal",
                    "url": "https://www.750g.com/pecorino-romano-maison-r352700.htm",
                    "description": "Fromage de brebis italien à pâte dure",
                    "source": "750g.com",
                    "score": 8,
                    "type": "brebis",
                },
                {
                    "title": "Fromage de brebis crémeux",
                    "url": "https://www.regal.fr/produit/fromage/recette-brebis-cremeux-100615",
                    "description": "Fromage de brebis à pâte molle et crémeuse",
                    "source": "regal.fr",
                    "score": 7,
                    "type": "brebis",
                },
                {
                    "title": "Roquefort maison (brebis bleu)",
                    "url": "https://www.femmeactuelle.fr/cuisine/guides-cuisine/roquefort-maison-215430",
                    "description": "Fromage bleu de brebis persillé",
                    "source": "femmeactuelle.fr",
                    "score": 7,
                    "type": "brebis",
                },
            ],
            "chèvre": [
                {
                    "title": "Fromage de chèvre frais maison",
                    "url": "https://www.marmiton.org/recettes/recette_fromage-chevre-frais_337338.aspx",
                    "description": "Chèvre frais à déguster dans les 3 jours",
                    "source": "marmiton.org",
                    "score": 9,
                    "type": "chèvre",
                },
                {
                    "title": "Crottin de Chavignol artisanal",
                    "url": "https://cuisine.journaldesfemmes.fr/recette/315921-crottin-chavignol",
                    "description": "Crottin de chèvre affiné à la cendre",
                    "source": "cuisine.journaldesfemmes.fr",
                    "score": 8,
                    "type": "chèvre",
                },
                {
                    "title": "Bûche de chèvre aux herbes",
                    "url": "https://www.750g.com/buche-chevre-herbes-r252700.htm",
                    "description": "Bûche de chèvre roulée dans des herbes de Provence",
                    "source": "750g.com",
                    "score": 8,
                    "type": "chèvre",
                },
                {
                    "title": "Sainte-Maure de Touraine maison",
                    "url": "https://www.regal.fr/produit/fromage/recette-sainte-maure-100715",
                    "description": "Fromage de chèvre en bûche avec paille",
                    "source": "regal.fr",
                    "score": 7,
                    "type": "chèvre",
                },
            ],
            "vache": [
                {
                    "title": "Camembert normand maison",
                    "url": "https://www.marmiton.org/recettes/recette_camembert-maison_551229.aspx",
                    "description": "Camembert à croûte fleurie au lait de vache",
                    "source": "marmiton.org",
                    "score": 9,
                    "type": "vache",
                },
                {
                    "title": "Comté affiné 6 mois maison",
                    "url": "https://cuisine.journaldesfemmes.fr/recette/512345-comte-maison",
                    "description": "Fromage à pâte pressée cuite de vache",
                    "source": "cuisine.journaldesfemmes.fr",
                    "score": 8,
                    "type": "vache",
                },
                {
                    "title": "Brie de Meaux artisanal",
                    "url": "https://www.750g.com/brie-meaux-maison-r452700.htm",
                    "description": "Brie crémeux à croûte fleurie",
                    "source": "750g.com",
                    "score": 8,
                    "type": "vache",
                },
            ],
        }

        # Sélectionner les recettes ADAPTÉES
        if lait_detecte and lait_detecte in lait_specific_recipes:
            relevant_recipes = lait_specific_recipes[lait_detecte]
            print(
                f"   🎯 {len(relevant_recipes)} recettes spécifiques pour {lait_detecte}"
            )
        else:
            # Fallback générique (mais filtré)
            relevant_recipes = []
            all_fallback = self._get_absolute_fallback("", "", 20)

            # Filtrer pour ÉVITER les incohérences
            for recipe in all_fallback:
                title_lower = recipe["title"].lower()

                # Si on a détecté un lait, EXCLURE les autres laits
                if lait_detecte:
                    if lait_detecte == "brebis":
                        # Pour brebis, éviter chèvre et vache
                        if any(
                            x in title_lower
                            for x in [
                                "chèvre",
                                "chevre",
                                "crottin",
                                "vache",
                                "bovin",
                                "camembert",
                                "brie",
                            ]
                        ):
                            continue
                    elif lait_detecte == "chèvre":
                        # Pour chèvre, éviter brebis et vache
                        if any(
                            x in title_lower
                            for x in [
                                "brebis",
                                "mouton",
                                "ovin",
                                "vache",
                                "bovin",
                                "camembert",
                            ]
                        ):
                            continue

                relevant_recipes.append(recipe)

        # Limiter et retourner
        final = relevant_recipes[:max_results]

        # Vérifier la cohérence
        if lait_detecte:
            lait_final = set()
            for r in final:
                if "brebis" in r["title"].lower() or "mouton" in r["title"].lower():
                    lait_final.add("brebis")
                elif "chèvre" in r["title"].lower() or "chevre" in r["title"].lower():
                    lait_final.add("chèvre")
                elif "vache" in r["title"].lower() or "bovin" in r["title"].lower():
                    lait_final.add("vache")

            if len(lait_final) > 1:
                print(
                    f"   ⚠️ Attention: mélange de laits dans les résultats: {lait_final}"
                )
            else:
                print(
                    f"   ✅ Cohérence: tous les résultats sont au lait de {lait_detecte}"
                )

        print(f"✅ Fallback: {len(final)} recettes COHÉRENTES")
        return final

    def _deduplicate_recipes(self, recipes):
        """Élimine les doublons tout en gardant les meilleures versions"""
        unique_recipes = []
        seen_urls = set()

        # Trier d'abord par score pour garder les meilleures versions
        recipes.sort(key=lambda x: x.get("score", 0), reverse=True)

        for recipe in recipes:
            norm_url = self._normalize_url(recipe["url"])

            if not norm_url:
                # Recette sans URL valide, on garde quand même
                unique_recipes.append(recipe)
            elif norm_url not in seen_urls:
                seen_urls.add(norm_url)
                unique_recipes.append(recipe)

        return unique_recipes

    def _generate_similar_recipes(self, ingredients, cheese_type, count):
        """Génère des recettes similaires avec des sources VARIÉES"""
        print(f"   🧠 Génération de {count} recettes variées...")

        similar_recipes = []

        # LISTE DE SOURCES CRÉDIBLES ET VARIÉES
        sources = [
            {
                "domain": "marmiton.org",
                "base_url": "https://www.marmiton.org/recettes/",
                "credibility": 9,
            },
            {
                "domain": "cuisine.journaldesfemmes.fr",
                "base_url": "https://cuisine.journaldesfemmes.fr/recette/",
                "credibility": 8,
            },
            {
                "domain": "750g.com",
                "base_url": "https://www.750g.com/",
                "credibility": 8,
            },
            {
                "domain": "regal.fr",
                "base_url": "https://www.regal.fr/produit/fromage/",
                "credibility": 8,
            },
            {
                "domain": "femmeactuelle.fr",
                "base_url": "https://www.femmeactuelle.fr/cuisine/",
                "credibility": 7,
            },
            {
                "domain": "chefclub.tv",
                "base_url": "https://chefclub.tv/recettes/",
                "credibility": 7,
            },
            {
                "domain": "allrecipes.fr",
                "base_url": "https://www.allrecipes.fr/recette/",
                "credibility": 7,
            },
            {
                "domain": "mesrecettesfaciles.fr",
                "base_url": "https://www.mesrecettesfaciles.fr/",
                "credibility": 6,
            },
        ]

        # Extraire des mots-clés des ingrédients
        ingredients_lower = ingredients.lower()

        # Détecter le type principal
        cheese_family = "fromage"
        if any(x in ingredients_lower for x in ["chèvre", "chevre"]):
            cheese_family = "chevre"
            titles = [
                "Fromage de chèvre maison",
                "Crottin de chèvre artisanal",
                "Bûche de chèvre à l'herbe",
                "Chèvre frais fermier",
                "Fromage de chèvre cendré",
            ]
        elif "brebis" in ingredients_lower:
            cheese_family = "brebis"
            titles = [
                "Fromage de brebis affiné",
                "Brebis des Pyrénées",
                "Fromage de brebis à pâte pressée",
                "Fromage de brebis crémeux",
            ]
        elif any(x in ingredients_lower for x in ["frais", "blanc"]):
            cheese_family = "frais"
            titles = [
                "Fromage frais maison",
                "Faisselle artisanale",
                "Fromage blanc crémeux",
                "Fromage frais aux herbes",
            ]
        else:
            titles = [
                "Fromage artisanal maison",
                "Recette de fromage traditionnel",
                "Fromage fait maison",
                "Fromage fermier artisanal",
            ]

        for i in range(count):
            # Choisir une source aléatoire
            source = self.rng.choice(sources)

            # Choisir un titre aléatoire
            title = self.rng.choice(titles)

            # Créer un slug pour l'URL
            import re

            slug = re.sub(r"[^a-z0-9]+", "-", title.lower())
            slug = slug.strip("-")

            # Ajouter un identifiant unique
            import time

            unique_id = int(time.time() * 1000) % 10000 + i

            # Construire l'URL selon le format de la source
            if source["domain"] == "marmiton.org":
                url = f"{source['base_url']}recette_{slug}_{unique_id}.aspx"
            elif source["domain"] == "cuisine.journaldesfemmes.fr":
                url = f"{source['base_url']}{unique_id}-{slug}"
            elif source["domain"] == "750g.com":
                url = f"{source['base_url']}{slug}-r{unique_id}.htm"
            else:
                url = f"{source['base_url']}{slug}-{unique_id}"

            # Description variable
            descriptions = [
                f"Recette détaillée de {title.lower()}",
                f"Comment faire un {title.lower()} étape par étape",
                f"Guide complet pour réaliser un {title.lower()}",
                f"{title} - Recette traditionnelle et facile",
            ]

            similar_recipes.append(
                {
                    "title": title,
                    "url": url,
                    "description": self.rng.choice(descriptions),
                    "source": source["domain"],
                    "score": source["credibility"]
                    - 2,  # Score un peu inférieur aux vrais résultats
                    "generated": True,
                    "type": cheese_family,
                }
            )

        return similar_recipes

    # ===== MOTEURS DE RECHERCHE INDIVIDUELS =====

    def _search_google(self, query, max_results):
        """Recherche Google via DuckDuckGo API (plus fiable)"""
        try:
            from urllib.parse import quote
            import requests

            # Utiliser DuckDuckGo Instant Answer API (moins restrictive)
            url = f"https://api.duckduckgo.com/?q={quote(query)}&format=json&no_html=1&skip_disambig=1"

            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; FromagerBot/1.0; +https://github.com/volubyl/fromager)"
            }

            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                recipes = []

                # 1. Résultats instantanés (Instant Answer)
                if "Abstract" in data and data["Abstract"]:
                    if any(
                        kw in data["Abstract"].lower()
                        for kw in ["fromage", "cheese", "recette"]
                    ):
                        recipes.append(
                            {
                                "title": (
                                    data["Heading"]
                                    if "Heading" in data
                                    else "Recette de fromage"
                                ),
                                "url": (
                                    data["AbstractURL"]
                                    if "AbstractURL" in data
                                    else "https://duckduckgo.com"
                                ),
                                "description": data["Abstract"][:200],
                                "source": "duckduckgo.com",
                                "score": 8,
                                "engine": "ddg_api",
                            }
                        )

                # 2. Liens externes (Related Topics)
                if "RelatedTopics" in data:
                    for topic in data["RelatedTopics"][: max_results * 2]:
                        if "Text" in topic and "FirstURL" in topic:
                            text = topic["Text"]
                            url = topic["FirstURL"]

                            if any(
                                kw in text.lower()
                                for kw in ["fromage", "cheese", "recette", "recipe"]
                            ):
                                # Extraire titre
                                title = (
                                    text.split(".")[0][:80]
                                    if "." in text
                                    else text[:80]
                                )

                                recipes.append(
                                    {
                                        "title": title,
                                        "url": url,
                                        "description": text[:150],
                                        "source": self._extract_domain(url),
                                        "score": 7,
                                        "engine": "ddg_api",
                                    }
                                )

                return recipes

        except Exception as e:
            print(f"⚠️ Google/DuckDuckGo error: {e}")

        return []

    def _search_bing(self, query, max_results):
        """Recherche Bing SIMPLIFIÉE"""
        try:
            from urllib.parse import quote
            import requests

            url = f"https://www.bing.com/search?q={quote(query)}"

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            }

            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                import re

                recipes = []
                html = response.text

                # Pattern Bing simple
                pattern = r'<li[^>]*class="[^"]*b_algo[^"]*"[^>]*>(.*?)</li>'
                matches = re.findall(
                    pattern, html, re.DOTALL | re.IGNORECASE
                )  # CORRECTION ICI

                for match in matches[: max_results * 2]:
                    try:
                        # Titre dans h2
                        title_match = re.search(
                            r"<h2[^>]*>(.*?)</h2>", match, re.IGNORECASE
                        )
                        if not title_match:
                            continue

                        title = re.sub(r"<[^>]+>", "", title_match.group(1)).strip()

                        # Lien
                        link_match = re.search(
                            r'<a[^>]+href="([^"]+)"[^>]*>', match, re.IGNORECASE
                        )
                        if not link_match:
                            continue

                        url = link_match.group(1)

                        if url and "http" in url and "bing" not in url:
                            if any(kw in title.lower() for kw in ["fromage", "cheese"]):
                                recipes.append(
                                    {
                                        "title": title[:100],
                                        "url": url,
                                        "description": "Recette trouvée via Bing",
                                        "source": self._extract_domain(url),
                                        "score": 8,
                                        "engine": "bing",
                                    }
                                )
                    except:
                        continue

                return recipes

        except Exception as e:
            print(f"⚠️ Bing error: {e}")

        return []

    def _search_ecosia(self, query, max_results):
        """Recherche Ecosia ULTRA simple"""
        try:
            from urllib.parse import quote
            import requests

            url = f"https://www.ecosia.org/search?q={quote(query)}"

            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                # Ecosia a un HTML simple
                import re

                recipes = []
                html = response.text

                # Chercher les liens
                link_pattern = r'<a[^>]+class="[^"]*result-title[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
                links = re.findall(
                    link_pattern, html, re.DOTALL | re.IGNORECASE
                )  # CORRECTION ICI

                for url, title_html in links[:max_results]:
                    try:
                        title = re.sub(r"<[^>]+>", "", title_html).strip()

                        if (
                            url
                            and "http" in url
                            and any(
                                kw in title.lower()
                                for kw in ["fromage", "cheese", "formaggio"]
                            )
                        ):
                            recipes.append(
                                {
                                    "title": title[:80],
                                    "url": url,
                                    "description": "Recette écologique via Ecosia",
                                    "source": self._extract_domain(url),
                                    "score": 7,
                                    "engine": "ecosia",
                                }
                            )
                    except:
                        continue

                return recipes

        except Exception as e:
            print(f"⚠️ Ecosia error: {e}")

        return []

    def _search_simple_ddg(self, query, max_results):
        """DuckDuckGo ULTRA simple qui fonctionne"""
        try:
            from urllib.parse import quote
            import requests

            # Version TEXT seulement (pas HTML)
            url = f"https://api.duckduckgo.com/?q={quote(query)}&format=json&no_html=1&skip_disambig=1"

            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                recipes = []

                # Utiliser les résultats instantanés
                if "RelatedTopics" in data:
                    for topic in data["RelatedTopics"][:max_results]:
                        if "Text" in topic and "FirstURL" in topic:
                            text = topic["Text"]
                            url = topic["FirstURL"]

                            if any(
                                kw in text.lower()
                                for kw in ["fromage", "cheese", "recette"]
                            ):
                                # Extraire titre du texte
                                title = text.split(".")[0][:80]

                                recipes.append(
                                    {
                                        "title": title,
                                        "url": url,
                                        "description": text[:150],
                                        "source": self._extract_domain(url),
                                        "score": 6,
                                        "engine": "ddg_api",
                                    }
                                )

                return recipes

        except Exception as e:
            print(f"⚠️ DDG API error: {e}")

        return []

    def _normalize_url(self, url):
        """Normalise une URL pour la comparaison"""
        if not url:
            return ""

        # Enlever les paramètres communs
        url = url.lower().split("#")[0]  # Enlever les ancres

        # Enlever les paramètres tracking
        tracking_params = ["utm_", "ref=", "source=", "campaign="]
        for param in tracking_params:
            if param in url:
                parts = url.split("?")
                if len(parts) > 1:
                    query_params = parts[1].split("&")
                    filtered_params = [
                        p
                        for p in query_params
                        if not any(tp in p for tp in tracking_params)
                    ]
                    if filtered_params:
                        url = parts[0] + "?" + "&".join(filtered_params)
                    else:
                        url = parts[0]

        return url.strip("/")

    def _get_enriched_fallback_recipes(self, ingredients, cheese_type, max_results):
        """Fallback enrichi avec plus de recettes"""
        base_recipes = self._get_static_fallback_recipes(ingredients, cheese_type)

        # Ajouter des recettes supplémentaires selon les ingrédients
        additional_recipes = []

        ingredients_lower = ingredients.lower()

        # Recettes supplémentaires par ingrédient
        if any(x in ingredients_lower for x in ["chèvre", "chevre"]):
            additional_recipes.extend(
                [
                    {
                        "title": "Bûche de chèvre cendrée maison",
                        "url": "https://www.chevre.com/recettes/buche-chevre-cendree",
                        "description": "Recette traditionnelle de bûche de chèvre à la cendre",
                        "source": "chevre.com",
                        "score": 8,
                    },
                    {
                        "title": "Crottin de Chavignol maison",
                        "url": "https://www.fromagermaison.fr/crottin-chavignol",
                        "description": "Apprendre à faire des crottins de chèvre affinés",
                        "source": "fromagermaison.fr",
                        "score": 7,
                    },
                ]
            )

        if "brebis" in ingredients_lower:
            additional_recipes.extend(
                [
                    {
                        "title": "Fromage de brebis des Pyrénées",
                        "url": "https://www.brebis.com/recettes/fromage-brebis-pyrenees",
                        "description": "Fromage à pâte pressée de brebis façon Pyrénées",
                        "source": "brebis.com",
                        "score": 8,
                    }
                ]
            )

        if any(x in ingredients_lower for x in ["frais", "blanc"]):
            additional_recipes.extend(
                [
                    {
                        "title": "Faisselle maison en 24h",
                        "url": "https://www.fromagefrais.fr/recette/faisselle",
                        "description": "Faisselle crémeuse à déguster nature ou aux herbes",
                        "source": "fromagefrais.fr",
                        "score": 7,
                    }
                ]
            )

        # Combiner et limiter
        all_fallback = base_recipes + additional_recipes
        return all_fallback[:max_results]

    def search_web_recipes_fallback(self, ingredients, cheese_type, max_results=6):
        """Fallback robuste avec différentes stratégies"""
        print("🔄 Activation du mode fallback")

        try:
            # Stratégie 1: Recherche très simple
            simple_results = self._search_simple(ingredients, cheese_type, max_results)
            if simple_results:
                print(f"✅ Fallback simple: {len(simple_results)} résultats")
                return simple_results

            # Stratégie 2: Retourner des recettes statiques de la base
            print("⚠️ Utilisation de la base statique")
            return self._get_static_fallback_recipes(ingredients, cheese_type)

        except Exception as e:
            print(f"❌ Erreur fallback: {e}")
            return []

    def _clean_description(self, description: str) -> str:
        """Nettoie et formate la description"""
        # Limiter la longueur
        if len(description) > 280:
            description = description[:280] + "..."

        # Supprimer les caractères bizarres
        description = description.replace("\n", " ").replace("\r", " ")
        description = " ".join(description.split())  # Nettoyer espaces multiples

        return description

    def _extract_domain(self, url: str) -> str:
        """Extrait le nom de domaine d'une URL"""
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            domain = parsed.netloc
            # Retirer 'www.' et garder le domaine principal
            domain = domain.replace("www.", "")
            return domain
        except:
            return "web"

    # =====  MÉTHODE de validationICI =====
    def _validate_combination(
        self, lait: str, type_pate: str, aromates: list = None
    ) -> tuple:
        """
        Valide une combinaison lait/pâte/aromates
        Returns: (bool, str) - (est_valide, raison)
        """
        rules = self.knowledge["regles_compatibilite"]

        # Vérifier les exclusions absolues
        for exclusion in rules["exclusions_absolues"]:
            combo = exclusion["combinaison"]
            if f"lait:{lait}" in combo and f"type_pate:{type_pate}" in combo:
                alternatives = ", ".join(exclusion.get("alternatives", []))
                message = f"❌ {exclusion['raison']}\n\nAlternatives suggérées : {alternatives}"
                return False, message

        # Vérifier compatibilité lait/pâte
        for combo in rules["lait_x_type_pate"]["combinaisons_valides"]:
            if combo["lait"] == lait.lower():
                if type_pate in combo.get("types_pate_incompatibles", []):
                    message = f"❌ {combo['raison']}\n\nFromages {lait} compatibles : {', '.join(combo['types_pate_compatibles'])}"
                    return False, message

        return True, "✅ Combinaison valide"

    def _download_history_from_hf(self):
        """Télécharge l'historique depuis HF Dataset"""
        if not self.api:
            print("⚠️  Pas de token HF - historique local uniquement")
            self.history = []
            return

        try:
            downloaded_path = hf_hub_download(
                repo_id=self.hf_repo,
                filename=self.recipes_file,
                repo_type="dataset",
                token=self.hf_token,
            )

            with open(downloaded_path, "r", encoding="utf-8") as src:
                history = json.load(src)

            with open(self.recipes_file, "w", encoding="utf-8") as dst:
                json.dump(history, dst, indent=2, ensure_ascii=False)

            print(f"✅ Historique chargé : {len(history)} recettes")

        except Exception as e:
            print(f"ℹ️  Pas d'historique existant: {e}")
            with open(self.recipes_file, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _upload_history_to_hf(self):
        """Upload l'historique vers HF Dataset"""
        if not self.api:
            print("⚠️  Pas de token HF - sauvegarde locale uniquement")
            return False

        try:
            self.api.upload_file(
                path_or_fileobj=self.recipes_file,
                path_in_repo=self.recipes_file,
                repo_id=self.hf_repo,
                repo_type="dataset",
                commit_message=f"Update: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            )
            print("✅ Historique synchronisé avec HF")
            return True
        except Exception as e:
            print(f"❌ Erreur upload HF: {e}")
            return False

    def _load_history(self):
        """Charge l'historique depuis le fichier local"""
        if os.path.exists(self.recipes_file):
            try:
                with open(self.recipes_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return []
        return []

    def _save_to_history(self, ingredients, cheese_type, constraints, recipe):
        """Sauvegarde dans l'historique LOCAL ET HF"""
        try:
            history = self._load_history()

            recipe_lines = recipe.split("\n")
            cheese_name = "Fromage personnalisé"
            for line in recipe_lines:
                if "🧀" in line and len(line) < 100:
                    cheese_name = (
                        line.replace("🧀", "").replace("═", "").replace("║", "").strip()
                    )
                    break

            entry = {
                "id": len(history) + 1,
                "date": datetime.now().isoformat(),
                "cheese_name": cheese_name,
                "ingredients": ingredients,
                "type": cheese_type,
                "constraints": constraints,
                "recipe_complete": recipe,
                "recipe_preview": recipe[:300] + "..." if len(recipe) > 300 else recipe,
            }

            history.append(entry)
            history = history[-100:]

            with open(self.recipes_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

            # ✅ FORCE RETRY HF (3 tentatives)
            import time

            for i in range(3):
                sync_success = self._upload_history_to_hf()
                if sync_success:
                    print(f"✅ Recette #{entry['id']} sauvegardée et synchronisée")
                    break
                print(f"⚠️  Tentative HF {i+1}/3...")
                time.sleep(1)
            else:
                print(f"⚠️  Recette #{entry['id']} sauvegardée localement (HF échoué)")

            return True

        except Exception as e:
            print(f"❌ Erreur sauvegarde: {e}")
            return False

    def get_history(self):
        """Retourne l'historique complet"""
        return self._load_history()

    def get_history_display(self):
        """Retourne l'historique formaté pour affichage"""
        history = self.get_history()

        if not history:
            return "📭 Aucune recette créée pour le moment.\n\nCommencez par créer votre première recette ! 🧀"

        display = f"📚 HISTORIQUE DE VOS FROMAGES ({len(history)} recettes)\n"
        display += f"💾 Synchronisé avec Hugging Face Datasets\n"
        display += "=" * 70 + "\n\n"

        for entry in reversed(history[-20:]):
            date_obj = datetime.fromisoformat(entry["date"])
            date_str = date_obj.strftime("%d/%m/%Y à %H:%M")

            display += f"🧀 #{entry['id']} - {entry.get('cheese_name', 'Fromage')}\n"
            display += f"📅 {date_str}\n"
            display += f"🏷️  Type: {entry['type']}\n"
            display += f"🥛 Ingrédients: {', '.join(entry['ingredients'][:3])}"

            if len(entry["ingredients"]) > 3:
                display += f" (+{len(entry['ingredients'])-3} autres)"
            display += "\n"

            if entry.get("constraints"):
                display += f"⚙️  Contraintes: {entry['constraints']}\n"

            display += "-" * 70 + "\n\n"

        if len(history) > 20:
            display += f"💡 {len(history) - 20} recettes plus anciennes disponibles\n"

        return display

    def get_recipe_by_id(self, recipe_id):
        """Récupère une recette complète par son ID"""
        history = self.get_history()
        for entry in history:
            if entry["id"] == int(recipe_id):
                return entry["recipe_complete"]
        return "❌ Recette non trouvée"

    def clear_history(self):
        """Efface l'historique LOCAL ET HF"""
        try:
            with open(self.recipes_file, "w", encoding="utf-8") as f:
                json.dump([], f)

            if self.api:
                self._upload_history_to_hf()
                return "✅ Historique effacé (local + HF) !"
            else:
                return "✅ Historique local effacé"

        except Exception as e:
            return f"❌ Erreur: {e}"

    def sync_from_hf(self):
        """Force la synchronisation depuis HF"""
        self._download_history_from_hf()
        return self.get_history_display()

    def validate_ingredients(self, ingredients_text):
        """Valide les ingrédients"""
        if not ingredients_text or not ingredients_text.strip():
            return False, "⚠️ Vous devez entrer au moins un ingrédient !"

        ingredients_lower = ingredients_text.lower()

        has_milk = any(
            word in ingredients_lower
            for word in ["lait", "milk", "vache", "chèvre", "brebis", "bufflonne"]
        )

        if not has_milk:
            return (
                False,
                "❌ Il faut du lait pour faire du fromage !\n💡 Ajoutez : lait de vache, chèvre, brebis...",
            )

        has_coagulant = any(
            word in ingredients_lower
            for word in ["présure", "presure", "citron", "vinaigre", "acide"]
        )

        if not has_coagulant:
            return (
                True,
                "⚠️ Aucun coagulant détecté. Je suggérerai présure ou citron dans la recette.\n✅ Validation OK.",
            )

        return True, "✅ Ingrédients parfaits pour faire du fromage !"

    def _extract_lait_from_text(self, text: str) -> str:
        """Extrait le type de lait d'un texte"""
        if not text:
            return None

        text_lower = text.lower()

        lait_patterns = {
            "vache": ["vache", "bovin", "cow", "lait de vache"],
            "chevre": [
                "chèvre",
                "chevre",
                "caprin",
                "goat",
                "lait de chèvre",
                "lait de chevre",
            ],
            "brebis": ["brebis", "mouton", "ovin", "sheep", "lait de brebis"],
            "bufflonne": ["bufflonne", "buffle", "buffalo", "lait de bufflonne"],
        }

        # Priorité aux patterns les plus spécifiques
        for lait_type, patterns in lait_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return lait_type

        return None

    def _validate_combination(self, lait: str, type_pate: str) -> tuple:
        """
        Valide une combinaison lait/pâte selon les règles fromagères traditionnelles
        Returns: (bool, str) - (est_valide, message)
        """
        if not lait or not type_pate:
            return True, "✅ OK"
        
        lait_lower = lait.lower()
        type_lower = type_pate.lower()
        
        # ===== RÈGLES D'INCOMPATIBILITÉ ABSOLUE =====
        
        # RÈGLE 1 : Pas de pâte molle (croûte fleurie) avec lait de chèvre
        if lait_lower in ['chèvre', 'chevre', 'caprin'] and 'molle' in type_lower:
            return False, """
    ❌ **INCOMPATIBILITÉ DÉTECTÉE** : Pâte molle avec lait de chèvre

    **Pourquoi ?**
    Le lait de chèvre développe naturellement une croûte cendrée ou naturelle,
    pas une croûte fleurie comme le Camembert ou le Brie.

    **Alternatives recommandées pour le chèvre :**
    1. **Fromage frais** (consommation rapide)
    2. **Pâte pressée non cuite** (Tomme de chèvre)
    3. **Fromage cendré** (Sainte-Maure, Crottin)
    """
        
        # RÈGLE 2 : Pas de pâte molle avec lait de brebis
        if lait_lower in ['brebis', 'mouton', 'ovin'] and 'molle' in type_lower:
            return False, """
    ❌ **INCOMPATIBILITÉ DÉTECTÉE** : Pâte molle avec lait de brebis

    **Pourquoi ?**
    La brebis est traditionnellement utilisée pour des fromages à pâte pressée
    ou persillée, pas pour des croûtes fleuries.

    **Alternatives recommandées pour la brebis :**
    1. **Pâte pressée non cuite** (Ossau-Iraty, Manchego)
    2. **Pâte persillée** (Roquefort, Bleu de brebis)
    3. **Fromage frais** (consommation rapide)
    """
    
        # RÈGLE 3 : Bufflonne = seulement fromages frais
        if lait_lower in ['bufflonne', 'buffle'] and 'molle' in type_lower:
            return False, """
    ❌ **INCOMPATIBILITÉ** : Pâte molle avec lait de bufflonne

    **Pourquoi ?**
    Le lait de bufflonne est presque exclusivement utilisé pour des fromages frais
    à pâte filée comme la Mozzarella di Bufala.

    **Utilisation traditionnelle :**
    • Mozzarella di Bufala (frais, pâte filée)
    • Burrata (frais, crémeux)
    """
        
        return True, f"✅ Combinaison valide : {lait} + {type_pate}"

    def _suggest_alternatives(self, lait: str, type_pate: str) -> str:
        """Suggère des alternatives compatibles"""
        lait_lower = lait.lower()
        
        alternatives_by_lait = {
            'vache': """
    **Pour le lait de vache, tout est possible !**
    • Fromage frais : Faisselle, fromage blanc, ricotta
    • Pâte molle : Camembert, Brie, Chaource
    • Pâte pressée non cuite : Saint-Nectaire, Tomme, Morbier
    • Pâte pressée cuite : Comté, Beaufort, Gruyère
    • Pâte persillée : Bleu d'Auvergne, Fourme d'Ambert
    """,
            'chèvre': """
    **Fromages de chèvre traditionnels :**
    • **Frais** : Fromage de chèvre frais (consommation rapide)
    • **Cendré** : Sainte-Maure, Selles-sur-Cher, Valençay
    • **Pressé non cuit** : Tomme de chèvre

    **À éviter avec chèvre :**
    ❌ Pâte molle type Camembert
    ❌ Pâte pressée cuite type Comté
    """,
            'brebis': """
    **Fromages de brebis traditionnels :**
    • **Persillés** : Roquefort (AOP), Bleu des Causses
    • **Pressés non cuits** : Ossau-Iraty (AOP), Manchego
    • **Frais** : Fromage blanc de brebis

    **À éviter avec brebis :**
    ❌ Pâte molle type Brie/Camembert
    """,
            'bufflonne': """
    **Utilisations traditionnelles de la bufflonne :**
    • **Mozzarella di Bufala** (frais, pâte filée)
    • **Burrata** (frais, très crémeux)

    **Limitations :**
    • Pas d'affinage long
    • Pas de pâte pressée
    • Consommation rapide (frais)
    """
        }
        
        for lait_key, alternatives in alternatives_by_lait.items():
            if lait_key in lait_lower:
                return alternatives
        
        return "Essayez un autre type de fromage plus adapté à votre lait."

    def generate_recipe(
        self, 
        ingredients: str, 
        cheese_type: str,
        constraints: str = "", 
        creativity: int = 1,
        profile: str = "🧀 Amateur"
    ) -> str:
        """Génère une recette adaptée au profil utilisateur"""
    
        print(f"🧀 Génération pour: {ingredients} | Type: {cheese_type} | Profil: {profile}")
        
        # Stocker le profil actuel pour les fonctions internes
        self.current_profile = profile
        
        ##### VALIDATIONS ####
        # Validation des ingrédients
        valid, message = self.validate_ingredients(ingredients)
        if not valid:
            return message
        
        ingredients_list = [ing.strip() for ing in ingredients.split(',')]  # ← DÉFINIR ICI !
        
        # ===== DÉTECTER LE LAIT =====
        lait = self._extract_lait_from_text(' '.join(ingredients_list))
        print(f"   🥛 Lait détecté: {lait}")
        
        # ===== CHOISIR UN TYPE DIFFÉRENT SELON PROFIL =====
        cheese_type_clean = cheese_type  # Valeur par défaut
        
        if cheese_type == "Laissez l'IA choisir":
            # CHANGEMENT PRINCIPAL : type différent selon profil
            if profile == "🧀 Amateur":
                # Amateur = toujours fromage frais (simple et rapide)
                cheese_type_clean = "Fromage frais maison"
                
            elif profile == "🏭 Producteur":
                # Producteur = fromage avec valeur ajoutée
                fromages_pro = ["Camembert affiné", "Brie de Meaux", "Tomme de vache", "Fromage à pâte pressée"]
                import random
                cheese_type_clean = random.choice(fromages_pro)
                
            elif profile == "🎓 Formateur":
                # Formateur = fromage pédagogique
                cheese_type_clean = "Fromage pédagogique étape par étape"
            else:
                # Par défaut
                cheese_type_clean = self._determine_type_based_on_ingredients(ingredients_list)
        
        else:
            # L'utilisateur a choisi un type spécifique
            cheese_type_clean = cheese_type
            
            # Validation de compatibilité lait/type
            if lait and cheese_type_clean not in ["Fromage artisanal", "Laissez l'IA choisir"]:
                is_valid, reason = self._validate_combination(lait, cheese_type_clean)
                if not is_valid:
                    alternatives = self._suggest_alternatives(lait, cheese_type_clean)
                    return f"""
    ❌ **IMPOSSIBLE DE CRÉER CETTE RECETTE**

    **Combinaison rejetée :** {lait} + {cheese_type_clean}

    {reason}

    **💡 Alternatives compatibles avec {lait} :**
    {alternatives}

    **Modifiez soit :**
    1. Vos ingrédients (changez de lait)
    2. Votre type de fromage (choisissez-en un compatible)
    """
        
        print(f"   🎯 Type final: {cheese_type_clean}")
        
        #### fin des validations ####
        
        # ===== GÉNÉRER LA RECETTE (avec le profil) =====
        # Utilisez l'argument 'creativity' comme niveau de créativité
        base_recipe = self._generate_unique_recipe(
            ingredients_list, 
            cheese_type_clean, 
            constraints,
            creativity,  # Niveau de créativité
            profile      # ← Passer le profil ici !
        )
        
        # ADAPTER LA PRÉSENTATION
        adapted_recipe = self.adapt_recipe_to_profile_advanced(
            base_recipe, 
            profile, 
            ingredients_list, 
            cheese_type_clean
        )
        
        # Sauvegarder dans l'historique
        self._save_to_history(ingredients_list, cheese_type_clean, constraints, adapted_recipe)
        
        return adapted_recipe    
    def adapt_recipe_to_profile_advanced(
        self, recipe: str, profile: str, ingredients: list, cheese_type: str
    ) -> str:
        """Adapte la recette selon le profil utilisateur - délègue à adapt_recipe_to_profile"""
        
        # Utiliser la fonction de base qui fonctionne bien
        return self.adapt_recipe_to_profile(recipe, profile)

    def adapt_with_llm(
        self, recipe: str, profile: str, user_context: dict = None
    ) -> str:
        """Utilise un LLM pour adapter finement la recette"""

        if not self.openrouter_enabled and not self.google_ai_enabled:
            return recipe  # Fallback sur l'adaptation sans LLM

        prompt = f"""
        Tu es un expert fromager qui adapte des recettes selon le profil de l'utilisateur.
        
        PROFIL : {profile}
        
        RECETTE À ADAPTER :
        {recipe[:2000]}
        
        CONTEXTE UTILISATEUR (optionnel) :
        {user_context if user_context else 'Non spécifié'}
        
        ADAPTATION DEMANDÉE :
        1. Ton et vocabulaire adaptés au profil
        2. Complexité des étapes ajustée
        3. Conseils spécifiques au profil
        4. Focus sur les besoins du profil
        
        RÉPONSE : Adapte cette recette pour qu'elle soit parfaite pour ce profil.
        """

        # Utiliser votre méthode chat_with_llm existante
        adapted = self.chat_with_llm(prompt, [])

        return adapted if adapted else recipe

    def _adapt_ingredients_for_profile(
        self, ingredients: list, profile: str, cheese_type: str
    ) -> str:
        """Adapte la liste d'ingrédients selon le profil"""

        base_ingredients = """
    🥛 INGRÉDIENTS (Pour environ 500g de fromage)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """

        if profile == "🧀 Amateur":
            base_ingredients += (
                "- 2 litres de lait entier pasteurisé (en grande surface)\n"
            )
            base_ingredients += (
                "- 2ml de présure liquide (en pharmacie ou magasin bio)\n"
            )
            base_ingredients += "- 10g de sel de cuisine\n"
            base_ingredients += (
                "- 1 yaourt nature (pour les ferments, optionnel mais recommandé)\n\n"
            )
            base_ingredients += "**Vos ingrédients spécifiques :**\n"
            for ing in ingredients[:5]:  # Limiter à 5 pour ne pas submerger
                base_ingredients += f"• {ing.capitalize()}\n"

        elif profile == "🏭 Producteur":
            base_ingredients += "### 📦 SPÉCIFICATIONS TECHNIQUES\n\n"
            base_ingredients += "**Matériel de base :**\n"
            base_ingredients += "- Lait cru de qualité fromagère (< 100 000 UFC/ml)\n"
            base_ingredients += "- Présure standardisée (1:10 000)\n"
            base_ingredients += "- Sel alimentaire non iodé\n"
            base_ingredients += "- Ferments mésophiles DVS\n\n"
            base_ingredients += "**Paramètres qualité :**\n"
            base_ingredients += "- Acidité du lait : 16-18°D\n"
            base_ingredients += "- Température optimale : 32°C ±0.5\n"
            base_ingredients += "- Rapport poids/sel : 2%\n"

        else:  # Formateur
            base_ingredients += "### 🧪 INGRÉDIENTS POUR ATELIER\n\n"
            base_ingredients += "**Pour 6 participants :**\n"
            base_ingredients += "- 12 litres de lait (2L par personne)\n"
            base_ingredients += "- 12ml de présure (pré-diluée)\n"
            base_ingredients += "- 60g de sel (en plusieurs bols)\n"
            base_ingredients += "- 6 yaourts nature (un par groupe)\n\n"
            base_ingredients += "**Matériel pédagogique :**\n"
            base_ingredients += "- Échantillons de chaque ingrédient\n"
            base_ingredients += "- Fiches avec photos des étapes\n"
            base_ingredients += "- Thermomètre par groupe\n"

        return base_ingredients

    def _adapt_steps_for_profile(
        self, recipe: str, profile: str, cheese_type: str
    ) -> list:
        """Adapte les étapes selon le profil"""

        if profile == "🧀 Amateur":
            return [
                "✅ **ÉTAPE 1 : On prépare tout** (5 min)",
                "   - Sortez tous les ingrédients",
                "   - Lavez-vous bien les mains",
                "   - Ayez un chrono près de vous",
                "",
                "✅ **ÉTAPE 2 : On chauffe doucement** (15 min)",
                "   - Le lait à 32°C, PAS PLUS !",
                "   - Comme un bain de bébé",
                "",
                "✅ **ÉTAPE 3 : On ajoute la présure** (2 min)",
                "   - Mélangez doucement 30 secondes",
                "   - Couvrez et NE TOUCHEZ PLUS !",
                "",
                "✅ **ÉTAPE 4 : On patiente** (45-60 min)",
                "   - C'est l'heure du café !",
                "   - Le caillé se forme tout seul",
                "",
            ]

        elif profile == "🏭 Producteur":
            return [
                "📋 **PROCÉDURE STANDARD :**",
                "",
                "**PHASE 1 : PRÉPARATION**",
                "1. Vérification qualité lait (pH, température, flore)",
                "2. Calcul des dosages précis",
                "3. Stérilisation équipement (nettoyage + désinfection)",
                "",
                "**PHASE 2 : TRANSFORMATION**",
                "4. Chauffage à 32°C ±0.5 (contrôle continu)",
                "5. Emprésurage : 2ml/10L, agitation 30s",
                "6. Caillage : 45min à 32°C (mesure pH cible : 6.4)",
                "",
                "**PHASE 3 : FINITION**",
                "7. Découpage : grille 1cm (temps précis)",
                "8. Égouttage : 12h à 20°C",
                "9. Salage : 2% poids final",
                "",
            ]

        else:  # Formateur
            return [
                "🎯 **OBJECTIFS PÉDAGOGIQUES :**",
                "1. Comprendre le rôle de chaque ingrédient",
                "2. Observer la transformation lait → caillé",
                "3. Identifier les points critiques",
                "",
                "⏱️ **DÉROULÉ DE L'ATELIER (3h) :**",
                "",
                "**0-30min : Théorie**",
                "- Présentation des ingrédients",
                "- Explication scientifique simple",
                "- Distribution des fiches",
                "",
                "**30-90min : Pratique**",
                "- Par groupes de 2-3 personnes",
                "- Chaque groupe suit les étapes",
                "- Animateur circule et aide",
                "",
                "**90-150min : Observations**",
                "- Comparaison des résultats",
                -"Explication des différences",
                "- Conseils pour la suite",
                "",
            ]

    def _adapt_advice_for_profile(self, profile: str, cheese_type: str) -> str:
        """Fournit des conseils adaptés au profil"""

        if profile == "🧀 Amateur":
            return """
    💡 **MES 3 CONSEILS POUR RÉUSSIR :**

    1. **NE STRESSER PAS !** Le fromage est vivant, il s'adapte.
    2. **HYGIÈNE OUI, STÉRILITÉ NON** : Lavez bien, pas besoin de bloc opératoire.
    3. **GOÛTEZ SANS PEUR** : À chaque étape, c'est comme ça qu'on apprend.

    😊 **CE QUI PEUT MAL SE PASSER (ET C'EST NORMAL) :**
    - Le caillé est trop mou ? → Plus de temps ou plus de présure
    - Trop acide ? → Moins de temps avant égouttage
    - Pas de goût ? → Plus d'affinage ou plus de sel

    🎉 **QUAND C'EST RÉUSSI :**
    Félicitations ! Vous venez de créer votre premier fromage.
    Partagez-le, montrez-le, soyez fier !
    """

        elif profile == "🏭 Producteur":
            return """
    📊 **POINTS DE CONTRÔLE QUALITÉ :**

    ✅ **Critères objectifs :**
    - Rendement : > 10% (poids fromage/poids lait)
    - pH final : 5.2-5.4
    - Taux d'humidité : 45-55%
    - Conservation : > 21 jours à 4°C

    ⚠️ **Non-conformités courantes :**
    - Acidité excessive → Réduire fermentation 10%
    - Croûte craquelée → Humidité cave à 90%
    - Goût amer → Vérifier flore contaminante

    📈 **OPTIMISATION :**
    - Traçabilité complète (lot, date, paramètres)
    - Analyse coûts/marge par batch
    - Fiches techniques à jour
    """

        else:  # Formateur
            return """
    🎓 **CONSEILS PÉDAGOGIQUES :**

    📝 **AVANT L'ATELIER :**
    1. Préparez des échantillons à chaque étape
    2. Anticipez les questions fréquentes
    3. Testez la recette vous-même

    🗣️ **PENDANT L'ATELIER :**
    1. Posez des questions ouvertes
    2. Valorisez chaque essai (même raté)
    3. Faites des liens avec la science

    📚 **APRÈS L'ATELIER :**
    1. Fournissez une fiche récap
    2. Proposez des ressources pour aller plus loin
    3. Créez un groupe d'échange

    ❓ **QUESTIONS À POSER AUX PARTICIPANTS :**
    - "Que remarquez-vous ?"
    - "Pourquoi selon vous ?"
    - "Comment pourrions-nous améliorer ?"
    """

    def _generate_detailed_recipe(self, ingredients, cheese_type, constraints):
        """Génère une recette UNIQUE enrichie avec variations"""
        
        # ===== RÉCUPÉRER LE PROFIL SI DISPONIBLE =====
        profile = None
        if hasattr(self, 'current_profile'):
            profile = self.current_profile
    
        # ===== ADAPTER LES INGRÉDIENTS SELON PROFIL =====
        if profile == "🧀 Amateur":
            # Amateur : quantités réduites, ingrédients simples
            lait_qty = "1 litre"  # Petit format pour test
            presure_source = "présure liquide (en pharmacie)"
            conseil_special = "✨ **ASTUCE DÉBUTANT** : Commencez avec 1L de lait pour tester !"
        
        elif profile == "🏭 Producteur":
            # Producteur : quantités professionnelles
            lait_qty = "10 litres"  # Format pro
            presure_source = "présure standardisée 1:10.000"
            conseil_special = "📊 **CALCUL RENDEMENT** : 10L de lait → ~1.2kg de fromage"
            
        elif profile == "🎓 Formateur":
            # Formateur : quantités pour atelier
            lait_qty = "5 litres"  # Format démonstration
            presure_source = "présure diluée pour démonstration"
            conseil_special = "🎯 **OBJECTIF PÉDAGOGIQUE** : Montrer chaque étape lentement"

        # ===== DOUBLE VALIDATION POST-DÉTERMINATION =====
        # Extraire le lait des ingrédients
        ingredients_str = " ".join(ingredients).lower()
        lait = self._extract_lait_from_text(ingredients_str)

        # Valider la combinaison finale
        if lait and cheese_type:
            is_valid, reason = self._validate_combination(lait, cheese_type)
            if not is_valid:
                # Forcer un type compatible
                rules = self.knowledge_base["regles_compatibilite"]
                for combo in rules["lait_x_type_pate"]["combinaisons_valides"]:
                    if combo["lait"] == lait.lower():
                        compatibles = combo["types_pate_compatibles"]
                        if compatibles:
                            cheese_type = compatibles[
                                0
                            ]  # Utiliser le premier compatible
                            break

        # Récupérer toutes les infos de la base
        type_info = self._get_type_info(cheese_type)
        temp_affinage = self._get_temperature_affinage(cheese_type)
        conservation_info = self._get_conservation_info(cheese_type)
        accord_vin = self._get_accord_vin(cheese_type)
        accord_mets = self._get_accord_mets(cheese_type)
        epices_suggestions = self._suggest_epices(ingredients, cheese_type)
        problemes_a_eviter = self._get_problemes_pertinents(cheese_type)
        materiel = self._get_materiel_debutant()

        # Générer nom créatif
        cheese_name = self._generate_creative_name(cheese_type, ingredients)

        # Construire la recette enrichie
        recipe = f"""
╔══════════════════════════════════════════════════════════════╗
║                    🧀 {cheese_name.upper()}                     
╚══════════════════════════════════════════════════════════════╝

📋 TYPE DE FROMAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{cheese_type}
{type_info['description']}
Exemples similaires : {type_info['exemples']}
Difficulté : {type_info['difficulte']}
Durée totale : {type_info['duree']}


🥛 INGRÉDIENTS (Pour environ 500g de fromage)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- 2 litres de lait entier pasteurisé
  (préférer lait cru de producteur local si possible)
- 2ml de présure liquide (ou 1/4 comprimé)
  Alternative : 60ml de jus de citron frais
- 10g de sel de mer fin ou gros sel
- Ferments lactiques (optionnel mais recommandé)

**Vos ingrédients spécifiques :**
{self._format_user_ingredients(ingredients)}

{epices_suggestions}


🔧 MATÉRIEL NÉCESSAIRE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{materiel}


📝 ÉTAPES DE FABRICATION DÉTAILLÉES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PHASE 1 : PRÉPARATION DU LAIT (20 minutes)
──────────────────────────────────────────
1. **Stérilisation** : Laver tout le matériel à l'eau bouillante
2. **Chauffage** : Verser le lait dans la casserole propre
3. **Température** : Chauffer doucement à 32°C (±1°C)
   ⚠️ NE JAMAIS dépasser 35°C au risque de tuer les ferments
4. **Stabilisation** : Maintenir 32°C pendant 10 minutes
5. **Ferments** (optionnel) : Ajouter et mélanger 1 minute


PHASE 2 : CAILLAGE (45-90 minutes)
────────────────────────────────────
6. **Ajout présure** : Diluer la présure dans 50ml d'eau froide
7. **Incorporation** : Verser en mélangeant délicatement 30 secondes
8. **Repos** : Couvrir et laisser reposer SANS BOUGER
   - Avec présure : 45-60 minutes
   - Avec citron : 20-30 minutes (plus rapide mais moins stable)
9. **Test de caillage** : Le caillé doit se briser net comme du tofu
   Si encore liquide → Attendre 15 minutes de plus


PHASE 3 : DÉCOUPAGE ET BRASSAGE (15 minutes)
─────────────────────────────────────────────
10. **Découpage** : Couper le caillé en cubes de 1cm
    Faire un quadrillage vertical puis horizontal
11. **Repos** : Laisser reposer 5 minutes (petit-lait sort)
12. **Brassage** : Mélanger TRÈS doucement 10 minutes
    Le caillé raffermit sans se désintégrer


PHASE 4 : MOULAGE ET ÉGOUTTAGE ({self._get_egouttage_time(cheese_type)})
───────────────────────────────────────
13. **Préparation** : Disposer l'étamine dans le moule perforé
14. **Transfert** : Verser le caillé à la louche (garder le petit-lait!)
15. **Égouttage naturel** : Laisser égoutter
    - Fromage frais : 2-4 heures à température ambiante
    - Autres types : 12-24 heures au frais (12°C)
16. **Retournement** : Retourner toutes les 4 heures


PHASE 5 : SALAGE
───────────────────────────────────────
17. **Démoulage** : Démouler délicatement sur une surface propre
18. **Salage** : Frotter toutes les faces avec le sel
    Quantité : 2% du poids du fromage (environ 10g pour 500g)
19. **Alternative saumure** : Immerger 2-4h dans eau salée (300g/L)


PHASE 6 : AFFINAGE
───────────────────────────────────────
20. **Conditions d'affinage** :
    {temp_affinage}
21. **Durée d'affinage** : {type_info['duree']}
22. **Soins** : {self._get_soins_affinage(cheese_type)}


⚠️ PROBLÈMES COURANTS ET SOLUTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{problemes_a_eviter}


📦 CONSERVATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{conservation_info}


🍷 DÉGUSTATION ET ACCORDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**Moment idéal** : {self._get_tasting_time(cheese_type)}
**Température de service** : 18-20°C (sortir 1h avant)

**Accords vins** : {accord_vin}
**Accords mets** : {accord_mets}

**Suggestion de présentation** :
Servir sur une planche en bois avec pain frais, quelques noix,
un peu de miel et des fruits de saison


🎨 VARIANTES CRÉATIVES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{self._get_variantes(cheese_type, ingredients)}


💡 CONSEILS DU MAÎTRE FROMAGER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{self._get_conseils_fromager()}

✨ **Le petit-lait est précieux !**
   Ne le jetez pas :
   - Faire du pain (remplace l'eau)
   - Ricotta (rechauffer à 90°C, récupérer les flocons)
   - Arroser les plantes (riche en nutriments)
   - Base de smoothies protéinés


📚 SCIENCE DU FROMAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**Le caillage** : La présure (enzyme) coupe les protéines du lait
(caséines) qui s'agglomèrent en réseau 3D emprisonnant eau et graisses.

**L'affinage** : Bactéries et levures transforment protéines et graisses
en molécules aromatiques. Plus long = goût plus prononcé.


{self._add_constraints_note(constraints)}

╔══════════════════════════════════════════════════════════════╗
║  Recette générée le {datetime.now().strftime('%d/%m/%Y à %H:%M')}           
║  Bonne fabrication ! 🧀                                       
║  Patience et hygiène sont les clés de la réussite            
╚══════════════════════════════════════════════════════════════╝
"""
        return recipe

    def _generate_unique_recipe(
        self, ingredients, cheese_type, constraints, creativity, profile=None
    ):
        """Génère une recette UNIQUE enrichie avec variations"""

        print(f"🎲 Génération UNIQUE avec: profil={profile}, créativité={creativity}")
        
        # === AJOUTER DE L'ALÉATOIRE BASÉ SUR LES INGRÉDIENTS ===
        import hashlib

        # Créer une "signature" unique basée sur les ingrédients
        ingredients_hash = hashlib.md5(",".join(ingredients).encode()).hexdigest()[:8]
        seed_value = int(ingredients_hash, 16) % 1000
        self.rng.seed(seed_value)  # Réinitialiser le générateur aléatoire

        print(f"🎲 Seed unique pour cette recette: {seed_value}")

        # ===== VARIABLES SPÉCIFIQUES AU PROFIL =====
        if profile:
            print(f"   🎯 Adaptation pour profil: {profile}")
            
            # Récupérer les paramètres selon le profil
            if profile == "🧀 Amateur":
                quantite_lait = "1 litre"
                temps_total = "24-48 heures"
                difficulte = "Facile"
                conseil_special = "✨ **ASTUCE DÉBUTANT** : Commencez petit pour apprendre !"
                
            elif profile == "🏭 Producteur":
                quantite_lait = "10 litres" 
                temps_total = "2-8 semaines"
                difficulte = "Technique"
                conseil_special = "📊 **CONSEIL PRO** : Notez tous les paramètres pour reproduire vos succès !"
                
            elif profile == "🎓 Formateur":
                quantite_lait = "5 litres"
                temps_total = "Variable selon atelier"
                difficulte = "Pédagogique"
                conseil_special = "🎓 **CONSEIL FORMATEUR** : Préparez des questions pour chaque étape !"
        else:
            # Valeurs par défaut
            quantite_lait = "2 litres"
            temps_total = "Variable"
            difficulte = "Moyenne"
            conseil_special = ""

        # ===== VARIATIONS UNIQUES BASÉES SUR LES INGRÉDIENTS =====

        # 1. Nom créatif unique (modifié selon profil)
        cheese_name = self._generate_unique_cheese_name(
            ingredients, cheese_type, seed_value
        )
        
        # Ajouter une mention du profil dans le nom
        if profile:
            if profile == "🧀 Amateur":
                cheese_name = f"{cheese_name} (Version Débutant)"
            elif profile == "🏭 Producteur":
                cheese_name = f"{cheese_name} (Édition Professionnelle)"
            elif profile == "🎓 Formateur":
                cheese_name = f"{cheese_name} (Version Pédagogique)"

        # 2. Ingrédients avec variations (MODIFIÉ pour utiliser quantite_lait)
        unique_ingredients = self._generate_unique_ingredients(
            ingredients, cheese_type, seed_value, quantite_lait  # ← PASSER quantite_lait !
        )

        # 3. Étapes avec variations (MODIFIÉ pour utiliser les paramètres du profil)
        unique_steps = self._generate_unique_steps(
            cheese_type, seed_value, creativity, profile, quantite_lait  # ← AJOUTER profil et quantite_lait
        )

        # 4. Conseils personnalisés
        unique_advice = self._generate_unique_advice(ingredients, cheese_type, seed_value)
        
        # Ajouter le conseil spécial du profil
        if conseil_special:
            unique_advice = f"{conseil_special}\n\n{unique_advice}"

        # ===== CONSTRUIRE LA RECETTE UNIQUE =====

        # Récupérer les infos de base (MAJ avec les valeurs du profil)
        type_info = self._get_type_info(cheese_type)
        
        # MODIFIER la durée avec celle du profil
        type_info_modified = type_info.copy()
        type_info_modified['duree'] = temps_total  # ← REMPLACER par la durée du profil
        type_info_modified['difficulte'] = difficulte  # ← REMPLACER par la difficulté du profil
        
        temp_affinage = self._get_temperature_affinage(cheese_type)
        conservation_info = self._get_conservation_info(cheese_type)
        accord_vin = self._get_accord_vin(cheese_type)
        accord_mets = self._get_accord_mets(cheese_type)
        epices_suggestions = self._suggest_epices(ingredients, cheese_type)
        problemes_a_eviter = self._get_problemes_pertinents(cheese_type)
        
        # Matériel selon profil (FONCTION À CRÉER)
        materiel = self._get_materiel_by_profile(profile)  # ← NOUVELLE FONCTION !

        # Construire la recette avec les parties uniques
        recipe = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🧀 {cheese_name.upper()}                     
    ║                    (Recette #{seed_value} - {profile if profile else "Standard"})
    ╚══════════════════════════════════════════════════════════════╝

    📋 TYPE DE FROMAGE
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    {cheese_type}
    {type_info_modified['description']}
    Exemples similaires : {type_info_modified['exemples']}
    Difficulté : {type_info_modified['difficulte']}
    Durée totale : {type_info_modified['duree']}

    {unique_ingredients}

    🔧 MATÉRIEL NÉCESSAIRE
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    {materiel}
        
    {unique_steps}

    ⚠️ PROBLÈMES COURANTS ET SOLUTIONS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    {problemes_a_eviter}

    📦 CONSERVATION
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    {conservation_info}

    🍷 DÉGUSTATION ET ACCORDS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    **Accords vins** : {accord_vin}
    **Accords mets** : {accord_mets}

    💡 CONSEILS PERSONNALISÉS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    {unique_advice}

    {self._add_constraints_note(constraints)}

    ╔══════════════════════════════════════════════════════════════╗
    ║  Recette générée le {datetime.now().strftime('%d/%m/%Y à %H:%M')}           
    ║  Quantité: {quantite_lait} - Profil: {profile if profile else "Standard"}                                      
    ╚══════════════════════════════════════════════════════════════╝
    """

        return recipe
    
    def _get_materiel_by_profile(self, profile):
        """Retourne le matériel adapté au profil"""
        if profile == "🧀 Amateur":
            return """• 1 grande casserole (3-5L)
    • Thermomètre de cuisine
    • Torchon propre ou étamine
    • Saladier percé (ou moule basique)
    • Cuillère en bois"""
        
        elif profile == "🏭 Producteur":
            return """• Cuve inox 20L
    • Thermomètre de précision (±0.5°C)
    • Presse à fromage
    • pH-mètre
    • Cave d'affinage contrôlée
    • Balance de précision (0.1g)
    • Cahier de suivi"""
        
        elif profile == "🎓 Formateur":
            return """• Matériel pour 6 participants
    • Thermomètres ×6
    • Moules ×6
    • Échantillons pédagogiques
    • Fiches d'observation
    • Paperboard ou tableau"""
        
        else:
            return """• Grande casserole inox
    • Thermomètre
    • Moule à fromage
    • Étamine
    • Louche"""    
    
    def _generate_unique_cheese_name(self, ingredients, cheese_type, seed_value):
        """Génère un nom de fromage unique"""
        ingredients_lower = ' '.join(ingredients).lower()
    
        # Mots liés aux ingrédients
        ingredient_words = []
        for ing in ingredients:
            ing_lower = ing.lower()
            if 'chèvre' in ing_lower or 'chevre' in ing_lower:
                ingredient_words.append("Chèvre")
            elif 'brebis' in ing_lower:
                ingredient_words.append("Brebis")
            elif 'thym' in ing_lower:
                ingredient_words.append("au Thym")
            elif 'romarin' in ing_lower:
                ingredient_words.append("au Romarin")
            elif 'poivre' in ing_lower:
                ingredient_words.append("Poivré")
            elif 'herbe' in ing_lower:
                ingredient_words.append("aux Herbes")
    
        # Réinitialiser le générateur aléatoire avec le seed
        import random
        local_rng = random.Random(seed_value)
        
        # Bases pour les noms
        prefixes = ["Délice", "Secret", "Trésor", "Velours", "Nuage", "Crème", "Douceur"]
        suffixes = ["du Terroir", "de la Maison", "Artisanal", "Fermier", "Lacté", "Gourmand"]
        
        if ingredient_words:
            name_part = local_rng.choice(ingredient_words)
        else:
            name_part = local_rng.choice(prefixes)
        
        suffix = local_rng.choice(suffixes)
        
        return f"{name_part} {suffix}"

    def _generate_unique_ingredients(self, ingredients, cheese_type, seed_value, quantite_lait="2 litres"):
        """Génère une liste d'ingrédients unique"""
        import random
        local_rng = random.Random(seed_value)
    
        base = f"""
🥛 INGRÉDIENTS (Pour environ {quantite_lait} de lait)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
        # Extraire la quantité numérique
        if "1 litre" in quantite_lait:
            lait_qty_num = 1
        elif "5 litres" in quantite_lait:
            lait_qty_num = 5
        elif "10 litres" in quantite_lait:
            lait_qty_num = 10
        else:
            lait_qty_num = 2  # Par défaut
        
        # Calculer les autres quantités proportionnellement
        presure_qty = lait_qty_num * 1.0  # 1ml par litre
        sel_qty = lait_qty_num * 5.0      # 5g par litre
        
        # Type de lait variable
        lait_types = ["lait entier pasteurisé", "lait cru", "lait de ferme", "lait bio"]
        lait_type = local_rng.choice(lait_types)
        
        base += f"- {lait_qty_num} litre(s) de {lait_type}\n"
        base += f"- {presure_qty}ml de présure liquide\n"
        base += f"- {sel_qty}g de sel de mer fin\n"
        
        # Ajouter des variations basées sur le type
        if "molle" in cheese_type.lower():
            base += "- 1 yaourt nature (pour les ferments)\n"
        elif "pressée" in cheese_type.lower():
            base += "- Ferments lactiques mésophiles\n"
        
        # Vos ingrédients spécifiques
        if ingredients:
            base += "\n**Vos ingrédients spécifiques :**\n"
            for ing in ingredients[:5]:
                base += f"• {ing.capitalize()}\n"
        
        return base

    def _generate_unique_steps(self, cheese_type, seed_value, creativity, profile=None, quantite_lait=None):
        """Génère des étapes uniques complètes avec adaptation au profil"""
        import random
        local_rng = random.Random(seed_value)
        
        # Variables aléatoires basées sur le seed
        repos_time = local_rng.choice(["45", "50", "55", "60"])
        temp_choice = local_rng.choice(["31", "32", "33", "34"])
        cube_size = local_rng.choice(["1", "1.5", "2"])
        
        # DÉBUT DES MODIFICATIONS : Adapter selon le profil
        if profile == "🧀 Amateur":
            # Amateur : simplifier et guider
            repos_time = local_rng.choice(["40", "45", "50"])  # Plus court
            temp_choice = "32"  # Température fixe pour simplifier
            cube_size = "1.5"   # Taille moyenne, plus facile
            mention_profil = "🎯 **RECETTE SIMPLIFIÉE POUR DÉBUTANT**"
            
        elif profile == "🏭 Producteur":
            # Producteur : plus précis et technique
            repos_time = local_rng.choice(["55", "60", "65"])  # Plus long
            temp_choice = local_rng.choice(["32.0", "32.5", "33.0"])  # Plus précis
            cube_size = local_rng.choice(["1.0", "1.2", "1.5"])  # Plus précis
            mention_profil = "🏭 **PROTOCOLE PROFESSIONNEL**"
            
        elif profile == "🎓 Formateur":
            # Formateur : pédagogique avec explications
            repos_time = "45"  # Fixe pour la démonstration
            temp_choice = "32"  # Fixe pour la démonstration
            cube_size = "2"    # Plus visible pour démonstration
            mention_profil = "🎓 **DÉMONSTRATION PÉDAGOGIQUE**"
        
        else:
            mention_profil = "📝 **ÉTAPES DE FABRICATION**"
        # FIN DES MODIFICATIONS
        
        # Déterminer le temps d'égouttage selon le type
        if "frais" in cheese_type.lower():
            egouttage = local_rng.choice(["2-4", "3-5", "4-6"]) + " heures"
            affinage = "Pas d'affinage nécessaire"
        elif "molle" in cheese_type.lower():
            egouttage = local_rng.choice(["12-18", "18-24", "24-36"]) + " heures"
            affinage = local_rng.choice(["2-3", "3-4", "4-6"]) + " semaines"
        else:
            egouttage = local_rng.choice(["18-24", "24-36", "36-48"]) + " heures"
            affinage = local_rng.choice(["3-6", "6-9", "9-12"]) + " semaines"
        
        # MODIFICATION : Ajouter le profil dans le titre
        steps = f"""
    {mention_profil}
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    PHASE 1 : PRÉPARATION (20 minutes)
    ──────────────────────────────────────────
    1. **Stérilisation** : Laver tout le matériel à l'eau bouillante
    2. **Chauffage** : Verser le lait dans la casserole propre
    3. **Température** : Chauffer doucement à {temp_choice}°C (±1°C)
    ⚠️ Ne jamais dépasser 35°C
    4. **Stabilisation** : Maintenir {temp_choice}°C pendant 5 minutes
    5. **Ferments** (optionnel) : Ajouter et mélanger 1 minute

    PHASE 2 : CAILLAGE ({repos_time} minutes)
    ──────────────────────────────────────────
    6. **Présure** : Diluer dans 50ml d'eau à température ambiante
    7. **Incorporation** : Verser en filet tout en tournant
    8. **Mélange** : 30 secondes exactement, puis arrêter
    9. **Repos** : Couvrir et laisser {repos_time} minutes SANS TOUCHER
    10. **Test** : Le caillé doit se briser net

    PHASE 3 : DÉCOUPAGE ET BRASSAGE (15 minutes)
    ─────────────────────────────────────────────
    11. **Découpage** : Grille de {cube_size}cm (vertical puis horizontal)
    12. **Repos** : 5 minutes pour laisser s'échapper le petit-lait
    13. **Brassage** : Mélanger TRÈS doucement 10 minutes

    PHASE 4 : MOULAGE ET ÉGOUTTAGE
    ───────────────────────────────
    14. **Moulage** : Étamine dans le moule, verser à la louche
    15. **Égouttage** : {egouttage} à température ambiante
    16. **Retournements** : Toutes les 4 heures pour une forme régulière

    PHASE 5 : SALAGE
    ────────────────
    17. **Démoulage** : Sur planche propre et sèche
    18. **Salage** : Frotter toutes les faces (2% du poids)
    19. **Alternative** : Saumure 2-4h (300g sel/L)

    PHASE 6 : AFFINAGE
    ──────────────────
    20. **Conditions** : Cave à 10-12°C, 85-90% humidité
    21. **Durée** : {affinage}
    22. **Soins** : Retourner quotidiennement la 1ère semaine
    """
        
        # MODIFICATION : Ajouter des conseils spécifiques au profil
        if profile == "🧀 Amateur":
            steps += "\n💡 **CONSEILS SPÉCIAUX POUR DÉBUTANT :**\n"
            steps += "• Ne stressez pas ! Le fromage est vivant et s'adapte.\n"
            steps += "• Si le caillage prend plus de temps, c'est normal.\n"
            steps += "• Goûtez à chaque étape pour comprendre l'évolution.\n"
        
        elif profile == "🏭 Producteur":
            steps += "\n📊 **POINTS DE CONTRÔLE QUALITÉ :**\n"
            steps += "• Température maintenue à ±0.5°C\n"
            steps += "• Temps de caillage documenté\n"
            steps += "• pH mesuré après 24h\n"
            steps += "• Rendement calculé (poids fromage/poids lait)\n"
        
        elif profile == "🎓 Formateur":
            steps += "\n🎯 **QUESTIONS PÉDAGOGIQUES À POSER :**\n"
            steps += "• 'Que remarquez-vous pendant le chauffage ?'\n"
            steps += "• 'Pourquoi la température est-elle cruciale ?'\n"
            steps += "• 'Quels sont les signes d'un bon caillage ?'\n"
            steps += "• 'Comment évolue la texture avec le temps ?'\n"
        
        # Ajouter des variations créatives (garder l'existant)
        if creativity >= 2:
            steps += "\n**🎨 VARIATIONS CRÉATIVES :**\n"
            
            creative_phases = [
                "✨ **Pré-infusion** : Faire infuser le lait avec des herbes 30 min avant chauffage",
                "✨ **Température alternée** : 33°C pour le caillage, 30°C pour le brassage",
                "✨ **Salage aromatisé** : Mélanger le sel avec des épices moulues",
                "✨ **Moulage en deux temps** : Remplir à moitié, attendre 1h, compléter",
                "✨ **Affinage accéléré** : 1ère semaine à 14°C, puis 10°C",
            ]
            
            # Sélectionner selon le niveau de créativité
            num_variations = min(creativity, 3)
            selected = local_rng.sample(creative_phases, num_variations)
            
            for variation in selected:
                steps += f"{variation}\n"
        
        # Conseils supplémentaires (garder l'existant)
        steps += f"\n💡 **CONSEIL UNIQUE #{seed_value} :** "
        conseils = [
            f"Vérifiez la température toutes les 10 minutes pendant le chauffage",
            f"Utilisez un minuteur pour ne pas dépasser le temps de caillage",
            f"Notez toutes les températures et durées pour reproduire la recette",
            f"Goûtez le petit-lait : il doit être légèrement sucré, pas amer",
        ]
        steps += local_rng.choice(conseils)
        
        return steps
    
    def _generate_unique_advice(self, ingredients, cheese_type, seed_value):
        """Génère des conseils personnalisés"""
        import random
        local_rng = random.Random(seed_value)
        
        advice_list = [
            "✨ **Conseil température** : Utilisez un thermomètre digital pour plus de précision",
            "✨ **Patience** : Ne précipitez pas le caillage, laissez la nature faire son œuvre",
            "✨ **Hygiène** : Stérilisez toujours votre matériel avant utilisation",
            "✨ **Qualité du lait** : Privilégiez le lait cru pour des saveurs plus complexes",
            "✨ **Observation** : Observez l'évolution du caillé, chaque fromage est unique",
            "✨ **Carnet de notes** : Notez vos paramètres pour reproduire vos succès",
        ]
        
        # Sélectionner 2-3 conseils aléatoires
        num_advice = local_rng.randint(2, 3)
        selected = local_rng.sample(advice_list, num_advice)
        
        advice_text = "\n".join(selected)
        
        # Ajouter un conseil spécifique basé sur les ingrédients
        ingredients_str = ' '.join(ingredients).lower()
        if 'chèvre' in ingredients_str or 'chevre' in ingredients_str:
            advice_text += "\n✨ **Spécial chèvre** : Le fromage de chèvre se consomme mieux jeune, dans les 2-3 semaines"
        elif 'brebis' in ingredients_str:
            advice_text += "\n✨ **Spécial brebis** : Le lait de brebis est plus riche, réduisez légèrement la durée de caillage"
        
        return advice_text

    def _generate_amateur_recipe(
        self,
        cheese_name,
        cheese_type,
        type_info,
        ingredients,
        temp_affinage,
        conservation_info,
        accord_vin,
        accord_mets,
        epices_suggestions,
        problemes_a_eviter,
        materiel,
    ):
        """Recette AMATEUR avec LLM pour langage accessible"""

        base_recipe = f"""
RECETTE : {cheese_name}
TYPE : {cheese_type} - {type_info['description']}
DURÉE : {type_info['duree']}

INGRÉDIENTS : {self._format_user_ingredients(ingredients)}
{epices_suggestions}

MATÉRIEL : {materiel}

ÉTAPES : Chauffer 32°C → Présure → Cailler 60min → Découper → Égoutter → Saler → Affiner

AFFINAGE : {temp_affinage}
PROBLÈMES : {problemes_a_eviter}
DÉGUSTATION : {accord_mets} | Vin : {accord_vin}
CONSERVATION : {conservation_info}
"""

        if self.openrouter_enabled or self.google_ai_enabled or self.ollama_enabled:
            prompt = f"""Tu es un fromager passionné qui explique à un DÉBUTANT COMPLET.

PROFIL : Amateur qui débute, matériel basique, besoin d'encouragements

RECETTE : {base_recipe}

CONSIGNES :
- TON : Chaleureux, encourageant, simple
- LANGAGE : Comme si tu parlais à un ami, avec emojis 🧀💡
- EXPLICATIONS : Détaillées avec analogies quotidiennes
- ASTUCES : Alternatives sans matériel pro
- FORMAT : 
╔══════════════════════════════════════════════════════════════╗
║ 🏠 {cheese_name.upper()} - RECETTE MAISON
╚══════════════════════════════════════════════════════════════╝
[Introduction encourageante]
🥛 INGRÉDIENTS [où les trouver]
🔧 MATÉRIEL [alternatives simples]
📝 ÉTAPES [très détaillées]
😰 PROBLÈMES ? [solutions]
🍴 DÉGUSTATION [suggestions]
🎉 BRAVO !

Génère la recette pour débutant."""

            adapted = self.chat_with_llm(prompt, [])
            if adapted and len(adapted) > 200:
                return adapted

        return f"╔══╗\n║ 🏠 {cheese_name.upper()} ║\n╚══╝\n\n{base_recipe}\n\n🎉 Bon courage !"

    def _generate_producer_recipe(
        self,
        cheese_name,
        cheese_type,
        type_info,
        ingredients,
        temp_affinage,
        conservation_info,
        problemes_a_eviter,
    ):
        """Recette PRODUCTEUR avec LLM pour fiche technique pro"""

        base_recipe = f"""
FICHE : {cheese_name}
RÉFÉRENCE : {cheese_type} - {type_info['description']}
CYCLE : {type_info['duree']}

MATIÈRES : {self._format_user_ingredients(ingredients)}

PROCESS : Préparation → Thermisation 32°C → Emprésurage → Caillage → Tranchage → Moulage → Salage → Affinage

AFFINAGE : {temp_affinage}
CCP : {problemes_a_eviter}
CONSERVATION : {conservation_info}
"""

        if self.openrouter_enabled or self.google_ai_enabled or self.ollama_enabled:
            prompt = f"""Tu es un ingénieur agroalimentaire en technologie fromagère.

PROFIL : Producteur professionnel, matériel pro, besoin normes HACCP

RECETTE : {base_recipe}

CONSIGNES :
- TON : Professionnel, technique, précis
- VOCABULAIRE : CCP, acidité Dornic, rendement fromager, UFC/ml
- PRÉCISION : Températures ±0.5°C, dosages au gramme
- NORMES : Règlements CE, traçabilité
- FORMAT :
╔══════════════════════════════════════════════════════════════╗
║ 🏭 FICHE TECHNIQUE - {cheese_name.upper()}
╚══════════════════════════════════════════════════════════════╝
📊 CARACTÉRISTIQUES [specs techniques]
📋 MATIÈRES PREMIÈRES [traçabilité]
🔬 PROTOCOLE [CCP à chaque phase]
📊 RENDEMENT [calculs]
⚠️ POINTS CRITIQUES [actions correctives]
🔍 CONFORMITÉ [règlements CE]

Génère la fiche technique professionnelle."""

            adapted = self.chat_with_llm(prompt, [])
            if adapted and len(adapted) > 200:
                return adapted

        return f"╔══╗\n║ 🏭 {cheese_name.upper()} ║\n╚══╝\n\n{base_recipe}\n\nDocument professionnel"

    def _generate_trainer_recipe(
        self,
        cheese_name,
        cheese_type,
        type_info,
        ingredients,
        temp_affinage,
        conservation_info,
        accord_vin,
        accord_mets,
        problemes_a_eviter,
        materiel,
    ):
        """Recette FORMATEUR avec LLM pour support pédagogique"""

        base_recipe = f"""
SUPPORT : {cheese_name}
TYPE : {cheese_type} - {type_info['description']}
MODULE : {type_info['duree']}

MATÉRIEL : {materiel}
INGRÉDIENTS : {self._format_user_ingredients(ingredients)}

ÉTAPES : Préparation → Chauffage → Emprésurage → Caillage → Découpage → Égouttage → Salage → Affinage

AFFINAGE : {temp_affinage}
ERREURS : {problemes_a_eviter}
DÉGUSTATION : {accord_mets} | {accord_vin}
CONSERVATION : {conservation_info}
"""

        if self.openrouter_enabled or self.google_ai_enabled or self.ollama_enabled:
            prompt = f"""Tu es un formateur en technologie fromagère.

PROFIL : Formateur qui anime des ateliers pour groupes

RECETTE : {base_recipe}

CONSIGNES :
- TON : Pédagogique, structuré
- STRUCTURE : Objectifs → Théorie → Pratique → Évaluation
- CONTENU : Explication scientifique simple + démonstration
- FORMAT :
╔══════════════════════════════════════════════════════════════╗
║ 🎓 SUPPORT PÉDAGOGIQUE - {cheese_name.upper()}
╚══════════════════════════════════════════════════════════════╝
📚 OBJECTIFS [savoir, savoir-faire, savoir-être]
📖 PRÉREQUIS
🥛 MATÉRIEL ATELIER
🔬 PROCESSUS COMMENTÉ [séquences avec théorie]
⚠️ ERREURS [analyse + correction]
🍴 DÉGUSTATION ANALYTIQUE [grille sensorielle]
📝 ÉVALUATION [critères + barème]
🚀 POUR ALLER PLUS LOIN

Génère le support pédagogique."""

            adapted = self.chat_with_llm(prompt, [])
            if adapted and len(adapted) > 200:
                return adapted

        return f"╔══╗\n║ 🎓 {cheese_name.upper()} ║\n╚══╝\n\n{base_recipe}\n\nSupport formation"

    def generate_recipe_creative(
        self,
        ingredients,
        cheese_type,
        constraints,
        creativity_level,
        texture_preference,
        affinage_duration,
        spice_intensity,
        experience_level=None,
    ):
        """Génère une recette avec mode créatif et micro-choix UNIQUE"""

        print(f"🧀 Génération créative UNIQUE avec:")
        print(f"  - Ingrédients: {ingredients}")
        print(f"  - Type: {cheese_type}")
        print(f"  - Créativité: {creativity_level}")
        print(f"  - Texture: {texture_preference}")
        print(f"  - Affinage: {affinage_duration}")
        print(f"  - Épices: {spice_intensity}")
        print(f"  - Niveau: {experience_level}")
        
        # ===== GÉNÉRER UNE BASE DE RECETTE DIFFÉRENTE SELON LE PROFIL =====
        
        # 1. AMATEUR : Recette simple et rapide
        if experience_level == "🧀 Amateur":
            cheese_type_clean = self._determine_amateur_cheese_type(ingredients)
            recette_speciale = {
                "difficulte": "Facile",
                "duree_totale": "24-48h max",
                "equipement": "basique",
                "focus": "succès rapide"
            }
        
        # 2. PRODUCTEUR : Recette technique et précise  
        elif experience_level == "🏭 Producteur":
            cheese_type_clean = self._determine_producer_cheese_type(ingredients)
            recette_speciale = {
                "difficulte": "Technique",
                "duree_totale": "2-8 semaines",
                "equipement": "professionnel",
                "focus": "rendement optimal"
            }
        
        # 3. FORMATEUR : Recette pédagogique
        elif experience_level == "🎓 Formateur":
            cheese_type_clean = self._determine_trainer_cheese_type(ingredients)
            recette_speciale = {
                "difficulte": "Pédagogique",
                "duree_totale": "variable",
                "equipement": "démonstration",
                "focus": "compréhension"
            }

        # Initialisation des variables
        is_valid = False
        recipe = ""
        lait = None

        try:
            # Validation de base
            valid, message = self.validate_ingredients(ingredients)
            if not valid:
                return message
            
            ingredients_list = [ing.strip() for ing in ingredients.split(',')]
            
            # Déterminer le type si non spécifié
            if cheese_type == "Laissez l'IA choisir":
                cheese_type_clean = self._determine_type_based_on_ingredients(ingredients_list)
            else:
                cheese_type_clean = cheese_type
            
            # ===== VALIDATION LAIT/PÂTE =====
            # Extraire le type de lait des ingrédients
            lait = self._extract_lait_from_text(' '.join(ingredients_list))
            
            # Vérifier la compatibilité si un lait et un type de pâte sont définis
            if lait and cheese_type_clean not in ["Fromage artisanal", "Laissez l'IA choisir"]:
                is_valid, reason = self._validate_combination(lait, cheese_type_clean)
                if not is_valid:
                    alternatives = self._suggest_alternatives(lait, cheese_type_clean)
                    return f"""
❌ **IMPOSSIBLE DE CRÉER CETTE RECETTE**

**Combinaison rejetée :** {lait.capitalize()} + {cheese_type_clean}

{reason}

**💡 Alternatives compatibles avec le lait de {lait} :**
{alternatives}

**Pour continuer, modifiez :**
• Soit vos ingrédients (changez le type de lait)
• Soit le type de fromage (choisissez-en un compatible)
"""
            # ===== FIN VALIDATION =====
            
            # Générer une recette UNIQUE
            recipe = self._generate_unique_recipe(
                ingredients_list, 
                cheese_type_clean, 
                constraints,
                creativity_level,
                experience_level or "🧀 Amateur"
            )
            
            # Appliquer les micro-choix
            recipe = self._apply_micro_choices_to_recipe(
                recipe, 
                texture_preference,
                spice_intensity,
                affinage_duration,
                creativity_level
            )
            
            # Sauvegarder
            self._save_to_history(ingredients_list, cheese_type_clean, constraints, recipe)
            
            return recipe
        
        except Exception as e:
            error_msg = f"❌ Erreur lors de la génération de la recette : {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg
    
    def _determine_amateur_cheese_type(self, ingredients):
        """Pour amateur : choisit toujours un fromage FACILE et RAPIDE"""
        ingredients_lower = ' '.join(ingredients).lower()
    
        # Amateur = fromage frais (toujours)
        if "chèvre" in ingredients_lower or "chevre" in ingredients_lower:
            return "Fromage de chèvre frais"
        elif "brebis" in ingredients_lower:
            return "Fromage de brebis frais"
        elif "vache" in ingredients_lower:
            return "Fromage frais nature"
        else:
            return "Fromage frais maison"

    def _determine_producer_cheese_type(self, ingredients):
        """Pour producteur : choisit un fromage avec VALEUR AJOUTÉE"""
        ingredients_lower = ' '.join(ingredients).lower()
    
        # Producteur = fromage à affiner (meilleure marge)
        if "chèvre" in ingredients_lower:
            return "Bûche de chèvre affinée"
        elif "brebis" in ingredients_lower:
            return "Fromage de brebis à pâte pressée"
        elif "vache" in ingredients_lower:
            # Choisir aléatoirement entre différents fromages de vache
            options = ["Camembert", "Brie", "Tomme de vache", "Fromage à pâte persillée"]
            import random
            return random.choice(options)
        else:
            return "Fromage à pâte pressée non cuite"

    def _determine_trainer_cheese_type(self, ingredients):
        """Pour formateur : choisit un fromage PÉDAGOGIQUE"""
        ingredients_lower = ' '.join(ingredients).lower()
        
        # Formateur = fromage qui montre bien les étapes
        if "chèvre" in ingredients_lower:
            return "Fromage de chèvre cendré"  # Montre bien les étapes
        elif "brebis" in ingredients_lower:
            return "Fromage de brebis à pâte pressée"  # Long processus éducatif
        elif "vache" in ingredients_lower:
            return "Pâte molle à croûte fleurie"  # Permet de voir l'évolution
        else:
            return "Fromage frais (atelier découverte)"
    
    def _apply_micro_choices_to_recipe(self, recipe, texture, spice_intensity, affinage, creativity):
        """Applique les micro-choix à une recette existante"""
    
        modifications = []
        
        # Texture
        if texture == "Très crémeux":
            modifications.append("🎯 **Texture crémeuse optimisée :**")
            modifications.append("- Augmenter la température à 34°C")
            modifications.append("- Réduire le temps de caillage de 15%")
            modifications.append("- Ajouter 50ml de crème fraîche")
        
        elif texture == "Très ferme":
            modifications.append("🎯 **Texture ferme optimisée :**")
            modifications.append("- Augmenter la présure de 20%")
            modifications.append("- Presser pendant 2h supplémentaires")
            modifications.append("- Ajouter 5g de sel supplémentaire")
        
        # Épices
        if spice_intensity == "Modéré":
            modifications.append("🌶️ **Aromatisation modérée :**")
            modifications.append("- Ajouter 1 cuillère à café d'herbes de Provence")
            modifications.append("- Poivrer généreusement en surface")
        
        elif spice_intensity == "Intense":
            modifications.append("🌶️ **Aromatisation intense :**")
            modifications.append("- Ajouter 2 cuillères à café d'épices mélangées")
            modifications.append("- Enrober de poivre concassé et d'ail")
            modifications.append("- Infuser le lait avec 1 bouquet garni")
        
        # Affinage
        if affinage > 8:
            modifications.append(f"⏱️ **Affinage long ({affinage} semaines) :**")
            modifications.append("- Température d'affinage : 12°C")
            modifications.append("- Humidité : 90%")
            modifications.append("- Retourner tous les 2 jours")
        
        # Créativité
        if creativity >= 2:
            modifications.append("🎨 **Variations créatives :**")
            creative_options = [
                "- Incorporer des noix concassées dans le caillé",
                "- Enrober de cendres végétales",
                "- Ajouter des pétales de rose séchés",
                "- Infuser le lait avec du thé Earl Grey"
            ]
            selected = self.rng.sample(creative_options, k=min(creativity, len(creative_options)))
            modifications.extend(selected)
        
        # Ajouter les modifications à la recette
        if modifications:
            recipe += "\n\n" + "🎛️ MICRO-CHOIX APPLIQUÉS\n"
            recipe += "━" * 50 + "\n"
            for mod in modifications:
                recipe += f"{mod}\n"
        
        return recipe
    
    def _add_creative_variations(self, recipe, creativity_level, cheese_type, lait):
        """Ajoute des variations créatives selon le niveau"""

        creative_section = "\n\n" + "=" * 70 + "\n"
        creative_section += "🎨 VARIATIONS CRÉATIVES\n"
        creative_section += "=" * 70 + "\n\n"

        variations = []

        # Niveau 1 : Suggestions simples
        if creativity_level >= 1:
            variations.append(self._get_simple_variation(cheese_type, lait))

        # Niveau 2 : Variations fusion
        if creativity_level >= 2:
            variations.append(self._get_fusion_variation(cheese_type, lait))

        # Niveau 3 : Expérimental
        if creativity_level >= 3:
            variations.append(self._get_experimental_variation(cheese_type, lait))

        for i, var in enumerate(variations, 1):
            # Utiliser .get() avec valeur par défaut pour éviter KeyError
            creative_section += (
                f"### Variation {i} : {var.get('title', 'Variation créative')}\n\n"
            )
            creative_section += (
                f"**Concept :** {var.get('concept', 'Création originale')}\n\n"
            )

        # Ingrédients
        ingredients = var.get("ingredients", [])
        if ingredients:
            creative_section += f"**Ingrédients supplémentaires :**\n"
            for ing in ingredients:
                creative_section += f"- {ing}\n"
            creative_section += "\n"

        # Technique - AVEC .get() pour éviter l'erreur
        technique = var.get("technique", "Incorporer selon votre méthode habituelle")
        creative_section += f"**Technique :** {technique}\n\n"
        creative_section += "---\n\n"

        return recipe + creative_section

    def _get_simple_variation(self, cheese_type, lait):
        """Variation simple : herbes et épices"""

        variations = {
            "Fromage frais": {
                "title": "Fromage frais aux fleurs",
                "concept": "Ajout de fleurs comestibles pour un fromage élégant",
                "ingredients": [
                    "Pétales de rose séchés",
                    "Lavande culinaire",
                    "Bleuet",
                ],
                "technique": "Incorporer les fleurs lors du moulage, parsemer sur le dessus",
            },
            "Pâte molle": {
                "title": "Pâte molle truffée",
                "concept": "Infusion de truffe pour un fromage luxueux",
                "ingredients": ["Huile de truffe (5ml)", "Copeaux de truffe"],
                "technique": "Badigeonner la croûte avec l'huile de truffe pendant l'affinage",
            },
            "Pâte pressée non cuite": {
                "title": "Tomme aux noix et miel",
                "concept": "Enrobage sucré-salé original",
                "ingredients": ["Noix concassées", "Miel de montagne", "Thym"],
                "   technique": "Enrober le fromage de noix et miel avant l'affinage final",
            },
            "Pâte pressée cuite": {
                "title": "Comté aux herbes de montagne",
                "concept": "Fromage alpin aromatisé",
                "ingredients": ["Génépi", "Fleurs de foin", "Ail des ours"],
                "technique": "Affiner sur une litière d'herbes séchées",
            },
            "Pâte persillée": {
                "title": "Bleu au miel et noix",
                "concept": "Association sucrée-salée gourmande",
                "ingredients": ["Miel de châtaignier", "Noix fraîches"],
                "technique": "Servir avec un filet de miel et des noix concassées",
            },
        }
        # Variation par défaut si type non trouvé
        default = {
            "title": "Variation classique",
            "concept": "Fromage aromatisé aux herbes",
            "ingredients": ["Herbes de Provence", "Ail séché"],
            "technique": "Mélanger les herbes dans le caillé avant moulage",
        }

        return variations.get(cheese_type, variations["Fromage frais"])

    def _get_fusion_variation(self, cheese_type, lait):
        """Variation fusion : inspiration internationale"""

        fusions = [
            {
                "title": "Inspiration méditerranéenne",
                "concept": "Fromage aux saveurs du sud",
                "ingredients": [
                    "Tomates séchées",
                    "Olives noires",
                    "Origan",
                    "Huile d'olive",
                ],
                "technique": "Incorporer dans le caillé avant moulage",
            },
            {
                "title": "Inspiration japonaise",
                "concept": "Fromage au yuzu et sésame noir",
                "ingredients": [
                    "Zeste de yuzu",
                    "Graines de sésame noir",
                    "Algue nori émincée",
                ],
                "technique": "Enrober le fromage de sésame et ajouter le yuzu en surface",
            },
            {
                "title": "Inspiration indienne",
                "concept": "Fromage aux épices chaudes",
                "ingredients": [
                    "Curry doux",
                    "Gingembre frais râpé",
                    "Coriandre",
                    "Curcuma",
                ],
                "technique": "Mélanger les épices au sel de salage",
            },
            {
                "title": "Inspiration mexicaine",
                "concept": "Fromage piquant et fumé",
                "ingredients": ["Piment chipotle", "Coriandre fraîche", "Lime"],
                "technique": "Incorporer le piment fumé dans le caillé",
            },
        ]

        return self.rng.choice(fusions)

    def _get_experimental_variation(self, cheese_type, lait):
        """Variation expérimentale : très créatif"""

        experiments = [
            {
                "title": "Fromage lacto-fermenté aux légumes",
                "concept": "Double fermentation avec légumes crus",
                "ingredients": [
                    "Carottes râpées",
                    "Betterave",
                    "Gingembre",
                    "Kombucha",
                ],
                "technique": "Ajouter les légumes lacto-fermentés pendant l'égouttage",
            },
            {
                "title": "Fromage aux algues et spiruline",
                "concept": "Superfood fromager, riche en protéines",
                "ingredients": ["Spiruline en poudre", "Wakame", "Graines de chia"],
                "technique": "Mélanger dans le lait avant caillage pour couleur verte",
            },
            {
                "title": "Fromage au café et cacao",
                "concept": "Dessert fromager original",
                "ingredients": ["Café espresso", "Poudre de cacao", "Vanille", "Miel"],
                "technique": "Infuser le lait avec café/cacao avant emprésurage",
            },
            {
                "title": "Fromage fumé aux bois exotiques",
                "concept": "Fumage à froid avec bois spéciaux",
                "ingredients": [
                    "Copeaux de hêtre",
                    "Copeaux de pommier",
                    "Romarin séché",
                ],
                "technique": "Fumer à froid pendant 2-3 heures après séchage",
            },
            {
                "title": "Fromage au thé matcha",
                "concept": "Fusion franco-japonaise délicate",
                "ingredients": [
                    "Thé matcha premium",
                    "Gingembre confit",
                    "Sésame blanc",
                ],
                "technique": "Infuser le lait avec matcha, parsemer de sésame",
            },
        ]

        return self.rng.choice(experiments)

    def _determine_type(self, ingredients):
        """Détermine le type selon les ingrédients en respectant les compatibilités"""
        ingredients_str = " ".join(ingredients).lower()

        # Extraire le type de lait
        lait = self._extract_lait_from_text(ingredients_str)

        # Détecter des indices sur le type souhaité
        if "citron" in ingredients_str or "vinaigre" in ingredients_str:
            return "Fromage frais"
        elif "bleu" in ingredients_str or "roquefort" in ingredients_str:
            return "Pâte persillée"

        # Sinon, choisir un type compatible avec le lait détecté
        if lait:
            rules = self.knowledge_base["regles_compatibilite"]
            for combo in rules["lait_x_type_pate"]["combinaisons_valides"]:
                if combo["lait"] == lait.lower():
                    compatibles = combo["types_pate_compatibles"]

                    # Logique de choix selon les ingrédients
                    if any(x in ingredients_str for x in ["herbe", "épice", "aromate"]):
                        # Si aromates : privilégier fromage frais ou pressée non cuite
                        if "Fromage frais" in compatibles:
                            return "Fromage frais"
                        elif "Pâte pressée non cuite" in compatibles:
                            return "Pâte pressée non cuite"

                    # Par défaut : choisir le premier type compatible (généralement le plus simple)
                    if compatibles:
                        return compatibles[0]

        # Si pas de lait détecté, fromage frais par défaut (le plus simple et universel)
        return "Fromage frais"

    def _determine_type_based_on_ingredients(self, ingredients_list):
        """Détermine le type de fromage basé sur les ingrédients de manière INTELLIGENTE"""
        ingredients_str = " ".join(ingredients_list).lower()

        print(f"🔍 Analyse ingrédients pour type: {ingredients_str}")

        # 1. Détecter le lait
        lait = self._extract_lait_from_text(ingredients_str)

        # 2. Analyser les ingrédients spéciaux
        has_herbs = any(
            word in ingredients_str
            for word in ["herbe", "thym", "romarin", "basilic", "origan"]
        )
        has_spices = any(
            word in ingredients_str for word in ["poivre", "cumin", "piment", "curry"]
        )
        has_blue_mold = any(
            word in ingredients_str for word in ["bleu", "roquefort", "penicillium"]
        )

        # 3. Détecter contraintes techniques
        has_long_aging = any(
            word in ingredients_str for word in ["affinage", "long", "mois", "cave"]
        )
        has_fresh = any(
            word in ingredients_str
            for word in ["frais", "blanc", "rapide", "consommation"]
        )

        # 4. Règles de décision
        if has_blue_mold:
            return "Pâte persillée"

        if has_fresh or "citron" in ingredients_str or "vinaigre" in ingredients_str:
            return "Fromage frais"

        if has_long_aging:
            if lait == "vache":
                return (
                    "Pâte pressée cuite"
                    if self.rng.random() > 0.5
                    else "Pâte pressée non cuite"
                )
            else:
                return "Pâte pressée non cuite"

        if has_herbs or has_spices:
            if lait == "chèvre":
                return "Fromage frais"  # Les herbes vont mieux avec fromage frais
            elif lait == "brebis":
                return "Pâte pressée non cuite"
            else:
                return "Pâte molle"

        # 5. Par défaut, basé sur le lait
        default_types = {
            "chèvre": "Fromage frais",
            "brebis": "Pâte pressée non cuite",
            "vache": self.rng.choice(["Pâte molle", "Fromage frais"]),
            "bufflonne": "Fromage frais",
            None: self.rng.choice(["Fromage frais", "Pâte molle"]),
        }

        return default_types.get(lait, "Fromage frais")

    def _get_type_info(self, cheese_type):
        """Récupère les infos du type de fromage"""
        for key, value in self.knowledge_base["types_pate"].items():
            if key.lower() in cheese_type.lower():
                return value
        return self.knowledge_base["types_pate"]["Fromage frais"]

    def _get_temperature_affinage(self, cheese_type):
        """Récupère la température d'affinage depuis la base"""
        if "temperatures_affinage" not in self.knowledge_base:
            return "10-12°C, 85-90% humidité"

        for key, value in self.knowledge_base["temperatures_affinage"].items():
            if key.lower() in cheese_type.lower():
                return value
        return "10-12°C, 85-90% humidité"

    def _get_conservation_info(self, cheese_type):
        """Récupère les infos de conservation"""
        if "conservation" not in self.knowledge_base:
            return "2-3 semaines au réfrigérateur dans papier adapté"

        for key, value in self.knowledge_base["conservation"].items():
            if key.lower() in cheese_type.lower():
                return value

        # Chercher par mot-clé
        if "frais" in cheese_type.lower():
            return self.knowledge_base["conservation"].get(
                "Fromage frais", "3-5 jours au frigo"
            )

        return "2-3 semaines au réfrigérateur dans papier adapté"

    def _get_accord_vin(self, cheese_type):
        """Récupère les accords vins"""
        if "accords_vins" not in self.knowledge_base:
            return "Vin rouge de caractère ou blanc sec selon préférence"

        # Recherche exacte
        for key, value in self.knowledge_base["accords_vins"].items():
            if key.lower() in cheese_type.lower():
                return value

        # Recherche par mot-clé
        if "frais" in cheese_type.lower():
            return self.knowledge_base["accords_vins"].get(
                "Fromage frais nature", "Vin blanc sec et vif"
            )
        elif "chèvre" in cheese_type.lower():
            return self.knowledge_base["accords_vins"].get(
                "Chèvre frais", "Sancerre, Sauvignon blanc"
            )
        elif "molle" in cheese_type.lower() or "camembert" in cheese_type.lower():
            return self.knowledge_base["accords_vins"].get(
                "Brie, Camembert", "Champagne ou rouge léger"
            )

        return "Vin rouge de caractère ou blanc sec selon préférence"

    def _get_accord_mets(self, cheese_type):
        """Récupère les accords mets"""
        if "accords_mets" not in self.knowledge_base:
            return "Pain frais, fruits secs, miel"

        for key, value in self.knowledge_base["accords_mets"].items():
            if key.lower() in cheese_type.lower():
                return value

        # Par mot-clé
        if "frais" in cheese_type.lower():
            return self.knowledge_base["accords_mets"].get(
                "Fromage frais", "Pain complet, fruits rouges, miel"
            )
        elif "chèvre" in cheese_type.lower():
            return self.knowledge_base["accords_mets"].get(
                "Chèvre", "Pain grillé, miel, salade verte"
            )

        return "Pain de campagne, fruits secs, confitures"

    def _suggest_epices(self, ingredients, cheese_type):
        """Suggère des épices selon le type"""
        suggestions = "\n💡 SUGGESTIONS D'AROMATES (depuis la base de connaissances)\n"
        suggestions += "━" * 70 + "\n"

        # Associations classiques
        if "associations_classiques" in self.knowledge_base:
            for key, value in self.knowledge_base["associations_classiques"].items():
                if key.lower() in cheese_type.lower() or any(
                    k.lower() in cheese_type.lower() for k in key.split()
                ):
                    suggestions += f"**Idéal pour ce type** : {value}\n\n"
                    break

        # Techniques d'aromatisation
        if "techniques_aromatisation" in self.knowledge_base:
            suggestions += "**Techniques d'incorporation** :\n"
            for tech, desc in list(
                self.knowledge_base["techniques_aromatisation"].items()
            )[:3]:
                suggestions += f"• {tech} : {desc}\n"
            suggestions += "\n"

        # Dosages
        if "dosages_recommandes" in self.knowledge_base:
            suggestions += "**Dosages recommandés** :\n"
            for ing, dosage in list(self.knowledge_base["dosages_recommandes"].items())[
                :4
            ]:
                suggestions += f"• {ing} : {dosage}\n"

        return suggestions

    def _get_problemes_pertinents(self, cheese_type):
        """Liste les problèmes courants à éviter"""
        if "problemes_courants" not in self.knowledge_base:
            return "Respecter températures et temps de repos"

        problemes = ""
        # Prendre les 5 problèmes les plus courants
        problemes_items = list(self.knowledge_base["problemes_courants"].items())
        selection = self.rng.sample(problemes_items, k=min(5, len(problemes_items)))
        for prob, sol in selection:
            problemes += f"❌ **{prob}**\n"
            problemes += f"   ✅ {sol}\n\n"

        return problemes

    def _get_materiel_debutant(self):
        """Liste le matériel pour débutants"""
        if "materiel_indispensable" not in self.knowledge_base:
            return (
                "• Grande casserole inox\n• Thermomètre\n• Moule à fromage\n• Étamine"
            )

        materiel_list = self.knowledge_base["materiel_indispensable"].get(
            "Pour débuter", []
        )
        return "\n".join([f"• {item}" for item in materiel_list])

    def _get_egouttage_time(self, cheese_type):
        """Durée d'égouttage selon le type"""
        if "frais" in cheese_type.lower():
            return "2-4 heures"
        elif "molle" in cheese_type.lower():
            return "12-18 heures"
        else:
            return "18-24 heures"

    def _get_soins_affinage(self, cheese_type):
        """Instructions de soins pendant l'affinage"""
        if "frais" in cheese_type.lower():
            return "Pas d'affinage nécessaire, consommer rapidement"
        elif "molle" in cheese_type.lower():
            return "Retourner tous les 2 jours, brosser si croûte blanche apparaît"
        elif "pressée" in cheese_type.lower():
            return "Retourner quotidiennement la 1ère semaine, puis 2x/semaine"
        else:
            return "Retourner régulièrement, surveiller l'apparition des moisissures"

    def _get_tasting_time(self, cheese_type):
        """Moment optimal de dégustation"""
        type_info = self._get_type_info(cheese_type)
        duree = type_info.get("duree", "")

        if "frais" in cheese_type.lower():
            return "Immédiatement après fabrication ou dans les 3-5 jours"
        elif "2-8 semaines" in duree:
            return "Après 3-6 semaines d'affinage minimum"
        elif "mois" in duree:
            return "Après la durée d'affinage indiquée, goûter régulièrement"
        else:
            return "Selon votre goût, goûter à différents stades d'affinage"

    def _get_variantes(self, cheese_type, ingredients):
        """Suggère des variantes créatives"""
        variantes = ""

        if "epices_et_aromates" in self.knowledge_base:
            variantes += "1. **Version aux herbes** : "
            herbes = self.rng.sample(
                self.knowledge_base["epices_et_aromates"].get("Herbes fraîches", []),
                k=3,
            )
            variantes += f"Incorporer {', '.join(herbes[:3][:])}\n\n"

            variantes += "2. **Version épicée** : "
            epices = self.knowledge_base["epices_et_aromates"].get("Épices chaudes", [])
            variantes += f"Enrober de {', '.join(epices[:2])}\n\n"

            variantes += "3. **Version gourmande** : "
            accomp = self.knowledge_base["epices_et_aromates"].get(
                "Accompagnements dans la pâte", []
            )
            variantes += f"Ajouter {', '.join(accomp[:3])}\n\n"
        else:
            variantes += "1. Version aux herbes : Basilic, thym, romarin\n"
            variantes += "2. Version poivrée : Enrober de poivre concassé\n"
            variantes += "3. Version aux noix : Incorporer noix concassées\n"

        return variantes

    def _get_conseils_fromager(self):
        """Conseils généraux du maître fromager"""
        return """✨ **Hygiène irréprochable** : Stériliser TOUT le matériel à l'eau bouillante

✨ **Température précise** : ±2°C peut totalement changer le résultat final

✨ **Patience** : Un bon fromage ne se précipite pas, respecter les temps

✨ **Qualité du lait** : Privilégier lait cru ou pasteurisé (JAMAIS UHT)

✨ **Carnet de bord** : Noter températures, durées, résultats pour progresser

✨ **Cave d'affinage DIY** : Une glacière avec bol d'eau + hygromètre suffit

✨ **Goûter régulièrement** : Le fromage évolue, trouver votre stade préféré"""

    def _generate_creative_name(self, cheese_type, ingredients):
        """Génère un nom créatif pour le fromage"""
        ingredients_str = " ".join(ingredients).lower()

        # Briques génériques
        base = ["Velours", "Délice", "Nuage", "Trésor", "Secret", "Essence"]
        lieu = ["de Cave", "du Terroir", "des Prés", "Lacté", "Artisan"]
        style = ["Fondant", "Rustique", "Crémeux", "Affiné", "Doux"]

        if "chèvre" in ingredients_str:
            base = ["Chèvre", "Caprice", "Blanc"]
            qualifier = ["des Prés", "Lacté", "Frais"]
        elif "brebis" in ingredients_str:
            base = ["Brebis", "Douceur", "Trésor"]
            qualifier = ["Pastorale", "de Bergère", "Montagnard"]
        elif "herbe" in ingredients_str or "épice" in ingredients_str:
            base = ["Jardin", "Bouquet", "Pré"]
            qualifier = ["Fromager", "Lacté", "Fleuri"]
        elif "frais" in cheese_type.lower():
            base = ["Blanc", "Nuage", "Fraîcheur"]
            qualifier = ["Matinale", "Lactée", "Pure"]
        elif "molle" in cheese_type.lower():
            base = ["Velours", "Crème", "Délice"]
            qualifier = ["de Cave", "d'Artisan", "Fondant"]
        elif "pressée" in cheese_type.lower():
            base = ["Roc", "Meule", "Pierre"]
            qualifier = ["du Terroir", "Tradition", "Lactée"]
        else:
            base = base
            qualifier = ["Maison", "Artisanale", "Fromagère"]

        return (
            f"{self.rng.choice(base)} {self.rng.choice(lieu)} {self.rng.choice(style)}"
        )

    def _format_user_ingredients(self, ingredients):
        """Formate joliment les ingrédients utilisateur"""
        formatted = ""
        for ing in ingredients:
            formatted += f"• {ing.capitalize()}\n"
        return formatted

    def _add_constraints_note(self, constraints):
        """Ajoute une note sur les contraintes"""
        if not constraints or constraints.strip() == "":
            return ""

        note = f"""
⚙️ ADAPTATIONS AUX CONTRAINTES : {constraints.upper()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
        constraints_lower = constraints.lower()

        if "végétarien" in constraints_lower or "vegetarien" in constraints_lower:
            note += "✓ **Présure végétale** : Utiliser présure d'origine végétale (chardon, figuier)\n"
            note += "✓ Vérifier que les ferments sont non-animaux\n\n"

        if "rapide" in constraints_lower:
            note += "✓ **Version rapide** : Privilégier fromage frais (4-6h total)\n"
            note += "✓ Utiliser citron pour caillage accéléré (20 min)\n\n"

        if "lactose" in constraints_lower:
            note += "✓ **Sans lactose** : Les fromages affinés contiennent naturellement peu de lactose\n"
            note += "✓ Utiliser lait délactosé ou lait de chèvre (plus digeste)\n\n"

        if "vegan" in constraints_lower or "végétalien" in constraints_lower:
            note += "✓ **Version végane** : Utiliser lait végétal (soja, cajou enrichi en calcium)\n"
            note += "✓ Coagulant : agar-agar, tapioca, ou acide citrique\n"
            note += "✓ Ferments : probiotiques en poudre ou rejuvelac\n\n"

        return note

    def _add_constraints_note(self, constraints):
        """Ajoute une note sur les contraintes"""
        if not constraints or constraints.strip() == "":
            return ""

        return f"""
⚙️ ADAPTATION AUX CONTRAINTES : {constraints.upper()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Adaptations suggérées selon vos contraintes.
"""

        # ===== MÉTHODES DE CHAT LLM =====

    def _test_ollama_connection(self):
        """Teste la connexion à Ollama (local)"""
        try:
            response = requests.post(
                self.ollama_url,
                json={"model": self.ollama_model, "prompt": "test", "stream": False},
                timeout=3,
            )
            return response.status_code == 200
        except:
            return False

    def chat_with_llm(self, user_message: str, conversation_history=None) -> str:
        """
        Chat intelligent avec fallback sur plusieurs fournisseurs gratuits
        Priorité: 1. OpenRouter → 2. Google AI → 3. Ollama → 4. Hugging Face → 5. Fallback local
        """
        print(f"💬 Question reçue: '{user_message[:100]}...'")

        # DEBUG: État des LLMs (avec vérification d'attributs pour éviter les erreurs)
        print("🔍 ÉTAT LLMs - ", end="")
        if hasattr(self, "openrouter_enabled"):
            print(f"OpenRouter: {self.openrouter_enabled}, ", end="")
        if hasattr(self, "google_ai_enabled"):
            print(f"Google AI: {self.google_ai_enabled}, ", end="")
        if hasattr(self, "ollama_enabled"):
            print(f"Ollama: {self.ollama_enabled}, ", end="")
        if hasattr(self, "together_enabled"):
            print(f"Together: {self.together_enabled}")
        print()

        # ===== TENTATIVE AVEC LES LLMS =====

        # 1. OPENROUTER (priorité haute - gratuit avec quotas)
        if hasattr(self, "openrouter_enabled") and self.openrouter_enabled:
            try:
                print("  🤖 Tentative OpenRouter...")
                # Vérifier si la méthode existe
                if hasattr(self, "_chat_openrouter"):
                    response = self._chat_openrouter(user_message, conversation_history)
                    if response and response.strip():
                        print(f"  ✅ Réponse OpenRouter ({len(response)} caractères)")
                        return response
                else:
                    print("  ⚠️ Méthode _chat_openrouter manquante!")
            except Exception as e:
                print(f"  ⚠️ OpenRouter échoué: {type(e).__name__}")

        # 2. GOOGLE AI / GEMINI
        if hasattr(self, "google_ai_enabled") and self.google_ai_enabled:
            try:
                print("  🤖 Tentative Google AI...")
                if hasattr(self, "_chat_google_ai"):
                    response = self._chat_google_ai(user_message, conversation_history)
                    if response and response.strip():
                        print(f"  ✅ Réponse Google AI ({len(response)} caractères)")
                        return response
            except Exception as e:
                print(f"  ⚠️ Google AI échoué: {type(e).__name__}")

        # 3. TOGETHER AI (si vous avez ajouté cette méthode)
        if hasattr(self, "together_enabled") and self.together_enabled:
            try:
                print("  🤖 Tentative Together AI...")
                if hasattr(self, "_chat_together_ai"):
                    response = self._chat_together_ai(
                        user_message, conversation_history
                    )
                    if response and response.strip():
                        print(f"  ✅ Réponse Together AI ({len(response)} caractères)")
                        return response
            except Exception as e:
                print(f"  ⚠️ Together AI échoué: {type(e).__name__}")

        # 4. OLLAMA (local)
        if hasattr(self, "ollama_enabled") and self.ollama_enabled:
            try:
                print("  🤖 Tentative Ollama...")
                if hasattr(self, "_chat_ollama"):
                    response = self._chat_ollama(user_message, conversation_history)
                    if response and response.strip():
                        print(f"  ✅ Réponse Ollama ({len(response)} caractères)")
                        return response
            except Exception as e:
                print(f"  ⚠️ Ollama échoué: {type(e).__name__}")

        # 5. DEEPSEEK (si vous le gardez)
        if hasattr(self, "deepseek_enabled") and self.deepseek_enabled:
            try:
                print("  🤖 Tentative DeepSeek...")
                if hasattr(self, "_chat_deepseek"):
                    response = self._chat_deepseek(user_message, conversation_history)
                    if response and response.strip():
                        print(f"  ✅ Réponse DeepSeek ({len(response)} caractères)")
                        return response
            except Exception as e:
                print(f"  ⚠️ DeepSeek échoué: {type(e).__name__}")

        # 6. HUGGING FACE
        if hasattr(self, "hf_inference_enabled") and self.hf_inference_enabled:
            try:
                print("  🤖 Tentative Hugging Face...")
                if hasattr(self, "_chat_huggingface"):
                    response = self._chat_huggingface(
                        user_message, conversation_history
                    )
                    if response and response.strip():
                        print(f"  ✅ Réponse Hugging Face ({len(response)} caractères)")
                        return response
            except Exception as e:
                print(f"  ⚠️ Hugging Face échoué: {type(e).__name__}")

        # 7. FALLBACK LOCAL (toujours disponible)
        print("  🧠 Tous les LLMs ont échoué → fallback local")
        return self._fallback_chat_response(user_message)

    def _get_cheese_context(self, question: str) -> str:
        """Extrait des infos de la base pour aider le LLM"""
        # Recherche simple
        if "cantal" in question:
            return "Le Cantal est un fromage AOP d'Auvergne au lait de vache, pâte pressée non cuite."
        elif "roquefort" in question:
            return "Le Roquefort est un fromage bleu AOP au lait de brebis, affiné en caves."
        elif "camembert" in question:
            return "Le Camembert est un fromage normand au lait de vache, à pâte molle et croûte fleurie."
        elif "chèvre" in question:
            return "Les fromages de chèvre incluent Crottin de Chavignol, Sainte-Maure, etc. Tous au lait de chèvre."
        return None

    def chat_with_together_ai(self, user_message, conversation_history=None):
        """Utilise Together AI (gratuit avec 25$ de crédit)"""
        try:
            api_key = os.environ.get("TOGETHER_API_KEY")
            if not api_key:
                return None

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            messages = [
                {
                    "role": "system",
                    "content": "Tu es un expert fromager français. Réponds avec précision et passion.",
                }
            ]

            if conversation_history:
                messages.extend(conversation_history[-5:])

            messages.append({"role": "user", "content": user_message})

            payload = {
                "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "messages": messages,
                "max_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.9,
            }

            response = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"❌ Together AI error: {response.status_code}")
                return None

        except Exception as e:
            print(f"❌ Together AI exception: {e}")
            return None

    def _chat_huggingface(
        self, user_message: str, conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """Utilise Hugging Face Inference API"""
        try:
            print(f"    🔑 HF Token: {self.hf_token[:10]}...")

            headers = {"Authorization": f"Bearer {self.hf_token}"}

            prompt = """<s>[INST] Tu es un expert fromager français. Réponds aux questions de manière précise et amicale. [/INST]"""

            if conversation_history:
                for msg in conversation_history[-3:]:
                    if msg["role"] == "user":
                        prompt += f"<s>[INST] {msg['content']} [/INST]"
                    else:
                        prompt += f" {msg['content']}</s>"

            prompt += f"<s>[INST] {user_message} [/INST]"

            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 300,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "do_sample": True,
                },
            }

            # MODÈLES GRATUITS DISPONIBLES (essayez-les) :
            models = [
                "mistralai/Mistral-7B-Instruct-v0.2",  # Très bon modèle français
                "google/flan-t5-xl",  # Plus léger
                "HuggingFaceH4/zephyr-7b-alpha",  # Version alpha si beta échoue
                "microsoft/phi-2",  # Petit mais efficace
                "Qwen/Qwen2.5-7B-Instruct",  # Modèle récent
            ]

            for model in models:
                try:
                    print(f"    🤖 Essai modèle: {model}")
                    response = requests.post(
                        f"https://api-inference.huggingface.co/models/{model}",
                        headers=headers,
                        json=payload,
                        timeout=60,
                    )

                    print(f"    📡 HF Status pour {model}: {response.status_code}")

                    if response.status_code == 200:
                        result = response.json()
                        if isinstance(result, list) and len(result) > 0:
                            text = result[0].get("generated_text", "")
                            if "[/INST]" in text:
                                parts = text.split("[/INST]")
                                if len(parts) > 1:
                                    return parts[-1].strip()
                            return text
                        return "❌ Format inattendu"
                    elif response.status_code == 503:
                        print(f"    ⏳ Modèle {model} en cours de chargement...")
                        continue

                except Exception as e:
                    print(f"    ⚠️ Erreur avec {model}: {e}")
                    continue

            return "❌ Tous les modèles HF ont échoué"

        except Exception as e:
            error_msg = f"❌ Exception Hugging Face: {str(e)}"
            print(f"    {error_msg}")
            return error_msg

    def _fallback_chat_response(self, user_message: str) -> str:
        """Réponse de fallback à partir de la base de connaissances"""

        # D'abord, chercher dans la base de connaissances
        knowledge_response = self._search_in_knowledge_base(user_message)
        if knowledge_response:
            return knowledge_response

        # Si pas trouvé, utiliser les catégories existantes

        user_lower = user_message.lower()

        # ===== RECHERCHE DANS LA BASE DE CONNAISSANCES =====

        # Question sur le Cantal
        if any(word in user_lower for word in ["cantal", "lait", "brebis", "vache"]):
            return self._get_cheese_specific_info(user_lower)

        # Questions sur les problèmes
        elif any(
            word in user_lower
            for word in ["problème", "erreur", "marche pas", "raté", "échoué"]
        ):
            return self._get_problem_advice(user_lower)

        # Questions sur les recettes
        elif any(
            word in user_lower
            for word in ["recette", "fabriquer", "faire", "comment faire"]
        ):
            return self._get_recipe_advice(user_lower)

        # Questions sur les accords
        elif any(
            word in user_lower for word in ["vin", "accord", "boire", "dégustation"]
        ):
            return self._get_pairing_advice(user_lower)

        # Questions sur le matériel
        elif any(
            word in user_lower
            for word in ["matériel", "outil", "équipement", "acheter"]
        ):
            return self._get_equipment_advice()

        # Questions sur l'affinage
        elif any(
            word in user_lower for word in ["affinage", "mûrir", "cave", "température"]
        ):
            return self._get_aging_advice()

        else:
            return self._get_general_advice()

    def _get_cheese_specific_info(self, question: str) -> str:
        """Réponse spécifique sur un fromage"""
        response = "🧀 **Maître Fromager Pierre:**\n\n"

        # Détecter le fromage demandé
        if "cantal" in question.lower():
            response += "**À propos du Cantal :**\n\n"
            response += (
                "✅ **Faux !** Le Cantal n'est PAS fait avec du lait de brebis.\n\n"
            )
            response += "📖 **Véritable composition :**\n"
            response += "• **Lait :** Lait de vache entier\n"
            response += "• **Type :** Pâte pressée non cuite\n"
            response += "• **Région :** Auvergne (France)\n"
            response += "• **Affinage :** 1 à 6 mois minimum\n"
            response += "• **Appellation :** AOP (Appellation d'Origine Protégée)\n\n"
            response += "🐄 **Le lait de vache** utilisé pour le Cantal vient exclusivement de vaches de race Salers ou Montbéliarde, nourries avec l'herbe des montagnes d'Auvergne.\n\n"
            response += "❌ **Pourquoi pas de brebis ?**\n"
            response += "Les fromages de brebis des Pyrénées (comme l'Ossau-Iraty) sont différents. Le Cantal est un fromage de tradition bovine.\n\n"
            response += "🍷 **Accord recommandé :** Vin rouge de caractère comme un Cahors ou un Madiran."

        elif any(word in question.lower() for word in ["roquefort", "bleu", "brebis"]):
            response += "**À propos du Roquefort :**\n\n"
            response += "✅ **Oui !** Le Roquefort est fait avec du lait de brebis.\n\n"
            response += "📖 **Caractéristiques :**\n"
            response += "• **Lait :** Lait de brebis cru\n"
            response += "• **Type :** Pâte persillée (bleu)\n"
            response += "• **Région :** Aveyron (France)\n"
            response += "• **Moisissure :** Penicillium roqueforti\n"
            response += "• **Affinage :** En caves naturelles\n\n"
            response += "🐑 **Le lait de brebis** donne au Roquefort sa texture crémeuse et son goût prononcé caractéristique."

        elif any(word in question.lower() for word in ["chèvre", "chevret", "crottin"]):
            response += "**À propos des fromages de chèvre :**\n\n"
            response += "🧀 **Exemples de fromages de chèvre :**\n"
            response += "• Crottin de Chavignol\n"
            response += "• Sainte-Maure de Touraine\n"
            response += "• Chabichou du Poitou\n"
            response += "• Pouligny-Saint-Pierre\n\n"
            response += "🐐 **Tous ces fromages sont faits avec du lait de chèvre**, ce qui leur donne une saveur caractéristique légèrement acidulée."

        else:
            # Recherche dans la base de connaissances pour d'autres fromages
            response += "**Voici ce que je sais sur les laits utilisés :**\n\n"
            response += (
                "🐄 **Lait de vache :** Cantal, Camembert, Brie, Comté, Beaufort\n"
            )
            response += (
                "🐑 **Lait de brebis :** Roquefort, Ossau-Iraty, Pecorino, Manchego\n"
            )
            response += "🐐 **Lait de chèvre :** Crottin, Sainte-Maure, Chabichou\n"
            response += "🐃 **Lait de bufflonne :** Mozzarella di Bufala\n\n"
            response += "💡 **Pour une réponse précise, nommez le fromage !**"

        return response

    def _get_problem_advice(self, question: str) -> str:
        """Conseils pour les problèmes courants"""
        problems = self.knowledge_base.get("problemes_courants", {})

        response = "🧀 **Maître Fromager Pierre:**\n\n"
        response += "Voici mes conseils pour résoudre vos problèmes :\n\n"

        # Identifier le problème spécifique
        if "acide" in question:
            response += "**Problème: Fromage trop acide**\n"
            response += "✓ Solution: " + problems.get(
                "Fromage trop acide", "Réduire le temps de fermentation"
            )
        elif "dur" in question or "durci" in question:
            response += "**Problème: Caillé trop dur**\n"
            response += "✓ Solution: " + problems.get(
                "Caillé trop dur", "Réduire la dose de présure"
            )
        elif "mou" in question or "liquide" in question:
            response += "**Problème: Caillé trop mou**\n"
            response += "✓ Solution: " + problems.get(
                "Caillé trop mou", "Augmenter le temps de caillage"
            )
        elif "salé" in question:
            response += "**Problème: Fromage trop salé**\n"
            response += "✓ Solution: " + problems.get(
                "Fromage trop salé", "Réduire le temps de salage"
            )
        else:
            # Conseils généraux
            response += "**Conseils généraux de dépannage:**\n"
            response += "1. Vérifiez la température (32°C idéal)\n"
            response += "2. Utilisez du lait pasteurisé, jamais UHT\n"
            response += "3. Stérilisez tout le matériel\n"
            response += "4. Respectez les temps indiqués\n"
            response += "5. Notez chaque étape pour ajuster\n"

        response += "\n\n💡 **Pour une aide plus précise, décrivez exactement ce qui se passe !**"
        return response

    def _get_recipe_advice(self, question: str) -> str:
        """Conseils pour les recettes"""
        response = "🧀 **Maître Fromager Pierre:**\n\n"
        response += "**Ma recette de base pour débutant:**\n\n"
        response += "📝 **Fromage frais maison** (facile, 24h)\n"
        response += "• 2L lait entier pasteurisé\n"
        response += "• 2ml présure liquide (ou jus de 2 citrons)\n"
        response += "• 10g sel fin\n"
        response += "• Option: 1 yaourt nature (ferments)\n\n"
        response += "👨‍🍳 **Étapes:**\n"
        response += "1. Chauffer lait à 32°C\n"
        response += "2. Ajouter présure, mélanger 30s\n"
        response += "3. Couvrir, attendre 45min (caillage)\n"
        response += "4. Découper le caillé en cubes\n"
        response += "5. Égoutter 4h dans une étamine\n"
        response += "6. Saler, consommer dans les 3 jours\n\n"
        response += "✨ **Conseil:** Commencez simple, puis variez les fromages !"
        return response

    def _get_pairing_advice(self, question: str) -> str:
        """Conseils d'accords"""
        accords = self.knowledge_base.get("accords_vins", {})

        response = "🍷 **Maître Fromager Pierre:**\n\n"
        response += "**Mes accords préférés:**\n\n"

        if "chèvre" in question:
            response += "🧀 **Fromage de chèvre:**\n"
            response += "• Sancerre blanc (classique)\n"
            response += "• Pouilly-Fumé (minéral)\n"
            response += "• Rosé de Provence (été)\n"
        elif "brebis" in question:
            response += "🧀 **Fromage de brebis:**\n"
            response += "• Irouléguy rouge (Pays Basque)\n"
            response += "• Madiran (puissant)\n"
            response += "• Jurançon moelleux (avec bleu)\n"
        elif any(word in question for word in ["brie", "camembert", "molle"]):
            response += "🧀 **Pâte molle (brie/camembert):**\n"
            response += "• Champagne brut (fête)\n"
            response += "• Beaujolais nouveau (léger)\n"
            response += "• Cidre brut (normand)\n"
        else:
            response += "**Règle d'or:**\n"
            response += "• Fromage local + vin local\n"
            response += "• Jeune fromage → vin léger\n"
            response += "• Fromage affiné → vin puissant\n"
            response += "• Bleu → vin doux (Sauternes)\n"

        return response

    def _get_equipment_advice(self) -> str:
        """Conseils sur le matériel"""
        response = "🔧 **Maître Fromager Pierre:**\n\n"
        response += "**Matériel essentiel pour débuter:**\n\n"
        response += "1. Thermomètre de cuisine (précis à ±1°C) - 15€\n"
        response += "2. Grande casserole inox 5L - 25€\n"
        response += "3. Moule à fromage perforé 500g - 8€\n"
        response += "4. Étamine (toile à fromage) - 5€\n"
        response += "5. Présure liquide - 10€ (dure longtemps)\n\n"
        response += "💰 **Budget total:** ~60€\n\n"
        response += "💡 **Où acheter?** Tom Press, Fromag'Home, Amazon"
        return response

    def _get_aging_advice(self) -> str:
        """Conseils d'affinage"""
        response = "⏳ **Maître Fromager Pierre:**\n\n"
        response += "**Secrets d'un bon affinage:**\n\n"
        response += "🌡️ **Températures idéales:**\n"
        response += "• Pâte molle: 10-12°C\n"
        response += "• Pâte pressée: 12-14°C\n"
        response += "• Fromage frais: 4-6°C (frigo)\n\n"
        response += "💧 **Humidité:** 85-90% (un bol d'eau dans la cave)\n\n"
        response += "🔄 **Retournement:**\n"
        response += "• Jours 1-7: Tous les jours\n"
        response += "• Jours 8-30: 2x/semaine\n"
        response += "• Après 1 mois: 1x/semaine\n\n"
        response += "🧼 **Nettoyage:** Brossez délicatement si moisissures indésirables"
        return response

    def _get_general_advice(self) -> str:
        """Conseils généraux"""
        import random

        conseils = [
            "🧀 **Commencez simple** avec un fromage frais avant de tenter les pâtes persillées !",
            "🌡️ **La température est cruciale** - ±2°C peut tout changer. Soyez précis !",
            "📝 **Tenez un carnet** - notez chaque étape pour progresser à chaque essai.",
            "🧼 **Hygiène absolue** - stérilisez TOUT le matériel à l'eau bouillante.",
            "⏳ **La patience paie** - un bon fromage ne se précipite pas.",
            "🥛 **Qualité du lait** - préférez lait cru ou pasteurisé, JAMAIS UHT.",
            "🔄 **Goûtez régulièrement** - l'affinage évolue, trouvez votre stade préféré.",
        ]

        response = "🧀 **Maître Fromager Pierre:**\n\n"
        response += random.choice(conseils)
        response += (
            "\n\n💭 **Posez-moi une question précise pour un conseil personnalisé !**"
        )
        return response

    def _get_general_advice(self) -> str:
        """Conseils généraux"""
        import random

        conseils = [
            "🧀 **Commencez simple** avec un fromage frais avant de tenter les pâtes persillées !",
            "🌡️ **La température est cruciale** - ±2°C peut tout changer. Soyez précis !",
            # ... (le reste de la fonction existante)
        ]

        response = "🧀 **Maître Fromager Pierre:**\n\n"
        response += random.choice(conseils)
        response += (
            "\n\n💭 **Posez-moi une question précise pour un conseil personnalisé !**"
        )
        return response

    # ===== AJOUTER ICI =====
    def _search_in_knowledge_base(self, query: str) -> str:
        """Recherche intelligente dans la base de connaissances"""
        query_lower = query.lower()

        # 1. Recherche sur les fromages spécifiques
        cheese_facts = {
            "cantal": {
                "lait": "vache",
                "type": "Pâte pressée non cuite",
                "region": "Auvergne",
                "info": "Fromage AOP au lait de vache Salers",
            },
            "roquefort": {
                "lait": "brebis",
                "type": "Pâte persillée",
                "region": "Aveyron",
                "info": "Bleu au lait de brebis cru",
            },
            "camembert": {
                "lait": "vache",
                "type": "Pâte molle",
                "region": "Normandie",
                "info": "Fromage à croûte fleurie",
            },
            "chèvre": {
                "lait": "chèvre",
                "type": "Fromage frais ou pressé",
                "region": "France",
                "info": "Fromage au lait de chèvre, souvent frais",
            },
        }

        # Vérifier les fromages connus
        for cheese_name, info in cheese_facts.items():
            if cheese_name in query_lower:
                response = f"🧀 **{cheese_name.upper()}**\n\n"
                response += f"🐄 **Lait :** {info['lait']}\n"
                response += f"🧈 **Type :** {info['type']}\n"
                response += f"📍 **Région :** {info['region']}\n"
                response += f"📝 **Info :** {info['info']}\n"

                # Ajouter des infos supplémentaires depuis la base
                if "accords_vins" in self.knowledge_base:
                    for cheese_key, wine in self.knowledge_base["accords_vins"].items():
                        if cheese_name in cheese_key.lower():
                            response += f"\n🍷 **Accord vin :** {wine}"
                            break

                return response

        # 2. Recherche générique sur les laits
        if any(word in query_lower for word in ["lait de", "fait avec"]):
            lait_types = {
                "brebis": ["roquefort", "ossau-iraty", "manchego", "pecorino"],
                "chèvre": ["crottin", "sainte-maure", "chabichou", "valençay"],
                "vache": ["cantal", "camembert", "brie", "comté", "beaufort"],
                "bufflonne": ["mozzarella di bufala", "burrata"],
            }

            for lait_type, fromages in lait_types.items():
                if lait_type in query_lower:
                    response = f"🐄 **Fromages au lait de {lait_type} :**\n\n"
                    for f in fromages[:5]:  # Limiter à 5 exemples
                        response += f"• {f.title()}\n"
                    return response

        # 3. Recherche dans la structure de base de connaissances
        # Types de pâte
        if "types_pate" in self.knowledge_base:
            for cheese_type, info in self.knowledge_base["types_pate"].items():
                if cheese_type.lower() in query_lower:
                    response = f"🧀 **{cheese_type.upper()}**\n\n"
                    response += f"📝 {info['description']}\n"
                    response += f"🏷️ Exemples: {info['exemples']}\n"
                    response += f"⏱️ Durée: {info['duree']}\n"
                    response += f"📊 Difficulté: {info['difficulte']}\n"
                    return response

        # Accords vins
        if "vin" in query_lower or "accord" in query_lower:
            if "accords_vins" in self.knowledge_base:
                for cheese, wine in self.knowledge_base["accords_vins"].items():
                    if any(word in query_lower for word in cheese.lower().split()):
                        return f"🍷 **Accord pour {cheese}:**\n{wine}"

        return None

    def _get_compatibility_info(self, query: str) -> str:
        """Donne des infos sur les compatibilités"""
        response = "🧀 **Règles de compatibilité lait/pâte:**\n\n"

        if "regles_compatibilite" not in self.knowledge_base:
            return "⚠️ Informations de compatibilité non disponibles."

        # Ajouter votre logique ici selon la question
        # ...

        return response

    def _chat_openrouter(self, user_message: str, conversation_history=None):
        """Utilise OpenRouter API avec des modèles GRATUITS qui fonctionnent"""
        try:
            print(f"    🔑 OpenRouter Key détectée")

            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/volubyl/fromager",
            }

            # Construire les messages
            messages = [
                {
                    "role": "system",
                    "content": """Tu es "Maître Fromager Pierre", expert français avec 40 ans d'expérience.
Tu es chaleureux, pédagogique et passionné. Réponds EN FRANÇAIS avec précision et enthousiasme.
Sois concis mais complet. Utilise des emojis fromagers occasionnellement 🧀.""",
                }
            ]

            # Ajouter l'historique si disponible
            if conversation_history:
                for msg in conversation_history[-3:]:  # Garder 3 derniers messages
                    messages.append({"role": msg["role"], "content": msg["content"]})

            # Ajouter le nouveau message
            messages.append({"role": "user", "content": user_message})

            # MODÈLES GRATUITS QUI FONCTIONNENT VRAIMENT SUR OPENROUTER
            free_models = [
                "meta-llama/llama-3.2-3b-instruct",  # ✅ GARANTI GRATUIT - Llama 3.2
                "microsoft/phi-3-mini-4k-instruct",  # ✅ GARANTI GRATUIT - Microsoft
                "qwen/qwen2.5-3b-instruct",  # ✅ GARANTI GRATUIT - Alibaba (bon français)
                "google/gemma-2-2b-it",  # ✅ GARANTI GRATUIT - Google
                "mistralai/mistral-7b-instruct-v0.2",  # ⚠️ Parfois gratuit
                "huggingfaceh4/zephyr-7b-beta",  # ⚠️ Parfois gratuit
            ]

            # Essayer chaque modèle jusqu'à ce qu'un fonctionne
            for model in free_models:
                try:
                    print(f"    🤖 Essai modèle: {model}")

                    payload = {
                        "model": model,
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 600,
                        "stream": False,
                    }

                    response = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=15,
                    )

                    print(
                        f"    📡 Status pour {model.split('/')[-1]}: {response.status_code}"
                    )

                    if response.status_code == 200:
                        result = response.json()
                        if "choices" in result and len(result["choices"]) > 0:
                            response_text = result["choices"][0]["message"]["content"]
                            print(
                                f"    ✅ Réponse obtenue avec {model.split('/')[-1]} ({len(response_text)} caractères)"
                            )
                            return response_text

                    elif response.status_code == 402:
                        print(
                            f"    💸 Modèle {model.split('/')[-1]} nécessite des crédits"
                        )
                        continue  # Essayer le modèle suivant

                    elif response.status_code == 404:
                        print(f"    🔍 Modèle {model.split('/')[-1]} non disponible")
                        continue  # Essayer le modèle suivant

                    else:
                        print(
                            f"    ❌ Erreur {response.status_code} pour {model.split('/')[-1]}"
                        )
                        continue

                except requests.exceptions.Timeout:
                    print(f"    ⏱️ Timeout pour {model.split('/')[-1]}")
                    continue

                except Exception as e:
                    print(
                        f"    ⚠️ Exception avec {model.split('/')[-1]}: {type(e).__name__}"
                    )
                    continue

            print("    ❌ Aucun modèle OpenRouter n'a fonctionné")
            return None

        except Exception as e:
            print(f"    ❌ Exception OpenRouter globale: {type(e).__name__}")
            return None

    # Fin de la classe


# Initialiser l'agent
agent = AgentFromagerHF()


def update_profile_description(profile):
    """Affiche une description selon le profil"""

    descriptions = {
        "🧀 Amateur": """
        ### 🏠 Mode Amateur
        - Explications claires et accessibles
        - Astuces pour débutants
        - Matériel de base
        - Recettes faciles à suivre
        """,
        "🏭 Producteur": """
        ### 🏭 Mode Producteur
        - Protocoles professionnels
        - Normes sanitaires
        - Rendements et coûts
        - Traçabilité des ingrédients
        """,
        "🎓 Formateur": """
        ### 🎓 Mode Formateur
        - Objectifs pédagogiques
        - Points d'attention
        - Erreurs courantes
        - Variantes et expérimentations
        """,
    }

    return descriptions.get(profile, "")

    return demo

def generate_all(
    ingredients, cheese_type, constraints, creativity, texture, affinage, spice, profile
):
    """Génère recette + recherche web + ACTUALISE automatiquement l'historique"""
    try:
        print("🚀 Début de generate_all")

        # 1. GÉNÉRER LA RECETTE (sauvegarde automatique dans generate_recipe_creative)
        recipe = agent.generate_recipe_creative(
            ingredients,
            cheese_type,
            constraints,
            creativity,
            texture,
            affinage,
            spice,
            profile,
        )

        print("✅ Recette générée")

        # 2. RECHERCHE WEB
        try:
            web_recipes = agent.search_web_recipes(
                ingredients, cheese_type, max_results=6
            )
            print(
                f"✅ Recherche web: {len(web_recipes) if web_recipes else 0} résultats"
            )
        except Exception as e:
            print(f"⚠️ Erreur recherche web: {e}")
            web_recipes = []

        # 3. CONSTRUIRE HTML DES RÉSULTATS WEB
        if not web_recipes:
            cards_html = """
            <div class="no-recipes">
                😔 Aucune recette trouvée sur le web<br>
                <small>💡 Essayez des ingrédients plus courants</small>
            </div>
            """
        else:
            cards_html = f"""
            <div class="search-status">
                ✅ {len(web_recipes)} recettes trouvées sur le web
            </div>
            """
            for i, r in enumerate(web_recipes, 1):
                cards_html += f"""
                <div class="recipe-card">
                    <div class="recipe-title">{i}. {r.get('title', 'Recette')}</div>
                    <div class="recipe-source">📍 {r.get('source', 'Web')}</div>
                    <div class="recipe-description">{r.get('description', '')[:200]}...</div>
                    <a href="{r.get('url', '#')}" target="_blank" class="recipe-link">🔗 Voir la recette</a>
                </div>
                """

        # ===== 4. ACTUALISATION AUTOMATIQUE DE L'HISTORIQUE =====
        print("🔄 Actualisation automatique de l'historique...")

        # A. Forcer le rechargement de l'historique
        agent.history = agent._load_history()

        # B. Créer un résumé mis à jour
        from datetime import datetime

        summary = "╔══════════════════════════════════════════════════════════╗\n"
        summary += f"║   📚 HISTORIQUE MIS À JOUR ({len(agent.history)} recettes)   \n"
        summary += "╚══════════════════════════════════════════════════════════╝\n\n"

        if agent.history:
            # Afficher les 3 dernières recettes
            for i, entry in enumerate(agent.history[-3:][::-1], 1):
                cheese_name = entry.get("cheese_name", "Sans nom")
                date_str = entry.get("timestamp", "")
                if not date_str and "date" in entry:
                    try:
                        dt = datetime.fromisoformat(
                            entry["date"].replace("Z", "+00:00")
                        )
                        date_str = dt.strftime("%d/%m/%Y %H:%M")
                    except:
                        date_str = entry["date"].split("T")[0]

                summary += f"🧀 {i}. {cheese_name}\n"
                summary += (
                    f"    📅 {date_str} | 🏷️ {entry.get('type', 'Type inconnu')}\n\n"
                )

        # C. Préparer les choix du dropdown
        choices = []
        if agent.history:
            for i, entry in enumerate(agent.history[-20:][::-1], 1):
                cheese_name = entry.get("cheese_name", "Sans nom")
                date_str = entry.get("timestamp", "")
                if not date_str and "date" in entry:
                    try:
                        dt = datetime.fromisoformat(
                            entry["date"].replace("Z", "+00:00")
                        )
                        date_str = dt.strftime("%d/%m/%Y")
                    except:
                        date_str = entry["date"].split("T")[0]

                choice_text = f"{i}. {cheese_name}"
                if date_str:
                    choice_text += f" ({date_str})"
                choices.append(choice_text)

        # D. Ajouter un message spécial pour la nouvelle recette
        if agent.history:
            last = agent.history[-1]
            summary += f"✨ **NOUVELLE RECETTE AJOUTÉE :** {last.get('cheese_name', 'Nouveau fromage')}\n"
            summary += f"   📍 Disponible dans la liste déroulante\n\n"

        # E. Si pas de recettes
        if not agent.history:
            summary += "📭 Aucune recette sauvegardée.\n"
            summary += "💡 Votre recette vient d'être créée et apparaîtra ici !\n\n"

        print(f"✅ Historique actualisé: {len(agent.history)} recettes")

        # ===== 5. RETOURNER TOUT (6 ÉLÉMENTS) =====
        # MAINTENANT : Il faut que votre callback Gradio ATTENDE 6 éléments !
        return (
            recipe,  # 1. La recette générée (Textbox)
            "",  # 2. Statut de recherche (Textbox)
            cards_html,  # 3. Cartes web (HTML)
            summary,  # 4. Historique mis à jour (Textbox)
            gr.Dropdown(choices=choices, value=None),  # 5. Liste pour dropdown (LIST)
            "",  # 6. Effacer l'affichage précédent (Textbox)
        )

    except Exception as e:
        print(f"❌ Erreur generate_all: {e}")
        import traceback

        traceback.print_exc()

        # Retourner 6 éléments d'erreur
        return (
            f"❌ Erreur: {str(e)}",  # 1. Message d'erreur (Textbox)
            "❌ Erreur",  # 2. Statut (Textbox)
            "<div class='no-recipes'>❌ Erreur technique</div>",  # 3. HTML
            "❌ Erreur lors de la génération",  # 4. Historique (Textbox)
            [],  # 5. Liste vide pour dropdown (LIST)
            "",  # 6. Vide (Textbox)
        )

# CREATE INTERFACE GRADIO
# ===== VERSION CORRIGÉE DE create_interface AVEC AUTHENTIFICATION =====

print("="*60)
print("🔍 DEBUG AUTHENTIFICATION")
print(f"AUTH_USERNAME chargé : {AUTH_USERNAME}")
print(f"AUTH_PASSWORD chargé : {AUTH_PASSWORD}")
print(f"Longueur password : {len(AUTH_PASSWORD) if AUTH_PASSWORD else 0}")
print("="*60)

def create_interface():
    """Interface avec authentification et génération simultanée"""

    import gradio as gr
    import json
    import os

    # Définir custom_css
    custom_css = """
    .no-recipes {
        text-align: center;
        padding: 40px;
        color: #666;
        font-size: 1.2em;
    }
    .recipe-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background: #f9f9f9;
    }
    #recipe-scroll {
        overflow-y: auto;
        max-height: 800px;
    }
    #chat-display {
        overflow-y: auto;
        max-height: 500px;
    }
    .login-box {
        max-width: 400px;
        margin: 100px auto;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(0,0,0,0.1);
        background: white;
    }
    """
    
    with gr.Blocks(
        title="🧀 Agent Fromager - Authentification",
        theme=gr.themes.Soft(primary_hue="orange", secondary_hue="amber"),
        css=custom_css,
        head="""
        <link rel="icon" type="image/png" href="https://em-content.zobj.net/source/apple/391/cheese-wedge_1f9c0.png">
        """,
    ) as demo:
        
        # État d'authentification
        is_authenticated = gr.State(value=False)
    
        # ===== ÉCRAN DE LOGIN =====
        with gr.Column(visible=True, elem_classes="login-box") as login_screen:
            gr.Markdown(f"""
            # 🔐 Agent Fromager
            ### Accès sécurisé
            
            **Identifiants attendus :**
            - Utilisateur : `{AUTH_USERNAME}`
            """)
            
            username_input = gr.Textbox(label="Nom d'utilisateur", placeholder="admin")
            password_input = gr.Textbox(label="Mot de passe", type="password", placeholder="••••••••")
            login_button = gr.Button("🔓 Se connecter", variant="primary", size="lg")
            login_status = gr.Markdown("")
        
    
        # ===== ÉCRAN PRINCIPAL =====
        with gr.Column(visible=False) as main_screen:
            
            gr.HTML("""
            <h1 style="text-align: center; color: #BF360C;">🧀 Agent Fromager Générateur de recettes</h1>
            <h3 style="text-align: center; color: #5D4037;">Créez vos fromages avec l'IA + Recherche web automatique</h3>
            """)

            # Sélecteur de profil
            gr.Markdown("## 👤 Personnalisez votre expérience")

            with gr.Row():
                profile_selector = gr.Radio(
                    choices=["🧀 Amateur", "🏭 Producteur", "🎓 Formateur"],
                    value="🧀 Amateur",
                    label="Quel est votre profil ?",
                    info="Les recettes seront adaptées à votre niveau et vos besoins",
                    interactive=True,
                    scale=2,
                )

                # Description des profils
                gr.Markdown("""
                **🧀 Amateur** : Recettes accessibles avec conseils pratiques  
                **🏭 Producteur** : Fiches techniques précises et professionnelles  
                **🎓 Formateur** : Supports pédagogiques avec objectifs d'apprentissage
                """)

            # ===== ZONE DE SAISIE =====
            with gr.Row():
                with gr.Column(scale=2):
                    ingredients_input = gr.Textbox(
                        label="🥛 Ingrédients disponibles",
                        placeholder="Ex: lait de chèvre, présure, sel, herbes",
                        lines=3,
                    )

                    cheese_type_input = gr.Dropdown(
                        choices=[
                            "Laissez l'IA choisir",
                            "Fromage frais",
                            "Pâte molle",
                            "Pâte pressée non cuite",
                            "Pâte pressée cuite",
                            "Pâte persillée",
                        ],
                        label="🧀 Type de fromage",
                        value="Laissez l'IA choisir",
                    )

                    constraints_input = gr.Textbox(
                        label="⚙️ Contraintes",
                        placeholder="Ex: végétarien, rapide...",
                        lines=2,
                    )

                    gr.Markdown("### 🎛️ Micro-choix")

                    with gr.Row():
                        creativity_slider = gr.Slider(
                            0, 3, value=0, step=1, label="🎨 Créativité"
                        )
                        texture_choice = gr.Radio(
                            ["Très crémeux", "Équilibré", "Très ferme"],
                            value="Équilibré",
                            label="🧈 Texture",
                        )

                    with gr.Row():
                        affinage_slider = gr.Slider(
                            0, 12, value=4, step=1, label="⏱️ Affinage (semaines)"
                        )
                        spice_choice = gr.Radio(
                            ["Neutre", "Modéré", "Intense"],
                            value="Neutre",
                            label="🌶️ Épices",
                        )

                    generate_all_btn = gr.Button(
                        "✨ Générer la recette + Recherche web",
                        variant="primary",
                        size="lg",
                    )

                    gr.Markdown(
                        "⏳ *La génération + recherche web prend 10-15 secondes...*"
                    )

                with gr.Column(scale=1):
                    gr.Markdown("""
                    ### 💡 Comment ça marche ?
                    
                    1️⃣ Entrez vos ingrédients
                    2️⃣ Ajustez les micro-choix
                    
                    3️⃣ Cliquez sur "Générer"
                    
                    **Résultat :**
                    - Onglet 1 : 📖 Votre recette
                    - Onglet 2 : 🌐 Recettes web
                    - Onglet 3 : 📚 Base de connaissances
                    - Onglet 4 : 🕒 Historique
                    - Onglet 5 : 💬 Expert Fromager
                    """)

            # ===== FONCTIONS LOCALES =====
            def load_history():
                """Charge l'historique"""
                try:
                    if hasattr(agent, "history") and agent.history:
                        history = agent.history
                    elif os.path.exists(agent.recipes_file):
                        with open(agent.recipes_file, "r", encoding="utf-8") as f:
                            history = json.load(f)
                    else:
                        return "📭 Aucune recette sauvegardée", []

                    if not history:
                        return "📭 Aucune recette sauvegardée", []

                    choices = []
                    for i, entry in enumerate(history[-20:][::-1], 1):
                        cheese_name = entry.get("cheese_name", "Sans nom")
                        date = entry.get("date", "").split("T")[0] if entry.get("date") else ""
                        
                        if date:
                            try:
                                year, month, day = date.split("-")
                                date_formatted = f"{day}/{month}/{year}"
                                choice_text = f"{i}. {cheese_name} ({date_formatted})"
                            except:
                                choice_text = f"{i}. {cheese_name}"
                        else:
                            choice_text = f"{i}. {cheese_name}"
                        
                        choices.append(choice_text)

                    summary = "╔══════════════════════════════════════════════════════════╗\n"
                    summary += f"║   📚 HISTORIQUE : {len(history)} RECETTE(S)   \n"
                    summary += "╚══════════════════════════════════════════════════════════╝\n\n"

                    for i, entry in enumerate(history[-10:][::-1], 1):
                        cheese_name = entry.get("cheese_name", "Sans nom")
                        date = entry.get("date", "").split("T")[0] if entry.get("date") else "????-??-??"
                        cheese_type = entry.get("type", "Type inconnu")
                        
                        summary += f"🧀 {i}. {cheese_name}\n"
                        summary += f"    ├─ 📅 {date}\n"
                        summary += f"    └─ 🧈 {cheese_type}\n\n"

                    return summary, choices

                except Exception as e:
                    return f"❌ Erreur: {str(e)}", []

            def show_recipe_select(choice):
                """Affiche la recette sélectionnée"""
                if not choice:
                    return ""
                
                try:
                    num_str = choice.split(".")[0].strip()
                    position = int(num_str)
                    
                    if hasattr(agent, "history") and agent.history:
                        history = agent.history
                    elif os.path.exists(agent.recipes_file):
                        with open(agent.recipes_file, "r", encoding="utf-8") as f:
                            history = json.load(f)
                    else:
                        return "❌ Historique introuvable"
                    
                    reversed_history = history[-20:][::-1]
                    
                    if position > 0 and position <= len(reversed_history):
                        entry = reversed_history[position - 1]
                        return entry.get("recipe_complete", "")
                    else:
                        return f"❌ Recette #{position} introuvable"
                        
                except Exception as e:
                    return f"❌ Erreur: {str(e)}"

            def agent_clear_history():
                """Efface l'historique"""
                try:
                    recipes_file = "recipes_history.json"
                    with open(recipes_file, "w", encoding="utf-8") as f:
                        json.dump([], f)
                    
                    if hasattr(agent, "history"):
                        agent.history = []
                    
                    return "✅ Historique effacé", [], ""
                except Exception as e:
                    return f"❌ Erreur: {str(e)}", [], ""

            def load_and_populate():
                """Charge et met à jour"""
                summary, choices = load_history()
                return summary, gr.Dropdown(choices=choices, value=None)

            def clear_and_reset():
                """Efface et reset"""
                return agent_clear_history()
            
            fallback_cache = None

            # ===== ONGLETS =====
            with gr.Tabs():
                # ONGLET 1 : Recette
                with gr.Tab("📖 Mon fromage"):
                    recipe_output = gr.Textbox(
                        label="Votre recette complète",
                        lines=25,
                        max_lines=90,
                        placeholder="Votre recette apparaîtra ici...",
                        elem_id="recipe-scroll",
                    )

                # ONGLET 2 : Web
                with gr.Tab("🌐 Recettes Web"):
                    search_status = gr.HTML(label="Statut", value="")
                    web_results = gr.HTML(
                        label="Résultats",
                        value="<div class='no-recipes'>Cliquez sur 'Générer'...</div>",
                    )

                # ONGLET 3 : Base de connaissances
                with gr.Tab("📚 Base de connaissances"):
                    with gr.Row():
                        knowledge_btn = gr.Button("📖 Charger résumé", variant="primary")
                    
                    knowledge_output = gr.Textbox(
                        label="🧀 SAVOIR FROMAGÈRE",
                        lines=45,
                        placeholder="Cliquez pour charger...",
                    )
                    
                    knowledge_btn.click(fn=agent.get_knowledge_summary, outputs=knowledge_output)

                # ONGLET 4 : Historique (VERSION DYNAMIQUE)
                with gr.Tab("🕒 Historique"):
                    gr.Markdown("### 📚 Historique de vos recettes")
                    
                    # ===== VARIABLES GLOBALES =====
                    recipe_map = {}
                    stats_visible = False
                    
                    # ===== COMPTEUR DYNAMIQUE =====
                    counter_card = gr.HTML("""
                    <div style="
                        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
                        color: white;
                        padding: 15px;
                        border-radius: 12px;
                        text-align: center;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                        margin-bottom: 10px;
                    ">
                        <div style="font-size: 12px; opacity: 0.9; letter-spacing: 1px;">RECETTES DISPONIBLES</div>
                        <div style="font-size: 36px; font-weight: bold; margin: 8px 0;">Chargement...</div>
                        <div style="font-size: 11px; opacity: 0.8; display: flex; justify-content: space-around;">
                            <span id="personal-count">? perso</span>
                            <span>•</span>
                            <span id="reference-count">? réf</span>
                        </div>
                    </div>
                    """)
                    
                    # ===== BOUTONS =====
                    with gr.Row():
                        history_btn = gr.Button("🔄 Actualiser", variant="primary")
                        count_btn = gr.Button("🔢 Statistiques", variant="secondary")
                        clear_btn = gr.Button("🗑️ Effacer", variant="stop")
                    
                    # ===== STATISTIQUES =====
                    stats_display = gr.HTML(
                        value="<div style='padding: 20px; text-align: center; color: #666;'/div>"
                    )
                    
                    # ===== HISTORIQUE PRINCIPAL =====
                    with gr.Row():
                        with gr.Column(scale=1):
                            history_summary = gr.Textbox(
                                label="📋 Vos recettes",
                                lines=10,
                                interactive=False,
                                value="Cliquez sur 'Actualiser' pour charger...",
                                show_label=True
                            )
                            
                            show_fallback_btn = gr.Button("📖 Voir recettes de référence")
                        
                        with gr.Column(scale=2):
                            # dropdown de test

                            recipe_dropdown = gr.Dropdown(
                                label="🍽️ Sélectionner une recette",
                                choices=["→ Sélectionner parmi les recettes"],  # ← Placeholder comme premier choix
                                interactive=True,
                                value="→ Sélectionner parmi les recettes",  # ← Sélectionné par défaut
                                allow_custom_value=False,
                                multiselect=False,
                                elem_id="recipe_dropdown_fixed"  # Nouvel ID
                            )
                            
                            recipe_display = gr.Textbox(
                                label="📖 Recette complète",
                                lines=20,
                                interactive=False,
                                value="",
                                show_label=True
                            )
                    
                    # ===== FONCTIONS DYNAMIQUES =====
                    # ===== VARIABLE SIMPLE POUR LE TOGGLE =====
                    stats_visible = False

                    def toggle_stats():
                        """Toggle propre entre 2 états seulement"""
                        global stats_visible
                        
                        # Inverser l'état
                        stats_visible = not stats_visible
                        
                        if stats_visible:
                            # ÉTAT 1: Stats VISIBLES
                            print("📊 Affichage des statistiques")
                            result = show_stats()
                            
                            # RETOURNER UN SEUL OBJET gr.update() pour le bouton
                            return [
                                result,  # stats_display
                                gr.update(value="👁️‍🗨️ Cacher", variant="stop")  # UN SEUL UPDATE
                            ]
                        
                        else:
                            # ÉTAT 2: Stats CACHÉES
                            print("👁️‍🗨️ Cache les statistiques")
                            
                            return [
                                "<div style='padding: 20px; text-align: center; color: #666;'>Cliquez sur 'Compter' pour voir les statistiques</div>",
                                gr.update(value="🔢 Compter", variant="secondary")  # UN SEUL UPDATE
                            ]
                    
                    def get_fallback_count():
                        """Retourne le nombre RÉEL de recettes de référence"""
                        try:
                            global fallback_cache
                            
                            if fallback_cache is None:
                                # Charger UNE FOIS avec un nombre grand
                                fallback_cache = agent._get_absolute_fallback("", "", 1000)
                            
                            real_count = len(fallback_cache)
                            print(f"📊 Nombre réel de recettes de référence: {real_count}")
                            return real_count
                            
                        except Exception as e:
                            print(f"❌ Erreur get_fallback_count: {e}")
                            return 0
                    
                    def update_interface():
                        """Actualise TOUTE l'interface - COMPTE RÉEL"""
                        global stats_visible
                        
                        # Réinitialiser l'état
                        stats_visible = False
                        
                        try:
                            print("🔄 Début update_interface")
                            
                            # 1. Récupérer données
                            history = agent.get_history()
                            fallback_count = get_fallback_count()  # ← Nombre RÉEL
                            
                            print(f"📊 Histoire réelle: {len(history)} entrées")
                            print(f"📊 Contenu histoire:")
                            for i, entry in enumerate(history):
                                print(f"  [{i}] ID: {entry.get('id')}, Nom: {entry.get('cheese_name', 'N/A')}")
                            
                            # 2. Compteur DYNAMIQUE
                            total = len(history) + fallback_count
                            counter_html = f"""
                            <div style="
                                background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
                                color: white;
                                padding: 15px;
                                border-radius: 12px;
                                text-align: center;
                                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                                margin-bottom: 10px;
                            ">
                                <div style="font-size: 12px; opacity: 0.9; letter-spacing: 1px;">RECETTES DISPONIBLES</div>
                                <div style="font-size: 36px; font-weight: bold; margin: 8px 0;">{total}</div>
                                <div style="font-size: 11px; opacity: 0.8; display: flex; justify-content: space-around;">
                                    <span id="personal-count">{len(history)} perso</span>
                                    <span>•</span>
                                    <span id="reference-count">{fallback_count} réf</span>
                                </div>
                            </div>
                            """
                            
                            # 3. Texte historique
                            if not history:
                                summary = "📭 **Votre historique est vide**\n\n"
                                summary += "💡 Créez votre première recette !\n\n"
                                summary += f"📚 **{fallback_count} recettes de référence** disponibles"
                            else:
                                summary = f"📚 **{len(history)} recettes personnelles**\n" + "="*40 + "\n\n"
                                for i, entry in enumerate(reversed(history[-5:]), 1):
                                    name = entry.get('cheese_name', f"Recette #{entry.get('id')}")
                                    date = entry.get('date', '')[:10]
                                    summary += f"{i}. {name}\n"
                                    summary += f"   📅 {date}\n"
                                    summary += "-"*30 + "\n"
                            
                            # 4. Dropdown
                            choices = []
                            global recipe_map
                            recipe_map = {}
                            
                            print(f"🎯 Création dropdown à partir de {len(history)} entrées")
                            
                            for entry in reversed(history):
                                entry_id = entry.get('id')
                                entry_name = entry.get('cheese_name', f"Recette #{entry_id}")
                                date = entry.get('date', '')[:10] if entry.get('date') else 'sans date'
                                
                                display_text = f"{entry_id}. {entry_name} ({date})"
                                
                                # Vérifier les doublons (au cas où)
                                if display_text not in recipe_map:
                                    choices.append(display_text)
                                    recipe_map[display_text] = entry_id
                                    print(f"   ➕ Ajouté: {display_text}")
                                else:
                                    print(f"   ⚠️ Doublon ignoré: {display_text}")
                            
                            print(f"✅ Dropdown créé avec {len(choices)} choix uniques")
                            
                            choices_with_placeholder = ["Sélectionner parmi les recettes 👉"] + choices
                            
                            print(f"✅ Interface: {len(history)} perso + {fallback_count} réf = {total} total")
                            
                            return [
                                counter_html,
                                summary,
                                # ✅ CHANGEMENT : Utiliser gr.update() pour mettre à jour le Dropdown
                                gr.update(
                                    choices=choices_with_placeholder,              # Les choix avec placeholder
                                    value="Sélectionner parmi les recettes 👉"     
                                ),
                                "Sélectionnez une recette...",
                            ]
                            
                        except Exception as e:
                            print(f"❌ Erreur update_interface: {e}")
                            import traceback
                            traceback.print_exc()
                            return [
                                f"<div style='color: red;'>Erreur: {str(e)[:50]}</div>",
                                f"Erreur: {str(e)}",
                                [],
                                f"Erreur: {str(e)}",
                                "<div style='padding: 20px; text-align: center; color: #666;'>Cliquez sur 'Compter' pour voir les statistiques</div>",
                                "🔢 Compter",
                                "secondary"
                            ]
                    
                    def show_stats():
                        """Affiche les statistiques RÉELLES"""
                        try:
                            print("📊 Début show_stats")
                            
                            history = agent.get_history()
                            global fallback_cache
                            
                            if fallback_cache is None:
                                fallback_cache = agent._get_absolute_fallback("", "", 1000)
                            
                            fallback_count = len(fallback_cache)
                            
                            # Compter par type de lait
                            lait_stats = {}
                            for recipe in fallback_cache:
                                lait = recipe.get('lait', 'mixte')
                                lait_stats[lait] = lait_stats.get(lait, 0) + 1
                            
                            # Construire HTML avec chiffres RÉELS
                            stats_html = f"""
                            <div style="padding: 20px; background: #f8f9fa; border-radius: 10px;">
                                <h3 style="margin-top: 0;">📊 Statistiques RÉELLES</h3>
                                
                                <div style="display: flex; gap: 20px; margin-bottom: 20px;">
                                    <div style="flex: 1; background: white; padding: 15px; border-radius: 8px; text-align: center;">
                                        <div style="font-size: 32px; color: #4CAF50; font-weight: bold;">{len(history)}</div>
                                        <div style="font-size: 12px; color: #666;">Vos créations</div>
                                    </div>
                                    <div style="flex: 1; background: white; padding: 15px; border-radius: 8px; text-align: center;">
                                        <div style="font-size: 32px; color: #2196F3; font-weight: bold;">{fallback_count}</div>
                                        <div style="font-size: 12px; color: #666;">Références</div>
                                    </div>
                                    <div style="flex: 1; background: white; padding: 15px; border-radius: 8px; text-align: center;">
                                        <div style="font-size: 32px; color: #FF9800; font-weight: bold;">{len(history) + fallback_count}</div>
                                        <div style="font-size: 12px; color: #666;">Total</div>
                                    </div>
                                </div>
                                
                                <h4 style="margin-bottom: 10px;">🥛 Répartition par type de lait</h4>
                                <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 20px;">
                            """
                            
                            for lait, count in lait_stats.items():
                                lait_name = lait if lait else 'mixte'
                                emoji = {'vache': '🐄', 'chèvre': '🐐', 'brebis': '🐑', 'bufflonne': '🐃'}.get(lait, '🥛')
                                stats_html += f"""
                                <div style="padding: 10px; background: white; border-radius: 6px; text-align: center; min-width: 100px;">
                                    <div style="font-size: 24px;">{emoji}</div>
                                    <div style="font-size: 20px; font-weight: bold;">{count}</div>
                                    <div style="font-size: 12px; color: #666;">{lait_name}</div>
                                </div>
                                """
                            
                            # Sources principales
                            source_stats = {}
                            for recipe in fallback_cache:
                                source = recipe.get('source', 'inconnue')
                                source_stats[source] = source_stats.get(source, 0) + 1
                            
                            stats_html += """
                                </div>
                                
                                <h4 style="margin-bottom: 10px;">🌐 Sources principales</h4>
                                <div style="max-height: 150px; overflow-y: auto; background: white; padding: 10px; border-radius: 6px;">
                            """
                            
                            for source, count in sorted(source_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
                                stats_html += f"""
                                <div style="display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #f0f0f0;">
                                    <span>{source}</span>
                                    <span style="font-weight: bold;">{count}</span>
                                </div>
                                """
                            
                            stats_html += f"""
                                </div>
                                
                                <div style="margin-top: 20px; padding: 15px; background: white; border-radius: 8px; text-align: center; border: 1px solid #e0e0e0;">
                                    <div style="font-size: 14px; color: #666;">Base de connaissances fromagère</div>
                                    <div style="font-size: 16px; font-weight: bold; color: #333; margin-top: 5px;">
                                        {fallback_count} recettes documentées
                                    </div>
                                </div>
                            </div>
                            """
                            
                            print(f"✅ Stats: {fallback_count} recettes de référence")
                            return stats_html
                            
                        except Exception as e:
                            print(f"❌ Erreur show_stats: {e}")
                            return f"<div style='color: red; padding: 20px;'>❌ Erreur: {str(e)}</div>"
                    
                    def show_fallback():
                        """Affiche TOUTES les recettes de référence"""
                        try:
                            print("📖 Début show_fallback")
                            
                            global fallback_cache
                            if fallback_cache is None:
                                fallback_cache = agent._get_absolute_fallback("", "", 1000)
                            
                            real_count = len(fallback_cache)
                            print(f"   📊 Affichage de {real_count} recettes")
                            
                            # Grouper par type de lait
                            lait_groups = {}
                            for recipe in fallback_cache:
                                lait = recipe.get('lait', 'mixte')
                                if lait not in lait_groups:
                                    lait_groups[lait] = []
                                lait_groups[lait].append(recipe)
                            
                            html = f"""
                            <div style="padding: 15px; max-height: 600px; overflow-y: auto;">
                                <h2 style="margin-top: 0;">📚 {real_count} RECETTES DE RÉFÉRENCE</h2>
                                <p style="color: #666; margin-bottom: 20px;">
                                    Base complète - {real_count} recettes documentées
                                </p>
                            """
                            
                            # Afficher par groupe
                            for lait, recipes in lait_groups.items():
                                lait_name = lait if lait else 'mixte'
                                lait_emoji = {'vache': '🐄', 'chèvre': '🐐', 'brebis': '🐑'}.get(lait, '🥛')
                                
                                html += f"""
                                <div style="margin-bottom: 25px; background: white; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0;">
                                    <h3 style="margin-top: 0; color: #444;">
                                        {lait_emoji} Lait de {lait_name} ({len(recipes)} recettes)
                                    </h3>
                                """
                                
                                # Limiter à 15 recettes par groupe pour ne pas surcharger
                                for i, recipe in enumerate(recipes[:15], 1):
                                    html += f"""
                                    <div style="margin-bottom: 10px; padding: 10px; border-bottom: 1px solid #f5f5f5;">
                                        <div style="font-weight: bold; color: #333;">{i}. {recipe['title']}</div>
                                        <div style="font-size: 13px; color: #666; margin: 5px 0;">{recipe['description'][:120]}...</div>
                                        <div style="font-size: 12px; color: #888;">
                                            <span>📍 {recipe['source']}</span>
                                            <span style="margin-left: 15px;">⭐ {recipe.get('score', '?')}/10</span>
                                            <a href="{recipe['url']}" target="_blank" style="margin-left: 15px; color: #2196F3; text-decoration: none;">
                                                🔗 Voir
                                            </a>
                                        </div>
                                    </div>
                                    """
                                
                                if len(recipes) > 15:
                                    html += f"""
                                    <div style="text-align: center; padding: 10px; color: #666; font-size: 13px;">
                                        ... et {len(recipes)-15} autres recettes de {lait_name}
                                    </div>
                                    """
                                
                                html += "</div>"
                            
                            # Résumé final
                            total_lait = len(lait_groups)
                            html += f"""
                                <div style="margin-top: 20px; padding: 20px; background: #f8f9fa; border-radius: 10px; text-align: center;">
                                    <div style="font-size: 14px; color: #666;">RÉSUMÉ</div>
                                    <div style="font-size: 28px; font-weight: bold; color: #333; margin: 10px 0;">{real_count}</div>
                                    <div style="font-size: 14px; color: #666;">
                                        recettes de référence | {total_lait} types de lait différents
                                    </div>
                                </div>
                            </div>
                            """
                            
                            print(f"✅ Affiché: {real_count} recettes, {len(lait_groups)} types de lait")
                            return html
                            
                        except Exception as e:
                            print(f"❌ Erreur show_fallback: {e}")
                            return f"<div style='color: red; padding: 20px;'>❌ Erreur: {str(e)}</div>"
                    
                    def clear_all():
                        """Efface l'historique - VERSION MODIFIÉE POUR LE TOGGLE"""
                        global stats_visible  # <-- AJOUTEZ CE 'global'
                        
                        # Réinitialiser l'état
                        stats_visible = False
                        
                        try:
                            print("🗑️ Début clear_all")
                            result = agent.clear_history()
                            print(f"✅ clear_all réussi: {result}")
                            
                            # Réinitialiser aussi le cache des stats
                            global STATS_CACHE
                            STATS_CACHE['visible'] = False
                            STATS_CACHE['html'] = None  # Réinitialiser le cache aussi
                            
                            return [
                                "✅ Historique effacé !",  # history_summary
                                [],                        # recipe_dropdown
                                "✅ Historique effacé",    # recipe_display
                                "<div style='padding: 20px; text-align: center; color: #666;'>Cliquez sur 'Compter' pour voir les statistiques</div>",  # stats_display
                                "🔢 Compter",              # count_btn texte
                                "secondary"                # count_btn style
                            ]
                            
                        except Exception as e:
                            print(f"❌ Erreur clear_all: {e}")
                            return [
                                f"❌ Erreur: {str(e)}",
                                [],
                                f"Erreur: {str(e)}",
                                f"<div style='color: red; padding: 20px;'>❌ Erreur: {str(e)}</div>",
                                "🔢 Compter",
                                "secondary"
                            ]
                    
                    def on_recipe_select(selected):
                        """Quand une recette est sélectionnée"""
                        
                        # ✅ FILTRER LE PLACEHOLDER - AJOUTER CES LIGNES
                        if not selected or selected == "Sélectionner parmi les recettes 👉" or selected.startswith("→"):
                            return "Sélectionnez une recette dans la liste..."
                        
                        print(f"🔍 recipe_display type: {type(recipe_display)}")
                        print(f"🔍 recipe_display: {recipe_display}")
                        
                        #Déclarer recipe_map comme GLOBAL DES LE DEBUT
                        global recipe_map
                        
                        print(f"🔍 Sélection reçue (type: {type(selected)}): {selected}")
                        
                        # ===== DEBUG ============
                        print("\n" + "="*60)
                        print("=== DEBUG COMPLET ===")
                        print("="*60)
                        print(f"Selected: {selected}")
                        print(f"Type: {type(selected)}")
                        
                        # Gérer les listes
                        if isinstance(selected, list):
                            print(f"⚠️ C'est une liste! Longueur: {len(selected)}")
                            if not selected:
                                print("❌ Liste vide")
                                return "Sélectionnez une recette..."
                            selected = selected[0]
                            print(f"✅ Premier élément extrait: {selected}")
                        
                        # Afficher le recipe_map AVANT recherche
                        print(f"\n=== RECIPE_MAP (taille: {len(recipe_map)}) ===")
                        if recipe_map:
                            print("5 premières entrées:")
                            for i, (key, value) in enumerate(list(recipe_map.items())[:5]):
                                print(f"  [{i}] '{key}' -> {value}")
                        else:
                            print("⚠️ recipe_map est VIDE!")
                        
                        # Récupérer l'historique UNE SEULE FOIS pour debug
                        history = agent.get_history()
                        print(f"\n=== HISTORIQUE ({len(history)} entrées) ===")
                        for i, entry in enumerate(history[:5]):  # Afficher seulement les 5 premières
                            print(f"[{i}] ID: {entry.get('id')} (type: {type(entry.get('id'))})")
                            print(f"    Clés disponibles: {list(entry.keys())}")
                            
                            # Afficher un aperçu du contenu
                            if 'recipe_complete' in entry:
                                content = entry['recipe_complete']
                                preview = content[:50].replace('\n', ' ') + "..." if len(content) > 50 else content
                                print(f"    Preview: {preview}")
                            print()
                        
                        if len(history) > 5:
                            print(f"... et {len(history) - 5} autres entrées")
                        
                        print("="*60 + "\n")
                        # ===== FIN DEBUG ============
                        
                        if not selected:
                            return "Sélectionnez une recette..."
                        
                        try:
                            # Chercher dans le mapping
                            recipe_id = None
                            
                            print(f"\n🔎 Recherche de '{selected}'...")
                            
                            if selected in recipe_map:
                                recipe_id = recipe_map[selected]
                                print(f"✅ Trouvé via recipe_map: {selected} -> ID {recipe_id}")
                            else:
                                # Extraire l'ID du format "ID. Nom (Date)"
                                import re
                                match = re.match(r'^(\d+)\.', str(selected))
                                if match:
                                    recipe_id = int(match.group(1))
                                    print(f"✅ ID extrait par regex: '{selected}' -> ID {recipe_id}")
                                else:
                                    # Essayer d'autres patterns
                                    print(f"⚠️ Regex échouée, tentative alternative...")
                                    numbers = re.findall(r'\d+', str(selected))
                                    if numbers:
                                        recipe_id = int(numbers[0])
                                        print(f"✅ Nombre extrait: ID {recipe_id}")
                                    else:
                                        return f"❌ Format invalide: '{selected}'"
                            
                            if recipe_id is None:
                                return "❌ Impossible de déterminer l'ID de la recette"
                            
                            # ========== DEBUG CRITIQUE ==========
                            print(f"\n🔬 RECHERCHE DÉTAILLÉE:")
                            print(f"   ID cherché: {recipe_id} (type: {type(recipe_id)})")
                            print(f"   ID comme string: '{str(recipe_id)}'")
                            
                            # Chercher la recette dans l'historique
                            history = agent.get_history()  # Re-récupérer l'historique
                            
                            print(f"\n   Parcours des {len(history)} entrées...")
                            
                            found = False
                            for i, entry in enumerate(history):
                                entry_id = entry.get('id')
                                entry_id_str = str(entry_id)
                                
                                # Vérifier différents types de correspondance
                                matches = []
                                if entry_id == recipe_id:
                                    matches.append("MATCH EXACT (entry_id == recipe_id)")
                                if entry_id_str == str(recipe_id):
                                    matches.append("MATCH STRING (str(entry_id) == str(recipe_id))")
                                if str(entry_id) == str(recipe_id):
                                    matches.append("MATCH DOUBLE STRING (str(entry_id) == str(recipe_id))")
                                
                                if matches:
                                    print(f"\n   ✅ TROUVÉ à l'index {i}!")
                                    print(f"      Entry ID: {entry_id} (type: {type(entry_id)})")
                                    print(f"      Type(s) de match: {', '.join(matches)}")
                                    print(f"      Clés de l'entrée: {list(entry.keys())}")
                                    
                                    # Chercher le contenu
                                    content_keys = ['recipe_complete', 'recipe', 'content', 'text', 'response']
                                    for key in content_keys:
                                        if key in entry:
                                            content = entry[key]
                                            print(f"      📄 Contenu trouvé dans clé '{key}' ({len(content)} caractères)")
                                            found = True
                                            
                                            # Aperçu du contenu
                                            preview = content[:100].replace('\n', ' ') + "..." if len(content) > 100 else content
                                            print(f"      Preview: {preview}")
                                            return content
                                    
                                    if not found:
                                        print(f"      ⚠️ Aucune clé de contenu trouvée!")
                                        return "⚠️ Recette sans contenu"
                                else:
                                    # Debug détaillé seulement pour quelques entrées
                                    if i < 3:  # Afficher les 3 premières comparaisons
                                        print(f"   [{i}] Entry ID: {entry_id} (vs {recipe_id}) - PAS DE MATCH")
                            
                            if not found:
                                print(f"\n❌ Aucune correspondance trouvée pour ID {recipe_id}")
                                print(f"📋 IDs présents dans l'historique: {[entry.get('id') for entry in history]}")
                                return f"❌ Recette ID {recipe_id} non trouvée"
                            
                        except Exception as e:
                            print(f"\n❌ ERREUR DÉTAILLÉE:")
                            print(f"   Message: {e}")
                            import traceback
                            traceback.print_exc()
                            return f"❌ Erreur: {str(e)}\nSélection: '{selected}'"
                                    
                    # ===== CONNECTIONS =====
                    
                    # Bouton Actualiser
                    history_btn.click(
                        fn=update_interface,
                        inputs=[],
                        outputs=[
                            counter_card,      # 0
                            history_summary,   # 1
                            recipe_dropdown,  # 3 choices
                            recipe_display,    # 4
                        ]
                    )
                    
                    # Bouton Compter (TOGGLE)
                    count_btn.click(
                        fn=toggle_stats,
                        inputs=[],
                        outputs=[
                            stats_display,  # Afficher/cacher HTML
                            count_btn,      # Changer texte bouton
                        ]
                    )
                    
                    # Bouton Effacer
                    clear_btn.click(
                        fn=clear_all,
                        inputs=[],
                        outputs=[
                            history_summary,
                            recipe_dropdown,
                            recipe_display,
                        ],
                        queue=False
                    )
                    
                    # Bouton Voir références
                    show_fallback_btn.click(
                        fn=show_fallback,
                        inputs=[],
                        outputs=[stats_display]
                    )
                    
                    # Sélection dropdown
                    recipe_dropdown.change(
                        fn=on_recipe_select,
                        inputs=[recipe_dropdown],
                        outputs=[recipe_display]
                    )
                    
                    # ===== INITIALISATION =====
                  
                    def init_on_load():
                        """Initialise avec les vrais chiffres"""
                        global stats_visible


                        stats_visible = False  # Initialiser l'état
                        print("⚡ Initialisation Historique")
                        return update_interface()
                    
                    demo.load(
                        fn=init_on_load,
                        inputs=[],
                        outputs=[
                            counter_card,
                            history_summary,
                            recipe_dropdown,
                            recipe_display,
                        ],
                        queue=False
                    )
                                                                                             
                # ONGLET 5 : Chat
                with gr.Tab("💬 Expert Fromager"):
                    gr.Markdown("### 🧀 Dialoguez avec Maître Fromager")
                    
                    chat_history = gr.State([])
                    
                    chat_display = gr.Textbox(
                        label="Conversation",
                        lines=15,
                        interactive=False,
                        elem_id="chat-display",
                    )

                    with gr.Row():
                        user_input = gr.Textbox(
                            label="Votre question",
                            placeholder="Ex: Mon fromage est trop acide...",
                            lines=3,
                            scale=4,
                        )
                        send_btn = gr.Button("💬 Envoyer", variant="primary", scale=1)

                    with gr.Row():
                        btn_problem = gr.Button("🚨 Problème", size="sm")
                        btn_recipe = gr.Button("📝 Recette", size="sm")
                        btn_wine = gr.Button("🍷 Accord vin", size="sm")
                        btn_clear_chat = gr.Button("🗑️ Effacer", size="sm")

                    def process_question(question, history):
                        if not question or not question.strip():
                            return history, "", ""
                        
                        response = agent.chat_with_llm(question, [])
                        history.append(f"👤 **Vous:** {question}")
                        history.append(f"🧀 **Maître Fromager:** {response}")
                        history.append("─" * 50)
                        
                        if len(history) > 15:
                            history = history[-15:]
                        
                        display_text = "\n\n".join(history)
                        return history, display_text, ""

                    def get_quick_question(btn_text):
                        questions = {
                            "🚨 Problème": "Mon fromage a des problèmes, que faire ?",
                            "📝 Recette": "Donne-moi une recette simple",
                            "🍷 Accord vin": "Quel vin avec un fromage de chèvre ?",
                        }
                        return questions.get(btn_text, "")

                    def clear_conversation():
                        return [], "", ""

                    send_btn.click(
                        fn=process_question,
                        inputs=[user_input, chat_history],
                        outputs=[chat_history, chat_display, user_input],
                    )

                    user_input.submit(
                        fn=process_question,
                        inputs=[user_input, chat_history],
                        outputs=[chat_history, chat_display, user_input],
                    )

                    btn_problem.click(fn=lambda: get_quick_question("🚨 Problème"), outputs=[user_input])
                    btn_recipe.click(fn=lambda: get_quick_question("📝 Recette"), outputs=[user_input])
                    btn_wine.click(fn=lambda: get_quick_question("🍷 Accord vin"), outputs=[user_input])
                    btn_clear_chat.click(fn=clear_conversation, outputs=[chat_history, chat_display, user_input])

            # ===== BOUTON GÉNÉRATION =====
            generate_all_btn.click(
                fn=generate_all,
                inputs=[
                    ingredients_input,
                    cheese_type_input,
                    constraints_input,
                    creativity_slider,
                    texture_choice,
                    affinage_slider,
                    spice_choice,
                    profile_selector,
                ],
                outputs=[
                    recipe_output,
                    search_status,
                    web_results,
                    history_summary,
                    recipe_dropdown,
                    recipe_display,
                ],
            )

            # ===== BOUTON DÉCONNEXION =====
            gr.Markdown("---")
            with gr.Row():
                gr.Markdown(f"**Connecté en tant que :** `{AUTH_USERNAME}`")
                logout_button = gr.Button("🚪 Déconnexion", variant="secondary", size="sm")
                
            gr.Markdown("""
            ---
            <center>
            Fait avec 🧀 et 🤖 | Hugging Face Spaces | © 2026 Braconier
            </center>
            """)
        
        # ===== FONCTIONS D'AUTHENTIFICATION =====
        def authenticate(username, password):
            """Vérifie les identifiants"""
            if username == AUTH_USERNAME and password == AUTH_PASSWORD:
                return (
                    gr.Column(visible=False),  # Cacher login
                    gr.Column(visible=True),   # Montrer main
                    "✅ Connexion réussie !",
                )
            else:
                return (
                    gr.Column(visible=True),   # Montrer login
                    gr.Column(visible=False),  # Cacher main
                    "❌ Identifiants incorrects",
                )
        
        def logout():
            """Déconnecte l'utilisateur"""
            return (
                gr.Column(visible=True),   # Montrer login
                gr.Column(visible=False),  # Cacher main
                "",  # Effacer le message
            )
        
        # ===== CONNEXIONS AUTHENTIFICATION =====
        login_button.click(
            fn=authenticate,
            inputs=[username_input, password_input],
            outputs=[login_screen, main_screen, login_status]
        )
        
        password_input.submit(
            fn=authenticate,
            inputs=[username_input, password_input],
            outputs=[login_screen, main_screen, login_status]
        )
        
        logout_button.click(
            fn=logout,
            outputs=[login_screen, main_screen, login_status]
        )
    
    return demo


# ===== NE PAS OUBLIER EN DÉBUT DE FICHIER =====
# AUTH_USERNAME = "admin"  # ou votre nom d'utilisateur
# AUTH_PASSWORD = "votre_mot_de_passe_securise"

def run_tests():
    """Lance des tests rapides"""
    print("\n" + "=" * 60)
    print("🧪 TESTS DE LA FONCTION _get_absolute_fallback")
    print("=" * 60)

    # Test 1: Lait de brebis
    print("\n📝 TEST 1: Lait de brebis spécifique")
    print("   Entrée: 'lait de brebis, présure'")
    recipes = agent._get_absolute_fallback(
        "lait de brebis, présure", "Fromage frais", 4
    )
    print(f"   Résultats: {len(recipes)} recettes")
    for i, r in enumerate(recipes, 1):
        print(f"   {i}. {r['title']} (lait: {r.get('lait', 'non spécifié')})")

    # Test 2: Lait de chèvre
    print("\n📝 TEST 2: Lait de chèvre spécifique")
    print("   Entrée: 'lait de chèvre, sel'")
    recipes = agent._get_absolute_fallback("lait de chèvre, sel", "Fromage frais", 4)
    print(f"   Résultats: {len(recipes)} recettes")
    for i, r in enumerate(recipes, 1):
        print(f"   {i}. {r['title']} (lait: {r.get('lait', 'non spécifié')})")

    # Test 3: Pas de lait spécifié
    print("\n📝 TEST 3: Pas de lait spécifié")
    print("   Entrée: 'présure, sel'")
    recipes = agent._get_absolute_fallback("présure, sel", "Fromage frais", 4)
    print(f"   Résultats: {len(recipes)} recettes")
    for i, r in enumerate(recipes, 1):
        print(f"   {i}. {r['title']} (lait: {r.get('lait', 'non spécifié')})")

    print("\n✅ Tests terminés!")
    print("=" * 60)

# DÉCOMMENT la ligne suivante pour lancer les tests automatiquement :
# run_tests()

    return demo  # ⬅️ IMPORTANT : retourner l'interface
# ========================================
# LANCEMENT DE L'APPLICATION
# ========================================
if __name__ == "__main__":
    # 🧀 THÈME FROMAGER - Couleurs chaudes et gourmandes
    fromage_theme = gr.themes.Soft(
        primary_hue="amber",  # Jaune doré comme un fromage affiné
        secondary_hue="orange",  # Orange crémeux
        neutral_hue="stone",  # Beige pierre comme une cave à fromage
        font=gr.themes.GoogleFont("Quicksand"),  # Police ronde et douce
    ).set(
        # Couleurs primaires
        body_background_fill="#FFF9E6",  # Crème légère
        body_background_fill_dark="#2C2416",  # Marron cave sombre
        # Boutons
        button_primary_background_fill="#FF8F00",  # Orange fromage
        button_primary_background_fill_hover="#FF6F00",  # Orange plus foncé
        button_primary_text_color="#FFFFFF",
        # Inputs
        input_background_fill="#FFFBF0",  # Blanc crémeux
        input_border_color="#FFB74D",  # Bordure orange douce
        # Tabs
        block_label_text_color="#E65100",  # Orange foncé
        block_title_text_color="#BF360C",  # Marron fromage affiné
    )

    # 🎨 CSS PERSONNALISÉ - Design fromager gourmand
    custom_css = """
    <style>
        /* ===== GLOBAL ===== */
        * {
            font-family: 'Quicksand', sans-serif !important;
        }
        
        /* Fond général avec texture fromage */
        .gradio-container {
            background-image: url('https://images.unsplash.com/photo-1452195100486-9cc805987862?w=1920') !important;
            background-size: cover !important;
            background-position: center !important;
            background-repeat: no-repeat !important;
            background-attachment: fixed !important;
        }
        /* Couche semi-transparente pour garder la lisibilité */
        .gradio-container::before {
            content: '' !important;
            position: absolute !important;
            top: 0 !important;
            left: 0 !important;
            width: 100% !important;
            height: 100% !important;
            background: linear-gradient(135deg, rgba(255, 249, 230, 0.92) 0%, rgba(255, 229, 180, 0.32) 100%) !important;
            pointer-events: none !important;
            z-index: 0 !important;
        }

        /* Assurer que le contenu reste au-dessus */
        .gradio-container > * {
            position: relative !important;
            z-index: 1 !important;
        }

        /* ===== TEXTE MARKDOWN - LISIBLE ===== */
        .prose, .markdown, p, li, span, label, .gr-box, div {
            color: #3E2723 !important;
        }
        
        /* En-tête avec ombre fromagère */
        h1, h2, h3 {
            color: #BF360C !important;
            text-shadow: 2px 2px 4px rgba(191, 54, 12, 0.2);
            font-weight: 700 !important;
        }
        
        /* Texte dans les zones d'information */
        .gr-prose p, .gr-prose li {
            color: #4E342E !important;
            font-size: 1.05em !important;
        }
        
        /* Labels des champs */
        label {
            color: #5D4037 !important;
            font-weight: 600 !important;
        }
        
        /* ===== ONGLETS - FOND OPAQUE ===== */
        .tabitem, .tab-nav, [role="tabpanel"] {
            background: #FFFBF0 !important;
            border-radius: 12px !important;
            padding: 20px !important;
        }
        
        .tab-content {
            background: #FFFBF0 !important;
            padding: 20px !important;
        }
        
        .tabs {
            background: transparent !important;
        }
        
        .tab-nav {
            background: transparent !important;
            border-bottom: 3px solid #FFE0B2 !important;
            padding: 0 !important;
        }
        
        .svelte-llgaql,
        .tab-nav button {
            background: #FFF3E0 !important;
            color: #5D4037 !important;
            border: 2px solid #FFE0B2 !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            font-size: 1.6em !important;           /* ← AJOUTÉ */
            padding: 14px 28px !important;         /* ← MODIFIÉ */
            margin: 0 4px !important;
            border-radius: 12px 12px 0 0 !important;
        }
        
        .tab-nav button:hover {
            background: #FFE0B2 !important;
            border-color: #FF8F00 !important;
            color: #3E2723 !important;
        }
        
        .tab-nav button.selected, .tab-nav button[aria-selected="true"] {
            background: linear-gradient(135deg, #FF8F00 0%, #F57C00 100%) !important;
            color: white !important;
            border-color: #E65100 !important;
            box-shadow: 0 4px 12px rgba(230, 81, 0, 0.3) !important;
        }
        
        /* ===== DROPDOWN / MENU DÉROULANT - CORRECTION COMPLÈTE ===== */
        
        /* Le champ dropdown lui-même */
        select, 
        .gr-dropdown, 
        .dropdown,
        .svelte-1gfkn6j,
        [data-testid="dropdown"] {
            background: #FFFBF0 !important;
            color: #3E2723 !important;
            border: 2px solid #FFE0B2 !important;
            border-radius: 12px !important;
            padding: 10px 16px !important;
            font-weight: 500 !important;
        }
        
        /* Texte du dropdown sélectionné */
        .gr-dropdown input,
        .dropdown input,
        .svelte-1gfkn6j input {
            background: #FFFBF0 !important;
            color: #3E2723 !important;
        }
        
        /* Menu déroulant ouvert */
        .gr-dropdown ul,
        .dropdown-menu,
        ul[role="listbox"],
        .svelte-1gfkn6j ul {
            background: #FFFBF0 !important;
            border: 2px solid #FFE0B2 !important;
            border-radius: 12px !important;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2) !important;
            padding: 8px !important;
        }
        
        /* Items du menu déroulant */
        .gr-dropdown li,
        .dropdown-item,
        li[role="option"],
        .svelte-1gfkn6j li {
            background: transparent !important;
            color: #3E2723 !important;
            padding: 10px 16px !important;
            border-radius: 8px !important;
            margin: 2px 0 !important;
            font-weight: 500 !important;
        }
        
        /* Item survolé */
        .gr-dropdown li:hover,
        .dropdown-item:hover,
        li[role="option"]:hover,
        .svelte-1gfkn6j li:hover {
            background: #FFE0B2 !important;
            color: #E65100 !important;
        }
        
        /* Item sélectionné */
        .gr-dropdown li.selected,
        .gr-dropdown li[aria-selected="true"],
        li[role="option"][aria-selected="true"],
        .svelte-1gfkn6j li.selected {
            background: #FF8F00 !important;
            color: white !important;
            font-weight: 700 !important;
        }
        
        /* Options natives du select HTML */
        option {
            background: #FFFBF0 !important;
            color: #3E2723 !important;
            padding: 8px !important;
        }
        
        option:hover,
        option:focus {
            background: #FFE0B2 !important;
            color: #E65100 !important;
        }
        
        /* Icône du dropdown */
        .gr-dropdown svg,
        .dropdown svg {
            fill: #FF8F00 !important;
        }
        
        /* ===== RADIO BUTTONS - FOND OPAQUE ===== */
        .gr-radio, .gr-radio-group {
            background: #FFFBF0 !important;
            padding: 12px !important;
            border-radius: 12px !important;
            border: 2px solid #FFE0B2 !important;
        }
        
        .gr-radio label, .gr-radio-group label {
            color: #3E2723 !important;
            font-weight: 500 !important;
        }
        
        input[type="radio"] {
            accent-color: #FF8F00 !important;
        }
        
        input[type="radio"]:checked {
            background: #FF8F00 !important;
            border-color: #E65100 !important;
        }
        
        /* ===== SLIDERS - LISIBLES ===== */
        .gr-slider {
            background: #FFFBF0 !important;
            padding: 12px !important;
            border-radius: 12px !important;
            border: 2px solid #FFE0B2 !important;
        }
        
        input[type="range"] {
            accent-color: #FF8F00 !important;
        }
        
        /* ===== BOUTON PRINCIPAL ===== */
        button[variant="primary"], .primary, button.primary {
            background: linear-gradient(135deg, #FF8F00 0%, #FF6F00 100%) !important;
            border: none !important;
            box-shadow: 0 4px 15px rgba(255, 111, 0, 0.4) !important;
            transition: all 0.3s ease !important;
            font-weight: 600 !important;
            color: white !important;
        }
        
        button[variant="primary"]:hover, .primary:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 8px 20px rgba(255, 111, 0, 0.6) !important;
        }
        
        /* Tous les autres boutons */
        button {
            background: #FFF3E0 !important;
            color: #5D4037 !important;
            border: 2px solid #FFE0B2 !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }
        
        button:hover {
            background: #FFE0B2 !important;
            border-color: #FF8F00 !important;
        }
        
        /* ===== CARTES DE RECETTES WEB ===== */
        .recipe-card {
            background: linear-gradient(145deg, #FFFBF0 0%, #FFE0B2 100%);
            border-left: 6px solid #FF8F00;
            border-radius: 16px;
            padding: 24px;
            margin: 20px 0;
            box-shadow: 
                0 4px 12px rgba(191, 54, 12, 0.15),
                inset 0 1px 0 rgba(255, 255, 255, 0.8);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .recipe-card::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(
                45deg,
                transparent,
                rgba(255, 255, 255, 0.3),
                transparent
            );
            transform: rotate(45deg);
            transition: all 0.6s;
            opacity: 0;
        }
        
        .recipe-card:hover::before {
            opacity: 1;
            left: 100%;
        }
        
        .recipe-card:hover {
            transform: translateY(-8px) scale(1.02);
            box-shadow: 
                0 12px 28px rgba(191, 54, 12, 0.25),
                0 0 0 1px rgba(255, 143, 0, 0.3);
            border-left-width: 8px;
        }
        
        .recipe-title {
            font-size: 1.4em;
            font-weight: 800;
            color: #E65100 !important;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .recipe-title::before {
            content: '🧀';
            font-size: 1.2em;
            animation: rotate 3s infinite ease-in-out;
        }
        
        @keyframes rotate {
            0%, 100% { transform: rotate(0deg); }
            50% { transform: rotate(15deg); }
        }
        
        .recipe-source {
            font-size: 0.95em;
            color: #795548 !important;
            margin-bottom: 12px;
            font-style: italic;
            font-weight: 500;
        }
        
        .recipe-description {
            color: #4E342E !important;
            line-height: 1.8;
            margin-bottom: 18px;
            font-size: 1.05em;
        }
        
        .recipe-link {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: linear-gradient(135deg, #FF8F00 0%, #F57C00 100%);
            color: white !important;
            padding: 12px 24px;
            border-radius: 50px;
            text-decoration: none;
            font-weight: 700;
            font-size: 1.05em;
            box-shadow: 0 4px 12px rgba(245, 124, 0, 0.4);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .recipe-link:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(245, 124, 0, 0.6);
            background: linear-gradient(135deg, #F57C00 0%, #E65100 100%);
        }
        
        /* ===== STATUT DE RECHERCHE ===== */
        .search-status {
            background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
            border-left: 5px solid #FF8F00;
            padding: 18px 24px;
            margin: 20px 0;
            border-radius: 12px;
            font-weight: 700;
            font-size: 1.1em;
            color: #E65100 !important;
            box-shadow: 0 3px 10px rgba(230, 81, 0, 0.2);
        }
        
        /* ===== MESSAGE "AUCUNE RECETTE" ===== */
        .no-recipes {
            text-align: center;
            padding: 60px 40px;
            color: #8D6E63 !important;
            font-style: italic;
            font-size: 1.2em;
            background: linear-gradient(135deg, #FFF8E1 0%, #FFECB3 100%);
            border-radius: 20px;
            margin: 30px 0;
            border: 3px dashed #FFB74D;
            box-shadow: inset 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .no-recipes::before {
            content: '🧀';
            display: block;
            font-size: 4em;
            margin-bottom: 16px;
            opacity: 0.5;
        }
        
        /* ===== INPUTS ET TEXTAREAS ===== */
        input, textarea {
            background: #FFFBF0 !important;
            border: 2px solid #FFE0B2 !important;
            border-radius: 12px !important;
            transition: all 0.3s ease !important;
            color: #3E2723 !important;
        }
        
        input::placeholder, textarea::placeholder {
            color: #A1887F !important;
        }
        
        input:focus, textarea:focus, select:focus {
            border-color: #FF8F00 !important;
            box-shadow: 0 0 0 3px rgba(255, 143, 0, 0.2) !important;
        }
        
        /* ===== TEXTBOX/TEXTAREA GRADIO ===== */
        .gr-text-input, .gr-text-area, .gr-textbox {
            background: #FFFBF0 !important;
            color: #3E2723 !important;
        }
        
        /* ===== COLONNES ET ROWS ===== */
        .gr-column, .gr-row {
            background: transparent !important;
        }
        
        /* ===== FOOTER ===== */
        footer {
            background: linear-gradient(135deg, #FFE0B2 0%, #FFCC80 100%) !important;
            color: #BF360C !important;
            font-weight: 600 !important;
            padding: 20px !important;
            border-top: 3px solid #FF8F00 !important;
        }
        
        footer p {
            color: #5D4037 !important;
        }
        
        /* ===== SCROLLBAR ===== */
        ::-webkit-scrollbar {
            width: 12px;
        }
        
        ::-webkit-scrollbar-track {
            background: #FFF3E0;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #FF8F00 0%, #F57C00 100%);
            border-radius: 10px;
            border: 2px solid #FFF3E0;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, #F57C00 0%, #E65100 100%);
        }
        
         /* ===== ASCENSEURS SPÉCIFIQUES ===== */
        /* Ascenseur pour la recette */
        #recipe-scroll textarea,
        #recipe-scroll .gr-textarea,
        .tabitem:nth-child(1) textarea {
            max-height: 600px !important;
            overflow-y: auto !important;
            resize: vertical !important;
        }
        
        /* Ascenseur pour le chat */
        #chat-display textarea,
        #chat-display .gr-textarea,
        .tabitem:nth-child(6) textarea {
            max-height: 500px !important;
            overflow-y: auto !important;
            resize: vertical !important;
        }
        
        /* Ascenseur pour l'historique */
        .tabitem:nth-child(4) textarea {
            max-height: 400px !important;
            overflow-y: auto !important;
            resize: vertical !important;
        }
        
        /* Style amélioré pour tous les textareas avec ascenseur */
        textarea[style*="overflow"],
        .gr-textarea[style*="overflow"] {
            scrollbar-width: thin !important;
            scrollbar-color: #FF8F00 #FFF3E0 !important;
        }
        
        /* Pour les navigateurs WebKit (Chrome, Safari, Edge) */
        textarea::-webkit-scrollbar,
        .gr-textarea::-webkit-scrollbar {
            width: 10px !important;
            height: 10px !important;
        }
        
        textarea::-webkit-scrollbar-track,
        .gr-textarea::-webkit-scrollbar-track {
            background: #FFF3E0 !important;
            border-radius: 8px !important;
        }
        
        textarea::-webkit-scrollbar-thumb,
        .gr-textarea::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #FF8F00 0%, #F57C00 100%) !important;
            border-radius: 8px !important;
            border: 2px solid #FFF3E0 !important;
        }
        
        textarea::-webkit-scrollbar-thumb:hover,
        .gr-textarea::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, #F57C00 0%, #E65100 100%) !important;
        }

        # 4. MODIFIER AUSSI L'AFFICHAGE DE LA RECETTE DANS L'HISTORIQUE
        # Dans l'onglet "🕒 Historique", modifier recipe_display :

        with gr.Tab("🕒 Historique"):
            # ... (code existant) ...
            
            with gr.Column(scale=2):
                recipe_dropdown = gr.Dropdown(
                    label="🍽️ Sélectionner une recette",
                    choices=[],
                    interactive=True,
                    value=None
                )
                
                recipe_display = gr.Textbox(
                    label="📖 Recette complète",
                    lines=15,  # Réduire de 25 à 15 pour forcer l'ascenseur
                    max_lines=50,
                    interactive=False,
                    placeholder="Sélectionnez une recette dans la liste...",
                    elem_id="history-recipe-display"
                )

        # 5. AJOUTER LE CSS POUR L'HISTORIQUE (ajouter dans custom_css)
            /* Ascenseur pour la recette dans l'historique */
            #history-recipe-display textarea,
            #history-recipe-display .gr-textarea {
                max-height: 500px !important;
                overflow-y: auto !important;
                resize: vertical !important;
            }
        
        /* ===== CONTRASTE ===== */
        strong, b {
            color: #BF360C !important;
        }
        
        em, i {
            color: #5D4037 !important;
        }
        
        code {
            background: #FFE0B2 !important;
            color: #E65100 !important;
            padding: 2px 6px;
            border-radius: 4px;
        }
        
        /* ===== ANIMATIONS ===== */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .recipe-card {
            animation: fadeInUp 0.6s ease-out;
        }
    </style>
    """

    # Créer et lancer l'interface
    interface = create_interface()
    if interface:  # Vérifier que ce n'est pas None
        interface.launch(
            theme=fromage_theme,  # <-- ICI
            css=custom_css,  # <-- ICI
            share=False,  # Optionnel
            debug=False,  # Optionnel
        )
    else:
        print("❌ Erreur: create_interface() a retourné None")
