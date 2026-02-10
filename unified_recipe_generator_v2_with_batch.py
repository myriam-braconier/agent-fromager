"""
SYSTÈME UNIFIÉ V2 - AVEC BASE DE CONNAISSANCES STATIQUE
=========================================================

Intègre :
1. Base de connaissances statique (self.knowledge_base)
2. Fichier JSON enrichi (complete_knowledge_base.json)
3. Scraping web dynamique
4. Génération LLM
5. Templates hardcodés
"""

import json
import os
import random
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Dict, Optional


class UnifiedRecipeGeneratorV2:
    """Générateur unifié avec intégration complète de la base statique"""
    
    def __init__(self, agent):
        """
        Args:
            agent: Instance de AgentFromagerHF avec accès aux LLMs et knowledge_base
        """
        self.agent = agent
        self.cache = {}
        self.history_file = "unified_recipes_history.json"
        
        # Accès à la base de connaissances statique de l'agent
        self.knowledge_base = agent.knowledge_base if hasattr(agent, 'knowledge_base') else {}
        
    # ===============================================================
    # MÉTHODE PRINCIPALE
    # ===============================================================
    
    def generate_recipe(
        self,
        ingredients: List[str],
        cheese_type: str,
        creativity: int = 1,
        profile: str = "🧀 Amateur",
        constraints: str = ""
    ) -> Dict:
        """
        Génère une recette avec stratégie multi-niveaux
        
        Niveaux de créativité :
        1 = Base statique + Templates
        2 = Base statique + Web scraping + LLM enrichissement  
        3 = Génération LLM pure (+ fallback sur niveaux inférieurs)
        """
        
        print("\n" + "="*70)
        print("🧀 GÉNÉRATEUR UNIFIÉ V2 (avec base statique)")
        print("="*70)
        print(f"📝 Ingrédients: {', '.join(ingredients)}")
        print(f"🧀 Type: {cheese_type}")
        print(f"🎨 Créativité: {creativity}/3")
        print(f"👤 Profil: {profile}")
        
        lait = self._extract_lait(ingredients)
        print(f"🥛 Lait détecté: {lait or 'non spécifié'}")
        
        recipe_data = None
        
        # ===========================================================
        # NIVEAU 3 : GÉNÉRATION LLM PURE
        # ===========================================================
        
        if creativity >= 3 and self._has_llm_available():
            print("\n🤖 MODE : GÉNÉRATION LLM PURE (avec contexte base statique)")
            print("-"*70)
            
            try:
                recipe_data = self._generate_with_llm_and_knowledge(
                    ingredients=ingredients,
                    cheese_type=cheese_type,
                    lait=lait,
                    profile=profile,
                    constraints=constraints
                )
                
                if recipe_data:
                    print("✅ Recette générée par LLM (enrichie base statique)")
                    recipe_data['generation_mode'] = 'llm_pure_with_knowledge'
                    
            except Exception as e:
                print(f"⚠️ Génération LLM échouée : {e}")
        
        # ===========================================================
        # NIVEAU 2 : BASE ENRICHIE + WEB SCRAPING + LLM
        # ===========================================================
        
        if not recipe_data and creativity >= 2:
            print("\n🌐 MODE : BASE ENRICHIE + WEB + LLM")
            print("-"*70)
            
            # Essayer d'abord la base enrichie (complete_knowledge_base.json)
            recipe_data = self._search_enriched_base(ingredients, cheese_type, lait)
            
            if recipe_data:
                print("✅ Recette trouvée dans base enrichie")
                recipe_data['generation_mode'] = 'enriched_base'
            
            # Sinon essayer le scraping web
            if not recipe_data:
                try:
                    scraped = self._scrape_web_recipe(ingredients, cheese_type, lait)
                    
                    if scraped:
                        # Enrichir avec LLM si disponible
                        if self._has_llm_available():
                            recipe_data = self._enrich_with_llm_and_knowledge(
                                scraped,
                                ingredients,
                                cheese_type,
                                profile,
                                constraints
                            )
                        else:
                            recipe_data = scraped
                        
                        if recipe_data:
                            print("✅ Recette scrapée et enrichie")
                            recipe_data['generation_mode'] = 'web_enriched'
                            
                except Exception as e:
                    print(f"⚠️ Scraping échoué : {e}")
        
        # ===========================================================
        # NIVEAU 1 : BASE STATIQUE + TEMPLATES
        # ===========================================================
        
        if not recipe_data:
            print("\n📋 MODE : BASE STATIQUE + TEMPLATES")
            print("-"*70)
            
            recipe_data = self._generate_from_static_knowledge(
                ingredients=ingredients,
                cheese_type=cheese_type,
                lait=lait,
                profile=profile,
                constraints=constraints
            )
            
            recipe_data['generation_mode'] = 'static_knowledge'
            print("✅ Recette générée depuis base statique")
        
        # ===========================================================
        # FINALISATION
        # ===========================================================
        
        recipe_data['profile'] = profile
        recipe_data['creativity_level'] = creativity
        recipe_data['generated_at'] = datetime.now().isoformat()
        recipe_data['ingredients_input'] = ingredients
        recipe_data['cheese_type_input'] = cheese_type
        
        # ❌ NE PLUS SAUVEGARDER ICI (les recettes sont déjà sauvegardées pendant scraping/génération LLM)
        # self._save_to_history(recipe_data)
        
        print("\n" + "="*70)
        print(f"✅ RECETTE GÉNÉRÉE (mode: {recipe_data['generation_mode']})")
        print("="*70)
        
        return recipe_data
    
    # ===============================================================
    # RECHERCHE DANS BASE ENRICHIE (JSON)
    # ===============================================================
    
    def _search_enriched_base(
        self,
        ingredients: List[str],
        cheese_type: str,
        lait: Optional[str]
    ) -> Optional[Dict]:
        """Cherche dans complete_knowledge_base.json"""
        
        enriched_file = "complete_knowledge_base.json"
        
        if not os.path.exists(enriched_file):
            print("   ℹ️ Pas de base enrichie (complete_knowledge_base.json)")
            return None
        
        try:
            with open(enriched_file, 'r', encoding='utf-8') as f:
                enriched_recipes = json.load(f)
            
            if not enriched_recipes:
                return None
            
            print(f"   📚 Base enrichie : {len(enriched_recipes)} recettes")
            
            # Filtrer par lait si spécifié
            if lait:
                filtered = [r for r in enriched_recipes if r.get('lait') == lait]
                if filtered:
                    print(f"   🎯 {len(filtered)} recettes pour lait de {lait}")
                    # Prendre la meilleure
                    best = max(filtered, key=lambda x: x.get('score', 0))
                    return best
            
            # Sinon prendre la meilleure globalement
            best = max(enriched_recipes, key=lambda x: x.get('score', 0))
            return best
            
        except Exception as e:
            print(f"   ⚠️ Erreur lecture base enrichie : {e}")
            return None
    
    # ===============================================================
    # GÉNÉRATION AVEC BASE STATIQUE (knowledge_base)
    # ===============================================================
    
    def _generate_from_static_knowledge(
        self,
        ingredients: List[str],
        cheese_type: str,
        lait: Optional[str],
        profile: str,
        constraints: str
    ) -> Dict:
        """Génère une recette en utilisant la base de connaissances statique"""
        
        import hashlib
        
        # Seed basé sur les ingrédients
        ingredients_str = ",".join(sorted(ingredients))
        seed = int(hashlib.md5(ingredients_str.encode()).hexdigest()[:8], 16) % 1000
        
        # Contexte profil
        profile_context = self._get_profile_context(profile)
        
        # Récupérer les infos du type de fromage depuis la base statique
        type_info = self._get_type_info_from_knowledge(cheese_type)
        
        # Nom créatif
        prefixes = ["Artisanal", "Fermier", "Maison", "du Terroir", "Authentique"]
        suffixes = ["Frais", "Traditionnel", "Rustique", "Nature", "Gourmand"]
        
        random.seed(seed)
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        
        title = f"{prefix} {cheese_type} {suffix}"
        
        # Ingrédients avec quantités adaptées au profil
        quantite_lait = profile_context['quantite_lait']
        
        ingredients_list = [
            f"{quantite_lait} lait {lait or 'entier'}",
        ]
        
        # Ajouter présure et ferments depuis la base statique
        if 'ingredients_base' in self.knowledge_base:
            if 'ferments' in self.knowledge_base['ingredients_base']:
                ingredients_list.append("Ferments lactiques (selon la base)")
            if 'presure' in self.knowledge_base['ingredients_base']:
                ingredients_list.append("Présure (selon dosage recommandé)")
        else:
            # Fallback si pas de base
            ingredients_list.extend([
                "5 ml présure liquide",
                "2 g ferments lactiques"
            ])
        
        ingredients_list.append(f"{profile_context['sel']} sel fin non iodé")
        
        # Ajouter aromates/épices depuis la base statique
        aromates = self._extract_aromates(ingredients)
        
        if aromates and 'epices_et_aromates' in self.knowledge_base:
            for aromate in aromates:
                # Vérifier le dosage recommandé
                dosage = self._get_dosage_from_knowledge(aromate)
                ingredients_list.append(f"{dosage} {aromate}")
        else:
            for aromate in aromates:
                ingredients_list.append(f"1 cuillère à café de {aromate}")
        
        # Étapes basées sur la base de connaissances
        etapes = self._generate_steps_from_knowledge(
            cheese_type,
            quantite_lait,
            type_info,
            profile_context
        )
        
        # Température d'affinage depuis la base
        temp_affinage = self._get_temperature_affinage_from_knowledge(cheese_type)
        
        # Conseils depuis la base
        conseils_base = self._get_conseils_from_knowledge(cheese_type)
        conseils = f"{profile_context['conseil']}\n\n{conseils_base}"
        
        # Construire la recette
        recipe = {
            'title': title,
            'description': f"{type_info.get('description', f'Fromage {cheese_type.lower()}')} adapté au profil {profile}",
            'lait': lait or 'vache',
            'type_pate': cheese_type,
            'ingredients': ingredients_list,
            'etapes': etapes,
            'duree_totale': type_info.get('duree', profile_context['duree_totale']),
            'difficulte': type_info.get('difficulte', profile_context['difficulte']),
            'temperature_affinage': temp_affinage,
            'conseils': conseils,
            'score': 7,
            'seed': seed
        }
        
        print(f"   📝 Recette basée sur knowledge_base : {title}")
        
        return recipe
    
    # ===============================================================
    # GÉNÉRATION LLM AVEC CONTEXTE BASE STATIQUE
    # ===============================================================
    
    def _generate_with_llm_and_knowledge(
        self,
        ingredients: List[str],
        cheese_type: str,
        lait: Optional[str],
        profile: str,
        constraints: str
    ) -> Optional[Dict]:
        """Génère avec LLM en utilisant le contexte de la base statique"""
        
        seed = int(time.time() * 1000 + random.randint(1, 999))
        
        # Récupérer le contexte depuis la base statique
        type_info = self._get_type_info_from_knowledge(cheese_type)
        aromates = self._extract_aromates(ingredients)
        profile_context = self._get_profile_context(profile)
        
        # Construire un contexte enrichi pour le LLM
        knowledge_context = f"""
**CONTEXTE DEPUIS LA BASE DE CONNAISSANCES:**

Type de fromage : {cheese_type}
- Description : {type_info.get('description', 'N/A')}
- Exemples similaires : {type_info.get('exemples', 'N/A')}
- Durée typique : {type_info.get('duree', 'N/A')}
- Difficulté : {type_info.get('difficulte', 'N/A')}

Température d'affinage recommandée : {self._get_temperature_affinage_from_knowledge(cheese_type)}

Aromates détectés : {', '.join(aromates) if aromates else 'aucun'}
"""
        
        # Ajouter les dosages recommandés si disponibles
        if aromates and 'dosages_recommandes' in self.knowledge_base:
            knowledge_context += "\nDosages recommandés :\n"
            for aromate in aromates:
                dosage = self._get_dosage_from_knowledge(aromate)
                knowledge_context += f"- {aromate} : {dosage}\n"
        
        prompt = f"""Tu es un maître fromager expert. Génère UNE recette UNIQUE et TECHNIQUE.

**CONTEXTE UTILISATEUR:**
- Ingrédients : {', '.join(ingredients)}
- Type de lait : {lait or "au choix"}
- Type de fromage : {cheese_type}
- Profil : {profile}
- Contraintes : {constraints or "aucune"}

{knowledge_context}

**PROFIL:**
{profile_context}

**SEED: {seed}**

**CONSIGNE:** Réponds UNIQUEMENT avec un JSON valide (sans markdown):

{{
    "title": "Nom original et appétissant",
    "description": "Description technique (150 caractères)",
    "lait": "{lait or 'vache'}",
    "type_pate": "{cheese_type}",
    "ingredients": ["Quantité + ingrédient", "..."],
    "etapes": ["1. Étape détaillée avec T° et durée", "...", "Minimum 6 étapes"],
    "duree_totale": "{type_info.get('duree', '24h')}",
    "difficulte": "{type_info.get('difficulte', 'Moyenne')}",
    "temperature_affinage": "T° précise",
    "conseils": "Conseils adaptés au profil {profile}",
    "score": 8.5
}}"""

        try:
            response = self.agent.chat_with_llm(prompt, [])
            
            # Nettoyage
            response = response.strip()
            if '```json' in response:
                response = response.replace('```json', '').replace('```', '')
            elif '```' in response:
                response = response.replace('```', '')
            response = response.strip()
            
            # Extraction JSON
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start == -1 or end <= start:
                raise ValueError("Pas de JSON trouvé")
            
            json_str = response[start:end]
            data = json.loads(json_str)
            
            if not data.get('title') or not data.get('etapes'):
                raise ValueError("JSON incomplet")
            
            print(f"   📝 LLM: {data['title']}")
            print(f"   🔢 {len(data.get('etapes', []))} étapes")
            
            return data
            
        except Exception as e:
            print(f"   ❌ Erreur LLM: {e}")
            return None
    
    # ===============================================================
    # HELPERS : ACCÈS À LA BASE STATIQUE
    # ===============================================================
    
    def _get_type_info_from_knowledge(self, cheese_type: str) -> Dict:
        """Récupère les infos d'un type depuis la base statique"""
        
        if not self.knowledge_base or 'types_pate' not in self.knowledge_base:
            return {
                'description': f'Fromage de type {cheese_type}',
                'exemples': 'Variés',
                'duree': '24 heures à plusieurs semaines',
                'difficulte': 'Moyenne'
            }
        
        # Chercher le type (correspondance exacte ou partielle)
        types_pate = self.knowledge_base['types_pate']
        
        # Exacte
        if cheese_type in types_pate:
            return types_pate[cheese_type]
        
        # Partielle (ex: "Fromage frais" dans "Fromage frais maison")
        for key, info in types_pate.items():
            if cheese_type.lower() in key.lower() or key.lower() in cheese_type.lower():
                return info
        
        # Défaut
        return {
            'description': f'Fromage de type {cheese_type}',
            'exemples': 'Variés',
            'duree': 'Variable',
            'difficulte': 'Moyenne'
        }
    
    def _get_temperature_affinage_from_knowledge(self, cheese_type: str) -> str:
        """Récupère la température d'affinage depuis la base"""
        
        if not self.knowledge_base or 'temperatures_affinage' not in self.knowledge_base:
            return "12°C, 85% HR"
        
        temps = self.knowledge_base['temperatures_affinage']
        
        # Chercher par correspondance
        for key, value in temps.items():
            if key.lower() in cheese_type.lower() or cheese_type.lower() in key.lower():
                return value
        
        return "12°C, 85% HR"
    
    def _get_dosage_from_knowledge(self, ingredient: str) -> str:
        """Récupère le dosage recommandé depuis la base"""
        
        if not self.knowledge_base or 'dosages_recommandes' not in self.knowledge_base:
            return "selon goût"
        
        dosages = self.knowledge_base['dosages_recommandes']
        
        # Chercher
        if ingredient in dosages:
            return dosages[ingredient]
        
        # Chercher par correspondance partielle
        for key, value in dosages.items():
            if ingredient.lower() in key.lower() or key.lower() in ingredient.lower():
                return value
        
        return "selon goût"
    
    def _get_conseils_from_knowledge(self, cheese_type: str) -> str:
        """Récupère les conseils depuis la base (problèmes courants, etc.)"""
        
        conseils = []
        
        if self.knowledge_base and 'problemes_courants' in self.knowledge_base:
            # Prendre 2-3 problèmes courants pertinents
            problemes = list(self.knowledge_base['problemes_courants'].items())[:3]
            for probleme, solution in problemes:
                conseils.append(f"❌ {probleme}\n   ✅ {solution}")
        
        return "\n".join(conseils) if conseils else "Respectez les températures et l'hygiène."
    
    def _generate_steps_from_knowledge(
        self,
        cheese_type: str,
        quantite_lait: str,
        type_info: Dict,
        profile_context: Dict
    ) -> List[str]:
        """Génère les étapes en utilisant la base de connaissances"""
        
        # Étapes de base standard
        etapes = [
            f"1. Chauffer {quantite_lait} lait à 32°C en remuant doucement (20 min).",
            "2. Retirer du feu, ajouter les ferments, mélanger 2 min.",
            "3. Laisser maturer 30 min à température ambiante couvert.",
            "4. Ajouter la présure diluée, mélanger 1 min.",
            "5. Laisser cailler 45 min sans bouger (test de la coupure nette).",
            "6. Découper le caillé en cubes de 1-2 cm.",
            "7. Brasser délicatement 15 min.",
            f"8. Mouler, égoutter {profile_context['temps_egouttage']} en retournant.",
        ]
        
        # Ajouter affinage si nécessaire
        if "affiné" in cheese_type.lower() or "molle" in cheese_type.lower() or "pressée" in cheese_type.lower():
            temp = self._get_temperature_affinage_from_knowledge(cheese_type)
            etapes.append(f"9. Saler à sec, affiner {profile_context['temps_affinage']} à {temp}.")
        
        return etapes
    
    # ===============================================================
    # SCRAPING WEB (comme avant)
    # ===============================================================
    
    def _scrape_web_recipe(self, ingredients, cheese_type, lait):
        """Scrape PLUSIEURS recettes (6 max) et les sauvegarde toutes"""
        query = self._build_search_query(ingredients, cheese_type, lait)
        print(f"   🔍 Requête: {query}")
        
        urls = self._find_recipe_urls(query)
        if not urls:
            return None
        
        print(f"   🌐 {len(urls)} URLs à tester")
        
        scraped_recipes = []
        max_recipes = 6  # ✅ Scraper jusqu'à 6 recettes
        
        for url in urls:
            if len(scraped_recipes) >= max_recipes:
                break
                
            try:
                recipe = self._scrape_url(url)
                if recipe:
                    scraped_recipes.append(recipe)
                    print(f"   ✅ {len(scraped_recipes)}/{max_recipes} recettes scrapées")
            except Exception as e:
                print(f"   ⚠️ Erreur scraping {url[:50]}: {e}")
                continue
        
        print(f"\n   📊 Total scrapé: {len(scraped_recipes)} recettes")
        
        # Retourner la première recette (meilleur score) pour la génération
        return scraped_recipes[0] if scraped_recipes else None
    
    def _scrape_url(self, url):
        """Scrape une URL, enrichit avec LLM et sauvegarde"""
        if url in self.cache:
            return self.cache[url]
        
        print(f"      🌐 Scraping: {url[:60]}")
        
        try:
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extraire titre
            title = soup.find('h1')
            title_text = title.get_text(strip=True) if title else "Recette fromage"
            
            # Extraire description
            description = ""
            meta_desc = soup.find('meta', {'name': 'description'})
            if meta_desc:
                description = meta_desc.get('content', '')[:200]
            else:
                first_p = soup.find('p')
                if first_p:
                    description = first_p.get_text(strip=True)[:200]
            
            # Extraire tout le texte
            raw_html = soup.get_text(separator='\n', strip=True)[:5000]
            
            # ✅ ENRICHIR avec le LLM pour extraire ingrédients/étapes
            enriched_recipe = self._enrich_scraped_with_llm(
                title=title_text,
                description=description,
                url=url,
                raw_text=raw_html
            )
            
            if enriched_recipe:
                enriched_recipe['source'] = self._extract_domain(url)
                enriched_recipe['source_type'] = 'scraped'
                enriched_recipe['url'] = url
                enriched_recipe['generated_at'] = datetime.now().isoformat()
                enriched_recipe['score'] = 8
                
                self.cache[url] = enriched_recipe
                
                # ✅ SAUVEGARDER dans l'historique dynamique
                self._save_to_history(enriched_recipe)
                print(f"      ✅ Sauvegardée: {title_text[:50]}")
                
                return enriched_recipe
            else:
                print(f"      ⚠️ Enrichissement échoué")
                return None
            
        except Exception as e:
            print(f"      ❌ Erreur: {e}")
            return None
    
    def _enrich_scraped_with_llm(self, title, description, url, raw_text):
        """Enrichit une recette scrapée avec le LLM pour extraire détails"""
        
        if not self._has_llm_available():
            print(f"      ⚠️ Pas de LLM disponible pour enrichir")
            # Retourner une version minimale
            return {
                'title': title,
                'description': description,
                'lait': None,
                'type_pate': 'Fromage',
                'ingredients': ["Voir la source pour les détails"],
                'etapes': ["Consulter la recette complète sur le site source"],
                'duree_totale': 'Voir source',
                'difficulte': 'Moyenne'
            }
        
        prompt = f"""Analyse ce texte de recette fromage et extrais UNIQUEMENT les informations.

**TITRE:** {title}
**DESCRIPTION:** {description}
**URL:** {url}

**TEXTE COMPLET DE LA PAGE:**
{raw_text[:3000]}

**CONSIGNE:** Extrais les informations et réponds en JSON (sans markdown, sans ```):

{{
    "title": "Titre exact de la recette",
    "description": "Description courte",
    "lait": "vache/chèvre/brebis/bufflonne ou null",
    "type_pate": "Fromage frais/Pâte molle/etc ou null",
    "ingredients": ["Liste COMPLÈTE avec quantités", "Ex: 2L lait entier", "..."],
    "etapes": ["Étape 1 détaillée", "Étape 2 détaillée", "..."],
    "duree_totale": "Durée totale si trouvée",
    "difficulte": "Facile/Moyenne/Difficile"
}}

Si une info manque dans le texte, utilise null."""

        try:
            response = self.agent.chat_with_llm(prompt, [])
            
            # Nettoyage
            response = response.strip()
            response = response.replace('```json', '').replace('```', '').strip()
            
            # Extraire JSON
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start == -1:
                return None
            
            json_str = response[start:end]
            enriched = json.loads(json_str)
            
            print(f"      🤖 Enrichi avec {len(enriched.get('ingredients', []))} ingrédients, {len(enriched.get('etapes', []))} étapes")
            
            return enriched
            
        except Exception as e:
            print(f"      ⚠️ Erreur enrichissement LLM: {e}")
            # Retourner version minimale en cas d'erreur
            return {
                'title': title,
                'description': description,
                'lait': None,
                'type_pate': 'Fromage',
                'ingredients': ["Voir la source pour les détails"],
                'etapes': ["Consulter la recette complète sur le site source"],
                'duree_totale': 'Voir source',
                'difficulte': 'Moyenne'
            }
    
    def _enrich_with_llm_and_knowledge(self, scraped, ingredients, cheese_type, profile, constraints):
        """Enrichit avec LLM + contexte base statique"""
        
        type_info = self._get_type_info_from_knowledge(cheese_type)
        profile_context = self._get_profile_context(profile)
        
        knowledge_context = f"""
Type de fromage : {cheese_type}
- {type_info.get('description', 'N/A')}
- Durée : {type_info.get('duree', 'N/A')}
- Difficulté : {type_info.get('difficulte', 'N/A')}
"""
        
        prompt = f"""Analyse ce texte de recette et extrais les informations.

**CONTEXTE:**
- Ingrédients souhaités : {', '.join(ingredients)}
- Type : {cheese_type}
- Profil : {profile}

{knowledge_context}

**TEXTE RECETTE:**
{scraped.get('raw_html', '')[:2500]}

Réponds JSON uniquement (sans markdown):
{{
    "title": "Titre adapté",
    "description": "Description",
    "lait": "type ou null",
    "type_pate": "type exact",
    "ingredients": ["liste complète"],
    "etapes": ["étapes détaillées"],
    "duree_totale": "durée",
    "difficulte": "{profile_context['difficulte']}",
    "temperature_affinage": "T°",
    "conseils": "Conseils {profile}"
}}"""

        try:
            response = self.agent.chat_with_llm(prompt, [])
            response = response.strip().replace('```json', '').replace('```', '').strip()
            
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start == -1:
                return scraped
            
            json_str = response[start:end]
            enriched = json.loads(json_str)
            
            return {**scraped, **enriched}
            
        except:
            return scraped
    
    # ===============================================================
    # MÉTHODES UTILITAIRES (identiques à V1)
    # ===============================================================
    
    def _extract_lait(self, ingredients):
        ingredients_str = " ".join(ingredients).lower()
        if "brebis" in ingredients_str:
            return "brebis"
        elif "chèvre" in ingredients_str or "chevre" in ingredients_str:
            return "chèvre"
        elif "bufflonne" in ingredients_str:
            return "bufflonne"
        elif "vache" in ingredients_str:
            return "vache"
        return None
    
    def _extract_aromates(self, ingredients):
        aromates_list = [
            "thym", "romarin", "basilic", "origan", "ail", "poivre",
            "cumin", "ciboulette", "persil", "aneth", "estragon"
        ]
        found = []
        ingredients_str = " ".join(ingredients).lower()
        for aromate in aromates_list:
            if aromate in ingredients_str:
                found.append(aromate)
        return found
    
    def _get_profile_context(self, profile):
        contexts = {
            "🧀 Amateur": {
                "quantite_lait": "1 litre",
                "sel": "10g",
                "temps_egouttage": "6h",
                "temps_affinage": "1 semaine",
                "duree_totale": "24-48 heures",
                "difficulte": "Facile",
                "conseil": "✨ Conseil débutant : Commencez petit !"
            },
            "🏭 Producteur": {
                "quantite_lait": "10 litres",
                "sel": "100g",
                "temps_egouttage": "12h",
                "temps_affinage": "2-8 semaines",
                "duree_totale": "2-8 semaines",
                "difficulte": "Technique",
                "conseil": "📊 Conseil pro : Mesurez le pH."
            },
            "🎓 Formateur": {
                "quantite_lait": "5 litres",
                "sel": "50g",
                "temps_egouttage": "8h",
                "temps_affinage": "Variable",
                "duree_totale": "Variable",
                "difficulte": "Pédagogique",
                "conseil": "🎓 Conseil formateur : Préparez des questions."
            }
        }
        return contexts.get(profile, contexts["🧀 Amateur"])
    
    def _has_llm_available(self):
        return any([
            getattr(self.agent, 'openrouter_enabled', False),
            getattr(self.agent, 'google_ai_enabled', False),
            getattr(self.agent, 'together_enabled', False),
            getattr(self.agent, 'ollama_enabled', False)
        ])
    
    def _build_search_query(self, ingredients, cheese_type, lait):
        parts = ["recette", "fromage"]
        if lait:
            parts.append(lait)
        if "frais" in cheese_type.lower():
            parts.append("frais")
        aromates = self._extract_aromates(ingredients)
        if aromates:
            parts.append(aromates[0])
        parts.append("maison")
        return " ".join(parts)
    
    def _find_recipe_urls(self, query):
        """Trouve des URLs de recettes (15 max pour avoir au moins 6 qui fonctionnent)"""
        try:
            if hasattr(self.agent, '_try_duckduckgo_html'):
                results = self.agent._try_duckduckgo_html(query, 15)  # ✅ Demander 15 résultats
                if results:
                    urls = [r['url'] for r in results if r.get('url')]
                    print(f"      🔎 DuckDuckGo: {len(urls)} URLs trouvées")
                    return urls
        except Exception as e:
            print(f"      ⚠️ Recherche DuckDuckGo échouée: {e}")
        
        # URLs par défaut ÉTENDUES (au moins 6 par catégorie)
        base_urls = {
            "fromage frais": [
                "https://www.marmiton.org/recettes/recette_fromage-frais-maison_337338.aspx",
                "https://cuisine.journaldesfemmes.fr/recette/315921-fromage-blanc-maison",
                "https://www.750g.com/recette-fromage-blanc-maison-r201534.htm",
                "https://www.cuisineaz.com/recettes/fromage-blanc-maison-13742.aspx",
                "https://chefsimon.com/gourmets/chef-simon/recettes/faisselle-maison",
                "https://www.ptitchef.com/recettes/autre/fromage-blanc-maison-fid-1565941"
            ],
            "fromage chèvre": [
                "https://www.750g.com/faire-son-fromage-de-chevre-maison-r152700.htm",
                "https://www.marmiton.org/recettes/recette_fromage-de-chevre-frais-maison_166133.aspx",
                "https://cuisine.journaldesfemmes.fr/recette/1019476-fromage-de-chevre-maison",
                "https://www.cuisineaz.com/recettes/fromage-de-chevre-frais-11284.aspx",
                "https://chefsimon.com/gourmets/chef-simon/recettes/fromage-de-chevre-frais",
                "https://www.femmeactuelle.fr/cuisine/recettes-de-cuisine/fromage-de-chevre-maison-2088825"
            ],
            "mozzarella": [
                "https://www.regal.fr/produit/fromage/recette-mozzarella-maison-100305",
                "https://cuisine.journaldesfemmes.fr/recette/347890-mozzarella-maison",
                "https://www.750g.com/mozzarella-maison-r89655.htm",
                "https://www.marmiton.org/recettes/recette_mozzarella-maison_38364.aspx",
                "https://chefsimon.com/gourmets/chef-simon/recettes/mozzarella-maison",
                "https://www.cuisineaz.com/recettes/mozzarella-maison-19847.aspx"
            ],
            "ricotta": [
                "https://cuisine.journaldesfemmes.fr/recette/415921-ricotta-maison",
                "https://www.750g.com/ricotta-maison-r51237.htm",
                "https://www.marmiton.org/recettes/recette_ricotta-maison_29890.aspx",
                "https://chefsimon.com/gourmets/chef-simon/recettes/ricotta-maison",
                "https://www.cuisineaz.com/recettes/ricotta-maison-61847.aspx",
                "https://www.ptitchef.com/recettes/autre/ricotta-maison-fid-1520134"
            ]
        }
        
        # Essayer de matcher avec la requête
        for key, urls in base_urls.items():
            if key in query.lower():
                print(f"      📋 URLs par défaut: {len(urls)} pour '{key}'")
                return urls
        
        # Si aucun match, retourner un mix de toutes les catégories
        all_urls = []
        for urls in base_urls.values():
            all_urls.extend(urls[:2])  # 2 URLs par catégorie
        
        print(f"      📋 Mix d'URLs génériques: {len(all_urls)}")
        return all_urls
    
    def _extract_domain(self, url):
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.replace('www.', '')
        except:
            return 'inconnu'
    
    def _save_to_history(self, recipe_data):
        try:
            history = []
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            
            history.append(recipe_data)
            history = history[-100:]
            
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Sauvegardé dans {self.history_file}")
        except Exception as e:
            print(f"⚠️ Sauvegarde échouée: {e}")


# ===============================================================
# FORMATEUR (identique à V1)
# ===============================================================

class RecipeFormatter:
    """Formate les recettes JSON en texte lisible"""
    
    @staticmethod
    def format_to_text(recipe_data: Dict) -> str:
        """Convertit JSON en texte formaté"""
        
        # ===== GÉNÉRATION INTELLIGENTE DU TITRE =====
        titre_base = recipe_data.get('title', 'Fromage Maison')
        
        # Si le titre est générique, créer un titre personnalisé
        if titre_base.upper() in ['FROMAGE PERSONNALISÉ', 'FROMAGE MAISON', 'FROMAGE']:
            import random
            
            lait = recipe_data.get('lait', 'vache')
            type_pate = recipe_data.get('type_pate', 'Fromage frais')
            ingredients = recipe_data.get('ingredients', [])
            profile = recipe_data.get('profile', 'Standard')
            
            # Extraire herbes/épices des ingrédients
            herbes = []
            for ing in ingredients:
                ing_lower = str(ing).lower()
                if any(h in ing_lower for h in ['thym', 'romarin', 'basilic', 'herbe', 'épice', 'poivre', 'ail', 'ciboulette', 'persil']):
                    # Extraire juste le nom de l'herbe
                    for herb_name in ['thym', 'romarin', 'basilic', 'poivre', 'ail', 'ciboulette', 'persil']:
                        if herb_name in ing_lower:
                            herbes.append(herb_name)
                            break
            
            # Noms de base selon le type de lait
            base_noms = {
                'vache': ['TOMME', 'FERMIER', 'CAMPAGNARD', 'TERROIR'],
                'chèvre': ['CABRI', 'CHÈVRE', 'CAPRIN', 'CHEVROTIN'],
                'brebis': ['BREBIS', 'OVIN', 'BERGER', 'PECORINO'],
                'bufflonne': ['BUFFALO', 'BUFFLONNE', 'MOZZARELLA']
            }
            
            nom_base = random.choice(base_noms.get(lait, ['ARTISAN', 'FERMIER', 'MAISON']))
            
            # Construire le titre
            if herbes:
                title = f"{nom_base} AU {herbes[0].upper()}"
            elif 'pressée' in type_pate.lower():
                title = f"{nom_base} PÂTE PRESSÉE"
            elif 'frais' in type_pate.lower():
                title = f"{nom_base} FRAIS"
            elif 'molle' in type_pate.lower():
                title = f"{nom_base} PÂTE MOLLE"
            else:
                title = f"{nom_base} AFFINÉ"
            
            # Ajouter qualificatif selon le profil
            if profile == "🏭 Producteur" and 'AFFINÉ' not in title:
                title += " AFFINÉ"
            elif profile == "🧀 Amateur" and 'MAISON' not in nom_base:
                title += " MAISON"
        else:
            title = titre_base
        # ===== FIN GÉNÉRATION INTELLIGENTE =====
        
        description = recipe_data.get('description', '')
        lait = recipe_data.get('lait', 'vache')
        type_pate = recipe_data.get('type_pate', 'Fromage frais')
        ingredients = recipe_data.get('ingredients', [])
        etapes = recipe_data.get('etapes', [])
        duree_totale = recipe_data.get('duree_totale', 'Variable')
        difficulte = recipe_data.get('difficulte', 'Moyenne')
        temperature_affinage = recipe_data.get('temperature_affinage', 'N/A')
        conseils = recipe_data.get('conseils', '')
        score = recipe_data.get('score', 8)
        mode = recipe_data.get('generation_mode', 'unknown')
        profile = recipe_data.get('profile', 'Standard')
        
        mode_icons = {
            'llm_pure_with_knowledge': '🤖📚',
            'enriched_base': '📚',
            'web_enriched': '🌐',
            'static_knowledge': '📋'
        }
        
        mode_icon = mode_icons.get(mode, '❓')
        
        ingredients_text = "\n".join([f"  • {ing}" for ing in ingredients])
        etapes_text = "\n\n".join(etapes)
        
        formatted = f"""
    ╔==============================================================╗
    ║  {mode_icon} {title.upper()}
    ║  (Profil: {profile} | Mode: {mode})
    ║  ⭐ Score: {score}/10
    ╚==============================================================╝

    📝 DESCRIPTION
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    {description}

    📋 INFORMATIONS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    🥛 Lait : {lait.capitalize()}
    🧀 Type de pâte : {type_pate}
    ⏱️ Durée totale : {duree_totale}
    📊 Difficulté : {difficulte}
    🌡️ Température affinage : {temperature_affinage}

    🛒 INGRÉDIENTS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    {ingredients_text}

    👨‍🍳 ÉTAPES DE FABRICATION
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    {etapes_text}

    💡 CONSEILS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    {conseils}

    ⚠️ RAPPEL : Respectez les règles d'hygiène strictes en fabrication fromagère.
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ✨ Bon fromage ! Recette générée spécialement pour vous.
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
        
        return formatted