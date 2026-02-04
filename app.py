import random
import gradio as gr
import json
import os
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download

class AgentFromagerHF:
    """Agent fromager avec persistance HF Dataset"""
    
    def __init__(self):
        self.rng = random.Random()
        self.knowledge_base = self._init_knowledge()
        self.recipes_file = 'recipes_history.json'
        self.hf_repo = "volubyl/fromager-recipes"
        self.hf_token = os.environ.get("HF_TOKEN")
        self.api = HfApi(token=self.hf_token) if self.hf_token else None
        
        # Charger l'historique depuis HF au démarrage
        self._download_history_from_hf()
    
    def _init_knowledge(self):
        """Base de connaissances fromage intégrée"""
        return {
        'types_pate': {
            'Fromage frais': {
                'description': 'Non affiné, humide, à consommer rapidement',
                'exemples': 'Fromage blanc, faisselle, ricotta, cottage cheese',
                'duree': '0-3 jours',
                'difficulte': 'Facile - Idéal débutants'
            },
            'Pâte molle': {
                'description': 'Croûte fleurie ou lavée, texture crémeuse',
                'exemples': 'Camembert, brie, munster, reblochon',
                'duree': '2-8 semaines',
                'difficulte': 'Moyenne - Nécessite une cave'
            },
            'Pâte pressée non cuite': {
                'description': 'Pressée sans cuisson, texture ferme',
                'exemples': 'Cantal, saint-nectaire, morbier, tomme',
                'duree': '1-6 mois',
                'difficulte': 'Moyenne - Matériel spécifique'
            },
            'Pâte pressée cuite': {
                'description': 'Caillé chauffé puis pressé, longue conservation',
                'exemples': 'Comté, gruyère, beaufort, parmesan',
                'duree': '3-36 mois',
                'difficulte': 'Difficile - Expertise requise'
            },
            'Pâte persillée': {
                'description': 'Avec moisissures bleues, goût prononcé',
                'exemples': 'Roquefort, bleu d\'Auvergne, gorgonzola, stilton',
                'duree': '2-6 mois',
                'difficulte': 'Difficile - Contrôle précis'
            }
        },
        'ingredients_base': {
            'Lait': ['Vache (doux)', 'Chèvre (acidulé)', 'Brebis (riche)', 'Bufflonne (crémeux)', 'Mélange'],
            'Coagulant': ['Présure animale', 'Présure végétale', 'Jus de citron', 'Vinaigre blanc'],
            'Ferments': ['Lactiques (yaourt)', 'Mésophiles (température ambiante)', 'Thermophiles (haute température)'],
            'Sel': ['Sel fin', 'Gros sel', 'Sel de mer', 'Saumure (eau + sel)'],
            'Affinage': ['Penicillium roqueforti (bleu)', 'Geotrichum (croûte)', 'Herbes', 'Cendres']
        },
        'epices_et_aromates': {
            'Herbes fraîches': [
                'Basilic (doux, fromages frais)',
                'Ciboulette (léger, fromages de chèvre)',
                'Thym (robuste, tommes)',
                'Romarin (puissant, pâtes pressées)',
                'Persil (neutre, universel)',
                'Aneth (anisé, fromages nordiques)',
                'Menthe (rafraîchissant, fromages méditerranéens)',
                'Coriandre (exotique, fromages épicés)'
            ],
            'Herbes séchées': [
                'Herbes de Provence (mélange classique)',
                'Origan (italien, fromages à pizza)',
                'Sarriette (poivrée, fromages de montagne)',
                'Estragon (anisé, fromages frais)',
                'Laurier (dans saumure)',
                'Sauge (forte, pâtes dures)'
            ],
            'Épices chaudes': [
                'Poivre noir (concassé ou moulu)',
                'Poivre rouge (Espelette, piment doux)',
                'Paprika (fumé ou doux)',
                'Cumin (terreux, fromages orientaux)',
                'Curry (mélange, fromages fusion)',
                'Piment de Cayenne (fort, avec modération)',
                'Ras el hanout (complexe, fromages marocains)'
            ],
            'Épices douces': [
                'Nigelle (sésame noir, fromages levantins)',
                'Graines de fenouil (anisées)',
                'Graines de carvi (pain, fromages nordiques)',
                'Fenugrec (sirop d\'érable, rare)',
                'Coriandre en graines (agrumes)'
            ],
            'Fleurs et pollen': [
                'Lavande (Provence, délicat)',
                'Safran (luxueux, fromages d\'exception)',
                'Pétales de rose (persan, subtil)',
                'Bleuet (visuel, doux)',
                'Pollen de fleurs (sauvage)'
            ],
            'Aromates spéciaux': [
                'Ail frais (haché ou confit)',
                'Échalote (finement ciselée)',
                'Oignon rouge (mariné)',
                'Gingembre (frais râpé, fusion)',
                'Citronnelle (asiatique, rare)',
                'Zeste d\'agrumes (citron, orange, bergamote)'
            ],
            'Cendres et croûtes': [
                'Cendres végétales (charbon de bois alimentaire)',
                'Cendres de sarment de vigne',
                'Charbon actif alimentaire (noir intense)',
                'Foin séché (affinage sur foin)',
                'Paille (affinage traditionnel)'
            ],
            'Accompagnements dans la pâte': [
                'Noix concassées (texture)',
                'Noisettes (doux, chèvre)',
                'Pistaches (vert, raffiné)',
                'Fruits secs (abricots, figues)',
                'Olives (noires ou vertes)',
                'Tomates séchées (umami)',
                'Truffe (luxe absolu)',
                'Champignons séchés (boisé)'
            ]
        },
        'techniques_aromatisation': {
            'Incorporation dans le caillé': 'Ajouter les épices au moment du moulage pour distribution homogène',
            'Enrobage externe': 'Rouler le fromage dans les épices après salage',
            'Affinage aromatisé': 'Placer herbes/épices dans la cave d\'affinage',
            'Saumure parfumée': 'Infuser la saumure avec aromates',
            'Huile aromatisée': 'Badigeonner la croûte d\'huile aux herbes',
            'Couche intermédiaire': 'Saupoudrer entre deux couches de caillé'
        },
        'dosages_recommandes': {
            'Herbes fraîches': '2-3 cuillères à soupe pour 1kg de fromage',
            'Herbes séchées': '1-2 cuillères à soupe pour 1kg',
            'Épices moulues': '1-2 cuillères à café pour 1kg',
            'Épices en grains': '1 cuillère à soupe concassée pour 1kg',
            'Ail/gingembre': '1-2 gousses/morceaux pour 1kg',
            'Zestes': '1 agrume entier pour 1kg',
            'Cendres': 'Fine couche sur la croûte'
        },
        'associations_classiques': {
            'Fromage de chèvre': 'Herbes de Provence, miel, lavande',
            'Brebis': 'Piment d\'Espelette, romarin, olives',
            'Pâte molle': 'Ail, fines herbes, poivre',
            'Pâte pressée': 'Cumin, fenugrec, noix',
            'Fromage frais': 'Ciboulette, aneth, menthe fraîche',
            'Bleu': 'Noix, figues, porto (pas dans le fromage)'
        },
        'temperatures_affinage': {
            'Fromage frais': '4-6°C (réfrigérateur)',
            'Pâte molle croûte fleurie': '10-12°C, 90-95% humidité',
            'Pâte molle croûte lavée': '12-14°C, 90-95% humidité',
            'Pâte pressée non cuite': '12-14°C, 85-90% humidité',
            'Pâte pressée cuite': '14-18°C, 85-90% humidité',
            'Pâte persillée': '8-10°C, 95% humidité',
            'Chèvre': '10-12°C, 80-85% humidité'
        },
        'problemes_courants': {
            'Caillé trop dur': 'Trop de présure ou température trop haute. Solution : Réduire la dose de présure de 20%',
            'Pas de caillage': 'Lait UHT (stérilisé) ou présure périmée. Solution : Utiliser du lait cru ou pasteurisé',
            'Caillé trop mou': 'Pas assez de présure ou temps insuffisant. Solution : Attendre 15-30 min de plus',
            'Fromage trop acide': 'Fermentation trop longue ou trop chaud. Solution : Réduire température ou temps d\'affinage',
            'Fromage trop salé': 'Excès de sel ou salage trop long. Solution : Utiliser 1,5% du poids au lieu de 2%',
            'Moisissures indésirables': 'Humidité excessive ou mauvaise hygiène. Solution : Nettoyer la cave, réduire humidité',
            'Croûte craquelée': 'Air trop sec. Solution : Augmenter humidité à 85-90%',
            'Fromage trop sec': 'Égouttage excessif. Solution : Réduire temps d\'égouttage de moitié',
            'Texture granuleuse': 'Caillage incomplet ou découpe trop brutale. Solution : Attendre caillage complet',
            'Goût amer': 'Sur-affinage ou contamination bactérienne. Solution : Réduire durée d\'affinage',
            'Fromage coule': 'Température trop élevée pendant affinage. Solution : Cave à 10-12°C maximum',
            'Yeux (trous) non désirés': 'Fermentation gazeuse. Solution : Presser davantage pour éliminer l\'air'
        },
        'conservation': {
            'Fromage frais': '3-5 jours au frigo (4°C) dans boîte hermétique',
            'Pâte molle jeune': '1-2 semaines au frigo dans papier fromagerie',
            'Pâte molle affinée': '2-3 semaines, sortir 1h avant dégustation',
            'Pâte pressée non cuite': '1-2 mois au frigo, bien emballer',
            'Pâte pressée cuite': '3-6 mois au frais (10-12°C), croûte protégée',
            'Pâte persillée': '3-4 semaines, papier alu pour limiter moisissures',
            'Chèvre frais': '1 semaine maximum au frigo',
            'Chèvre affiné': '2-3 semaines en cave ou frigo',
            'Conseil général': 'Ne jamais congeler (texture détruite), emballer dans papier respirant'
        },
        'accords_vins': {
            'Fromage frais nature': 'Vin blanc sec et vif (Muscadet, Picpoul de Pinet)',
            'Fromage frais aux herbes': 'Blanc aromatique (Sauvignon, Riesling)',
            'Chèvre frais': 'Sancerre, Pouilly-Fumé, Sauvignon blanc',
            'Chèvre sec': 'Blanc minéral (Chablis) ou rouge léger (Pinot Noir)',
            'Brie, Camembert': 'Champagne, Crémant, ou rouge léger (Beaujolais)',
            'Munster, Maroilles': 'Blanc puissant (Gewurztraminer) ou bière',
            'Comté jeune': 'Vin jaune du Jura, Chardonnay',
            'Comté vieux': 'Vin jaune, Porto Tawny',
            'Cantal, Salers': 'Rouge charpenté (Cahors, Madiran)',
            'Roquefort': 'Blanc doux (Sauternes, Monbazillac) ou Porto',
            'Bleu d\'Auvergne': 'Rouge puissant (Côtes du Rhône) ou blanc moelleux',
            'Brebis des Pyrénées': 'Rouge du Sud-Ouest (Irouléguy, Madiran)',
            'Morbier': 'Vin blanc du Jura (Chardonnay)',
            'Reblochon': 'Blanc de Savoie (Apremont, Chignin)',
            'Règle d\'or': 'Accord régional : fromage et vin de la même région'
        },
        'accords_mets': {
            'Fromage frais': 'Pain complet, fruits rouges, miel, concombre',
            'Pâte molle': 'Baguette fraîche, pommes, raisins, confiture de figues',
            'Pâte pressée': 'Pain de campagne, noix, cornichons, charcuterie',
            'Pâte persillée': 'Pain aux noix, poire, miel de châtaignier, céleri',
            'Chèvre': 'Pain grillé, miel, salade verte, betterave',
            'Fromages forts': 'Pain de seigle, oignon confit, pomme de terre'
        },
        'materiel_indispensable': {
            'Pour débuter': [
                'Thermomètre de cuisson (précision ±1°C) - 10-15€',
                'Grande casserole inox 3-5L - 20-30€',
                'Moule à fromage perforé 500g - 5-10€',
                'Étamine/mousseline (toile à fromage) - 5€',
                'Louche et couteau long - 10€'
            ],
            'Pour progresser': [
                'Hygromètre pour cave (mesure humidité) - 15-20€',
                'Presse à fromage - 50-100€',
                'Set de moules variés - 30-50€',
                'pH-mètre - 30-50€',
                'Claie d\'affinage en bois - 20-40€'
            ],
            'Pour expert': [
                'Cave d\'affinage électrique - 300-800€',
                'Trancheuse à caillé professionnelle - 100€',
                'Balance de précision 0.1g - 30€',
                'Kit de cultures spécifiques - 50€/an'
            ]
        },
        'fournisseurs_recommandes': {
            'Présure et ferments': 'Tom Press, Ferments-et-vous.com, Fromage-maison.com',
            'Matériel': 'Tom Press (FR), Fromag\'Home, Le Parfait',
            'Moules': 'Amazon, Tom Press, magasins cuisine spécialisés',
            'Lait cru': 'Producteurs locaux, AMAP, marchés fermiers',
            'Livres': '"Fromages et laitages naturels faits maison" de Marie-Claire Frédéric'
        },
        'calendrier_fromager': {
            'Printemps (Mars-Mai)': 'Saison idéale pour chèvre (lait riche). Fromages frais, chèvre frais',
            'Été (Juin-Août)': 'Éviter pâtes molles (chaleur). Privilégier fromages frais, ricotta',
            'Automne (Sept-Nov)': 'Excellente période pour tous types. Lancer affinage pour Noël',
            'Hiver (Déc-Fév)': 'Fromages d\'affinage, pâtes pressées. Cave naturellement fraîche'
        }
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
                token=self.hf_token
            )
            
            with open(downloaded_path, 'r', encoding='utf-8') as src:
                history = json.load(src)
            
            with open(self.recipes_file, 'w', encoding='utf-8') as dst:
                json.dump(history, dst, indent=2, ensure_ascii=False)
            
            print(f"✅ Historique chargé : {len(history)} recettes")
            
        except Exception as e:
            print(f"ℹ️  Pas d'historique existant: {e}")
            with open(self.recipes_file, 'w', encoding='utf-8') as f:
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
                commit_message=f"Update: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
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
                with open(self.recipes_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_to_history(self, ingredients, cheese_type, constraints, recipe):
        """Sauvegarde dans l'historique LOCAL ET HF"""
        try:
            history = self._load_history()
            
            recipe_lines = recipe.split('\n')
            cheese_name = "Fromage personnalisé"
            for line in recipe_lines:
                if '🧀' in line and len(line) < 100:
                    cheese_name = line.replace('🧀', '').replace('═', '').replace('║', '').strip()
                    break
            
            entry = {
                'id': len(history) + 1,
                'date': datetime.now().isoformat(),
                'cheese_name': cheese_name,
                'ingredients': ingredients,
                'type': cheese_type,
                'constraints': constraints,
                'recipe_complete': recipe,
                'recipe_preview': recipe[:300] + "..." if len(recipe) > 300 else recipe
            }
            
            history.append(entry)
            history = history[-100:]
            
            with open(self.recipes_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            
            sync_success = self._upload_history_to_hf()
            
            if sync_success:
                print(f"✅ Recette #{entry['id']} sauvegardée et synchronisée")
            else:
                print(f"⚠️  Recette #{entry['id']} sauvegardée localement")
            
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
        display += "="*70 + "\n\n"
        
        for entry in reversed(history[-20:]):
            date_obj = datetime.fromisoformat(entry['date'])
            date_str = date_obj.strftime('%d/%m/%Y à %H:%M')
            
            display += f"🧀 #{entry['id']} - {entry.get('cheese_name', 'Fromage')}\n"
            display += f"📅 {date_str}\n"
            display += f"🏷️  Type: {entry['type']}\n"
            display += f"🥛 Ingrédients: {', '.join(entry['ingredients'][:3])}"
            
            if len(entry['ingredients']) > 3:
                display += f" (+{len(entry['ingredients'])-3} autres)"
            display += "\n"
            
            if entry.get('constraints'):
                display += f"⚙️  Contraintes: {entry['constraints']}\n"
            
            display += "-"*70 + "\n\n"
        
        if len(history) > 20:
            display += f"💡 {len(history) - 20} recettes plus anciennes disponibles\n"
        
        return display
    
    def get_recipe_by_id(self, recipe_id):
        """Récupère une recette complète par son ID"""
        history = self.get_history()
        for entry in history:
            if entry['id'] == int(recipe_id):
                return entry['recipe_complete']
        return "❌ Recette non trouvée"
    
    def clear_history(self):
        """Efface l'historique LOCAL ET HF"""
        try:
            with open(self.recipes_file, 'w', encoding='utf-8') as f:
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
        
        has_milk = any(word in ingredients_lower for word in 
                      ['lait', 'milk', 'vache', 'chèvre', 'brebis', 'bufflonne'])
        
        if not has_milk:
            return False, "❌ Il faut du lait pour faire du fromage !\n💡 Ajoutez : lait de vache, chèvre, brebis..."
        
        has_coagulant = any(word in ingredients_lower for word in 
                           ['présure', 'presure', 'citron', 'vinaigre', 'acide'])
        
        if not has_coagulant:
            return True, "⚠️ Aucun coagulant détecté. Je suggérerai présure ou citron dans la recette.\n✅ Validation OK."
        
        return True, "✅ Ingrédients parfaits pour faire du fromage !"
    
    def generate_recipe(self, ingredients, cheese_type, constraints):
        """Génère une recette de fromage détaillée"""
        
        valid, message = self.validate_ingredients(ingredients)
        if not valid:
            return message
        
        ingredients_list = [ing.strip() for ing in ingredients.split(',')]
        
        cheese_type_clean = cheese_type if cheese_type != "Laissez l'IA choisir" else "Fromage artisanal"
        
        recipe = self._generate_detailed_recipe(ingredients_list, cheese_type_clean, constraints)
        
        self._save_to_history(ingredients_list, cheese_type_clean, constraints, recipe)
        
        return recipe
    
    def _generate_detailed_recipe(self, ingredients, cheese_type, constraints):
        """Génère une recette enrichie avec la base de connaissances"""
        
        # Déterminer le type si "artisanal"
        if cheese_type == "Fromage artisanal":
            cheese_type = self._determine_type(ingredients)
        
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
    
    def _determine_type(self, ingredients):
        """Détermine le type selon les ingrédients"""
        ingredients_str = ' '.join(ingredients).lower()
        
        if 'citron' in ingredients_str or 'vinaigre' in ingredients_str:
            return "Fromage frais"
        elif any(x in ingredients_str for x in ['herbe', 'épice', 'cendr']):
            return "Pâte molle aromatisée"
        elif 'bleu' in ingredients_str or 'roquefort' in ingredients_str:
            return "Pâte persillée"
        else:
            return "Pâte molle"
    
    def _get_type_info(self, cheese_type):
        """Récupère les infos du type de fromage"""
        for key, value in self.knowledge_base['types_pate'].items():
            if key.lower() in cheese_type.lower():
                return value
        return self.knowledge_base['types_pate']['Fromage frais']
    
    def _get_temperature_affinage(self, cheese_type):
        """Récupère la température d'affinage depuis la base"""
        if 'temperatures_affinage' not in self.knowledge_base:
            return "10-12°C, 85-90% humidité"
        
        for key, value in self.knowledge_base['temperatures_affinage'].items():
            if key.lower() in cheese_type.lower():
                return value
        return "10-12°C, 85-90% humidité"
    
    def _get_conservation_info(self, cheese_type):
        """Récupère les infos de conservation"""
        if 'conservation' not in self.knowledge_base:
            return "2-3 semaines au réfrigérateur dans papier adapté"
        
        for key, value in self.knowledge_base['conservation'].items():
            if key.lower() in cheese_type.lower():
                return value
        
        # Chercher par mot-clé
        if 'frais' in cheese_type.lower():
            return self.knowledge_base['conservation'].get('Fromage frais', '3-5 jours au frigo')
        
        return "2-3 semaines au réfrigérateur dans papier adapté"
    
    def _get_accord_vin(self, cheese_type):
        """Récupère les accords vins"""
        if 'accords_vins' not in self.knowledge_base:
            return "Vin rouge de caractère ou blanc sec selon préférence"
        
        # Recherche exacte
        for key, value in self.knowledge_base['accords_vins'].items():
            if key.lower() in cheese_type.lower():
                return value
        
        # Recherche par mot-clé
        if 'frais' in cheese_type.lower():
            return self.knowledge_base['accords_vins'].get('Fromage frais nature', 'Vin blanc sec et vif')
        elif 'chèvre' in cheese_type.lower():
            return self.knowledge_base['accords_vins'].get('Chèvre frais', 'Sancerre, Sauvignon blanc')
        elif 'molle' in cheese_type.lower() or 'camembert' in cheese_type.lower():
            return self.knowledge_base['accords_vins'].get('Brie, Camembert', 'Champagne ou rouge léger')
        
        return "Vin rouge de caractère ou blanc sec selon préférence"
    
    def _get_accord_mets(self, cheese_type):
        """Récupère les accords mets"""
        if 'accords_mets' not in self.knowledge_base:
            return "Pain frais, fruits secs, miel"
        
        for key, value in self.knowledge_base['accords_mets'].items():
            if key.lower() in cheese_type.lower():
                return value
        
        # Par mot-clé
        if 'frais' in cheese_type.lower():
            return self.knowledge_base['accords_mets'].get('Fromage frais', 'Pain complet, fruits rouges, miel')
        elif 'chèvre' in cheese_type.lower():
            return self.knowledge_base['accords_mets'].get('Chèvre', 'Pain grillé, miel, salade verte')
        
        return "Pain de campagne, fruits secs, confitures"
    
    def _suggest_epices(self, ingredients, cheese_type):
        """Suggère des épices selon le type"""
        suggestions = "\n💡 SUGGESTIONS D'AROMATES (depuis la base de connaissances)\n"
        suggestions += "━"*70 + "\n"
        
        # Associations classiques
        if 'associations_classiques' in self.knowledge_base:
            for key, value in self.knowledge_base['associations_classiques'].items():
                if key.lower() in cheese_type.lower() or any(k.lower() in cheese_type.lower() for k in key.split()):
                    suggestions += f"**Idéal pour ce type** : {value}\n\n"
                    break
        
        # Techniques d'aromatisation
        if 'techniques_aromatisation' in self.knowledge_base:
            suggestions += "**Techniques d'incorporation** :\n"
            for tech, desc in list(self.knowledge_base['techniques_aromatisation'].items())[:3]:
                suggestions += f"• {tech} : {desc}\n"
            suggestions += "\n"
        
        # Dosages
        if 'dosages_recommandes' in self.knowledge_base:
            suggestions += "**Dosages recommandés** :\n"
            for ing, dosage in list(self.knowledge_base['dosages_recommandes'].items())[:4]:
                suggestions += f"• {ing} : {dosage}\n"
        
        return suggestions
    
    def _get_problemes_pertinents(self, cheese_type):
        """Liste les problèmes courants à éviter"""
        if 'problemes_courants' not in self.knowledge_base:
            return "Respecter températures et temps de repos"
        
        problemes = ""
        # Prendre les 5 problèmes les plus courants
        problemes_items = list(self.knowledge_base['problemes_courants'].items())
        selection = self.rng.sample(
            problemes_items,
            k=min(5, len(problemes_items))
)
        for prob, sol in selection:
            problemes += f"❌ **{prob}**\n"
            problemes += f"   ✅ {sol}\n\n"
                  
        return problemes
    
    def _get_materiel_debutant(self):
        """Liste le matériel pour débutants"""
        if 'materiel_indispensable' not in self.knowledge_base:
            return "• Grande casserole inox\n• Thermomètre\n• Moule à fromage\n• Étamine"
        
        materiel_list = self.knowledge_base['materiel_indispensable'].get('Pour débuter', [])
        return '\n'.join([f"• {item}" for item in materiel_list])
    
    def _get_egouttage_time(self, cheese_type):
        """Durée d'égouttage selon le type"""
        if 'frais' in cheese_type.lower():
            return "2-4 heures"
        elif 'molle' in cheese_type.lower():
            return "12-18 heures"
        else:
            return "18-24 heures"
    
    def _get_soins_affinage(self, cheese_type):
        """Instructions de soins pendant l'affinage"""
        if 'frais' in cheese_type.lower():
            return "Pas d'affinage nécessaire, consommer rapidement"
        elif 'molle' in cheese_type.lower():
            return "Retourner tous les 2 jours, brosser si croûte blanche apparaît"
        elif 'pressée' in cheese_type.lower():
            return "Retourner quotidiennement la 1ère semaine, puis 2x/semaine"
        else:
            return "Retourner régulièrement, surveiller l'apparition des moisissures"
    
    def _get_tasting_time(self, cheese_type):
        """Moment optimal de dégustation"""
        type_info = self._get_type_info(cheese_type)
        duree = type_info.get('duree', '')
        
        if 'frais' in cheese_type.lower():
            return "Immédiatement après fabrication ou dans les 3-5 jours"
        elif '2-8 semaines' in duree:
            return "Après 3-6 semaines d'affinage minimum"
        elif 'mois' in duree:
            return "Après la durée d'affinage indiquée, goûter régulièrement"
        else:
            return "Selon votre goût, goûter à différents stades d'affinage"
    
    def _get_variantes(self, cheese_type, ingredients):
        """Suggère des variantes créatives"""
        variantes = ""
        
        if 'epices_et_aromates' in self.knowledge_base:
            variantes += "1. **Version aux herbes** : "
            herbes = self.rng.sample(self.knowledge_base['epices_et_aromates'].get('Herbes fraîches', []),k=3)
            variantes += f"Incorporer {', '.join(herbes[:3][:])}\n\n"
            
            variantes += "2. **Version épicée** : "
            epices = self.knowledge_base['epices_et_aromates'].get('Épices chaudes', [])
            variantes += f"Enrober de {', '.join(epices[:2])}\n\n"
            
            variantes += "3. **Version gourmande** : "
            accomp = self.knowledge_base['epices_et_aromates'].get('Accompagnements dans la pâte', [])
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
    ingredients_str = ' '.join(ingredients).lower()

    # Briques génériques
    base_general = ["Trésor", "Délice", "Nuage", "Essence", "Secret", "Velours"]
    style_general = ["Lacté", "Artisan", "Fondant", "Crémeux", "Rustique"]

    if 'chèvre' in ingredients_str:
        base = ["Chèvre", "Caprice", "Blanc"]
        qualifier = ["des Prés", "Lacté", "Frais"]
    elif 'brebis' in ingredients_str:
        base = ["Brebis", "Douceur", "Trésor"]
        qualifier = ["Pastorale", "de Bergère", "Montagnard"]
    elif 'herbe' in ingredients_str or 'épice' in ingredients_str:
        base = ["Jardin", "Bouquet", "Pré"]
        qualifier = ["Fromager", "Lacté", "Fleuri"]
    elif 'frais' in cheese_type.lower():
        base = ["Blanc", "Nuage", "Fraîcheur"]
        qualifier = ["Matinale", "Lactée", "Pure"]
    elif 'molle' in cheese_type.lower():
        base = ["Velours", "Crème", "Délice"]
        qualifier = ["de Cave", "d'Artisan", "Fondant"]
    elif 'pressée' in cheese_type.lower():
        base = ["Roc", "Meule", "Pierre"]
        qualifier = ["du Terroir", "Tradition", "Lactée"]
    else:
        base = base_general
        qualifier = ["Maison", "Artisanale", "Fromagère"]
        
        return f"{self.rng.choice(base)} {self.rng.choice(qualifier)} {self.rng.choice(style_general)}"
    
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
        
        if 'végétarien' in constraints_lower or 'vegetarien' in constraints_lower:
            note += "✓ **Présure végétale** : Utiliser présure d'origine végétale (chardon, figuier)\n"
            note += "✓ Vérifier que les ferments sont non-animaux\n\n"
        
        if 'rapide' in constraints_lower:
            note += "✓ **Version rapide** : Privilégier fromage frais (4-6h total)\n"
            note += "✓ Utiliser citron pour caillage accéléré (20 min)\n\n"
        
        if 'lactose' in constraints_lower:
            note += "✓ **Sans lactose** : Les fromages affinés contiennent naturellement peu de lactose\n"
            note += "✓ Utiliser lait délactosé ou lait de chèvre (plus digeste)\n\n"
        
        if 'vegan' in constraints_lower or 'végétalien' in constraints_lower:
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
    
    def get_knowledge_summary(self):
        """Retourne un résumé complet de la base de connaissances"""
        summary = "📚 BASE DE CONNAISSANCES FROMAGE COMPLÈTE\n\n"
        
        # Types de pâte
        summary += "🧀 TYPES DE PÂTE :\n"
        summary += "="*70 + "\n\n"
        
        for name, info in self.knowledge_base['types_pate'].items():
            summary += f"• {name.upper()}\n"
            summary += f"  {info['description']}\n"
            summary += f"  Exemples : {info['exemples']}\n"
            summary += f"  Durée : {info['duree']} | Difficulté : {info['difficulte']}\n\n"
        
        # Ingrédients de base
        summary += "\n" + "="*70 + "\n"
        summary += "🥛 INGRÉDIENTS ESSENTIELS :\n"
        summary += "="*70 + "\n\n"
        
        for category, items in self.knowledge_base['ingredients_base'].items():
            summary += f"\n• {category.upper()} :\n"
            for item in items:
                summary += f"  - {item}\n"
        
        # Épices et aromates
        if 'epices_et_aromates' in self.knowledge_base:
            summary += "\n" + "="*70 + "\n"
            summary += "🌶️ ÉPICES ET AROMATES :\n"
            summary += "="*70 + "\n\n"
            
            for category, items in self.knowledge_base['epices_et_aromates'].items():
                summary += f"• {category.upper()} :\n"
                for item in items[:5]:
                    summary += f"  - {item}\n"
                if len(items) > 5:
                    summary += f"  ... et {len(items)-5} autres\n"
                summary += "\n"
        
        # Techniques d'aromatisation
        if 'techniques_aromatisation' in self.knowledge_base:
            summary += "\n" + "="*70 + "\n"
            summary += "🎨 TECHNIQUES D'AROMATISATION :\n"
            summary += "="*70 + "\n\n"
            
            for tech, desc in self.knowledge_base['techniques_aromatisation'].items():
                summary += f"• {tech} :\n  {desc}\n\n"
        
        # Dosages recommandés
        if 'dosages_recommandes' in self.knowledge_base:
            summary += "\n" + "="*70 + "\n"
            summary += "📐 DOSAGES RECOMMANDÉS :\n"
            summary += "="*70 + "\n\n"
            
            for ingredient, dosage in self.knowledge_base['dosages_recommandes'].items():
                summary += f"• {ingredient} : {dosage}\n"
        
        # Associations classiques
        if 'associations_classiques' in self.knowledge_base:
            summary += "\n" + "="*70 + "\n"
            summary += "🎯 ASSOCIATIONS CLASSIQUES :\n"
            summary += "="*70 + "\n\n"
            
            for fromage, assoc in self.knowledge_base['associations_classiques'].items():
                summary += f"• {fromage} : {assoc}\n"
        
        # Températures d'affinage
        if 'temperatures_affinage' in self.knowledge_base:
            summary += "\n" + "="*70 + "\n"
            summary += "🌡️ TEMPÉRATURES D'AFFINAGE :\n"
            summary += "="*70 + "\n\n"
            
            for fromage_type, temp in self.knowledge_base['temperatures_affinage'].items():
                summary += f"• {fromage_type} : {temp}\n"
        
        # Problèmes courants
        if 'problemes_courants' in self.knowledge_base:
            summary += "\n" + "="*70 + "\n"
            summary += "🚨 PROBLÈMES COURANTS ET SOLUTIONS :\n"
            summary += "="*70 + "\n\n"
            
            for probleme, solution in list(self.knowledge_base['problemes_courants'].items())[:8]:
                summary += f"❌ {probleme}\n"
                summary += f"   ✅ {solution}\n\n"
            
            remaining = len(self.knowledge_base['problemes_courants']) - 8
            if remaining > 0:
                summary += f"... et {remaining} autres problèmes documentés\n"
        
        # Conservation
        if 'conservation' in self.knowledge_base:
            summary += "\n" + "="*70 + "\n"
            summary += "📦 CONSERVATION :\n"
            summary += "="*70 + "\n\n"
            
            for fromage_type, duree in self.knowledge_base['conservation'].items():
                summary += f"• {fromage_type} : {duree}\n"
        
        # Accords vins
        if 'accords_vins' in self.knowledge_base:
            summary += "\n" + "="*70 + "\n"
            summary += "🍷 ACCORDS VINS :\n"
            summary += "="*70 + "\n\n"
            
            for fromage_type, vin in list(self.knowledge_base['accords_vins'].items())[:12]:
                summary += f"• {fromage_type} → {vin}\n"
            
            remaining = len(self.knowledge_base['accords_vins']) - 12
            if remaining > 0:
                summary += f"\n... et {remaining} autres accords\n"
        
        # Accords mets
        if 'accords_mets' in self.knowledge_base:
            summary += "\n" + "="*70 + "\n"
            summary += "🍽️ ACCORDS METS :\n"
            summary += "="*70 + "\n\n"
            
            for fromage_type, mets in self.knowledge_base['accords_mets'].items():
                summary += f"• {fromage_type} : {mets}\n"
        
        # Matériel indispensable
        if 'materiel_indispensable' in self.knowledge_base:
            summary += "\n" + "="*70 + "\n"
            summary += "🛠️ MATÉRIEL RECOMMANDÉ :\n"
            summary += "="*70 + "\n\n"
            
            for niveau, items in self.knowledge_base['materiel_indispensable'].items():
                summary += f"\n📌 {niveau.upper()} :\n"
                for item in items:
                    summary += f"  - {item}\n"
        
        # Fournisseurs recommandés
        if 'fournisseurs_recommandes' in self.knowledge_base:
            summary += "\n" + "="*70 + "\n"
            summary += "🏪 FOURNISSEURS RECOMMANDÉS :\n"
            summary += "="*70 + "\n\n"
            
            for cat, fournisseurs in self.knowledge_base['fournisseurs_recommandes'].items():
                summary += f"• {cat} : {fournisseurs}\n"
        
        # Calendrier fromager
        if 'calendrier_fromager' in self.knowledge_base:
            summary += "\n" + "="*70 + "\n"
            summary += "📅 CALENDRIER FROMAGER :\n"
            summary += "="*70 + "\n\n"
            
            for saison, conseil in self.knowledge_base['calendrier_fromager'].items():
                summary += f"• {saison} :\n  {conseil}\n\n"
        
        # Conseils généraux
        summary += "\n" + "="*70 + "\n"
        summary += "💡 CONSEILS GÉNÉRAUX DU MAÎTRE FROMAGER :\n"
        summary += "="*70 + "\n\n"
        summary += "✨ Hygiène irréprochable : stériliser tout le matériel à l'eau bouillante\n"
        summary += "✨ Température précise : ±2°C peut totalement changer le résultat\n"
        summary += "✨ Patience : un bon fromage ne se précipite pas, respecter les temps\n"
        summary += "✨ Qualité du lait : préférer lait cru ou pasteurisé (JAMAIS UHT)\n"
        summary += "✨ Tenir un carnet : noter températures, durées et résultats\n"
        summary += "✨ Commencer simple : fromage frais avant pâtes pressées\n"
        summary += "✨ Cave d'affinage DIY : Une glacière + bol d'eau + hygromètre suffit\n"
        summary += "✨ Le petit-lait est précieux : pain, ricotta, plantes\n\n"
        
        # Statistiques
        summary += "="*70 + "\n"
        summary += "📊 STATISTIQUES DE LA BASE DE CONNAISSANCES :\n"
        summary += "="*70 + "\n"
        summary += f"• Types de pâte documentés : {len(self.knowledge_base.get('types_pate', {}))}\n"
        summary += f"• Catégories d'ingrédients : {len(self.knowledge_base.get('ingredients_base', {}))}\n"
        if 'epices_et_aromates' in self.knowledge_base:
            summary += f"• Catégories d'épices : {len(self.knowledge_base['epices_et_aromates'])}\n"
            total_epices = sum(len(items) for items in self.knowledge_base['epices_et_aromates'].values())
            summary += f"• Total épices/aromates : {total_epices}\n"
        summary += f"• Températures d'affinage : {len(self.knowledge_base.get('temperatures_affinage', {}))}\n"
        summary += f"• Problèmes documentés : {len(self.knowledge_base.get('problemes_courants', {}))}\n"
        summary += f"• Infos conservation : {len(self.knowledge_base.get('conservation', {}))}\n"
        summary += f"• Accords vins : {len(self.knowledge_base.get('accords_vins', {}))}\n"
        summary += f"• Accords mets : {len(self.knowledge_base.get('accords_mets', {}))}\n"
        summary += f"• Techniques d'aromatisation : {len(self.knowledge_base.get('techniques_aromatisation', {}))}\n"
        summary += "\n🎉 Base de connaissances très complète pour devenir maître fromager !\n"
        
        return summary


# Initialiser l'agent
agent = AgentFromagerHF()

def create_interface():
    """Crée l'interface Gradio"""
    
    with gr.Blocks(title="🧀 Agent Fromager") as demo:
        
        gr.Markdown("""
        # 🧀 Agent Fromager Intelligent
        ### Créez vos fromages artisanaux avec l'IA
        
        Entrez vos ingrédients et laissez l'intelligence artificielle vous guider pas à pas.
        """)
        
        with gr.Tabs():
            # TAB 1 : Création de recette
            with gr.Tab("🎨 Créer une recette"):
                with gr.Row():
                    with gr.Column(scale=2):
                        ingredients_input = gr.Textbox(
                            label="🥛 Ingrédients disponibles",
                            placeholder="Ex: lait de chèvre, présure, sel de mer, herbes de Provence",
                            lines=3,
                            info="Séparez les ingrédients par des virgules"
                        )
                        
                        cheese_type_input = gr.Dropdown(
                            choices=[
                                "Laissez l'IA choisir",
                                "Fromage frais",
                                "Pâte molle",
                                "Pâte pressée non cuite",
                                "Pâte pressée cuite",
                                "Pâte persillée"
                            ],
                            label="🧀 Type de fromage souhaité",
                            value="Laissez l'IA choisir"
                        )
                        
                        constraints_input = gr.Textbox(
                            label="⚙️ Contraintes (optionnel)",
                            placeholder="Ex: végétarien, rapide, sans lactose...",
                            lines=2
                        )
                        
                        generate_btn = gr.Button(
                            "✨ Générer la recette",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        ### 💡 Conseils
                        
                        **Ingrédients minimums :**
                        - Lait (vache, chèvre, brebis...)
                        - Coagulant (présure ou citron)
                        - Sel
                        
                        **Types recommandés :**
                        - 🟢 Facile : Fromage frais
                        - 🟡 Moyen : Pâte molle
                        - 🔴 Difficile : Pâte persillée
                        """)
                
                recipe_output = gr.Textbox(
                    label="📖 Votre recette complète",
                    lines=30,
                    max_lines=50
                )
                
                generate_btn.click(
                    fn=agent.generate_recipe,
                    inputs=[ingredients_input, cheese_type_input, constraints_input],
                    outputs=recipe_output
                )
            
            # TAB 2 : Base de connaissances
            with gr.Tab("📚 Base de connaissances"):
                knowledge_output = gr.Textbox(
                    label="Documentation fromage",
                    value=agent.get_knowledge_summary(),
                    lines=40,
                    max_lines=60
                )
            
            # TAB 3 : Historique
            with gr.Tab("🕒 Historique"):
                gr.Markdown("### 📚 Vos recettes sauvegardées")
                gr.Markdown("💾 Persistance garantie avec Hugging Face Datasets")
                
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Actualiser", variant="secondary")
                    sync_btn = gr.Button("☁️ Synchroniser depuis HF", variant="secondary")
                    clear_btn = gr.Button("🗑️ Effacer tout", variant="stop")
                
                history_display = gr.Textbox(
                    label="",
                    value=agent.get_history_display(),
                    lines=30,
                    max_lines=50
                )
                
                gr.Markdown("---")
                
                with gr.Row():
                    recipe_id_input = gr.Number(
                        label="🔍 Numéro de la recette",
                        value=1,
                        precision=0
                    )
                    load_recipe_btn = gr.Button("📖 Charger la recette", variant="primary")
                
                loaded_recipe = gr.Textbox(
                    label="📖 Recette complète",
                    lines=30,
                    max_lines=50
                )
                
                refresh_btn.click(
                    fn=agent.get_history_display,
                    outputs=history_display
                )
                
                sync_btn.click(
                    fn=agent.sync_from_hf,
                    outputs=history_display
                )
                
                clear_btn.click(
                    fn=agent.clear_history,
                    outputs=history_display
                )
                
                load_recipe_btn.click(
                    fn=agent.get_recipe_by_id,
                    inputs=recipe_id_input,
                    outputs=loaded_recipe
                )
            
            # TAB 4 : À propos
            with gr.Tab("ℹ️ À propos"):
                gr.Markdown("""
                ## 🧀 Agent Fromager Intelligent
                
                ### Créé par Myriam avec ❤️
                
                **Fonctionnalités :**
                - ✅ Recettes détaillées étape par étape
                - ✅ Base de connaissances fromagère
                - ✅ Historique persistant avec HF Datasets
                - ✅ Adaptation aux contraintes
                
                **Version :** 2.0  
                **Dernière mise à jour :** Février 2025
                
                ---
                
                💬 **Feedback ?** N'hésitez pas à laisser un commentaire !
                """)
        
        gr.Markdown("""
        ---
        <center>
        Fait avec 🧀 et 🤖 | Hugging Face Spaces | 2025
        </center>
        """)
    
    return demo

if __name__ == "__main__":
    interface = create_interface()
    interface.launch()