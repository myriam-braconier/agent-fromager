import random
import gradio as gr
import json
import os
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download
import pandas as pd

# ✅ AJOUTE ÇA ICI (ligne ~10)
pd.set_option('future.no_silent_downcasting', True)

class AgentFromagerHF:
    """Agent fromager avec persistance HF Dataset"""
    
    def __init__(self):
        self.rng = random.Random()
        self.knowledge_base = self._init_knowledge()
        self.recipes_file = 'recipes_history.json'
        self.hf_repo = "volubyl/fromager-recipes"
        self.hf_token = os.environ.get("HF_TOKEN")
        self.api = HfApi(token=self.hf_token) if self.hf_token else None
        
        print(f"🔍 HF_TOKEN détecté : {'✅ OUI' if os.environ.get('HF_TOKEN') else '❌ NON'}")
        print(f"🔍 Repo cible : {self.hf_repo}")
        print(f"🔍 API initialisée : {'✅ OUI' if self.api else '❌ NON'}")
        
        # Charger l'historique depuis HF au démarrage
        self._download_history_from_hf()
        
         # ✅ AJOUTER CETTE LIGNE
        self.history = self._load_history()  # Charger l'historique en mémoire
    
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
        'regles_compatibilite': {
            'lait_x_type_pate': {
                'description': 'Associations valides entre types de lait et types de pâte',
                'combinaisons_valides': [
                    {
                        'lait': 'vache',
                        'types_pate_compatibles': ['Fromage frais', 'Pâte molle', 'Pâte pressée non cuite', 
                                                   'Pâte pressée cuite', 'Pâte persillée'],
                        'exemples': ['camembert', 'brie', 'comté', 'roquefort']
                    },
                    {
                        'lait': 'chevre',
                        'types_pate_compatibles': ['Fromage frais', 'Pâte pressée non cuite'],
                        'types_pate_incompatibles': ['Pâte molle'],
                        'raison': 'Le lait de chèvre donne naturellement une croûte cendrée/naturelle, pas de croûte fleurie',
                        'exemples': ['crottin de Chavignol', 'sainte-maure', 'tomme de chèvre']
                    },
                    {
                        'lait': 'brebis',
                        'types_pate_compatibles': ['Fromage frais', 'Pâte pressée non cuite', 'Pâte pressée cuite', 'Pâte persillée'],
                        'types_pate_incompatibles': ['Pâte molle'],
                        'raison': 'La brebis est traditionnellement utilisée pour fromages pressés ou bleus, pas pour croûtes fleuries',
                        'exemples': ['roquefort', 'ossau-iraty', 'manchego', 'pecorino']
                    },
                    {
                        'lait': 'bufflonne',
                        'types_pate_compatibles': ['Fromage frais'],
                        'types_pate_incompatibles': ['Pâte molle', 'Pâte pressée cuite'],
                        'raison': 'Lait très riche utilisé principalement pour fromages frais italiens',
                        'exemples': ['mozzarella di bufala', 'burrata']
                    }
                ]
            },
            
            'lait_x_aromates': {
                'description': 'Associations classiques et harmonieuses',
                'affinites': [
                    {
                        'lait': 'chevre',
                        'aromates_recommandes': ['herbes de Provence', 'miel', 'lavande', 'thym', 'cendre'],
                        'aromates_deconseilles': ['curry fort', 'cumin intense'],
                        'raison': 'Le chèvre a un goût délicat qui peut être écrasé par épices trop fortes'
                    },
                    {
                        'lait': 'brebis',
                        'aromates_recommandes': ['piment d\'Espelette', 'romarin', 'olives', 'tomates séchées'],
                        'aromates_deconseilles': [],
                        'raison': 'Goût prononcé de brebis supporte bien épices méditerranéennes fortes'
                    },
                    {
                        'lait': 'vache',
                        'aromates_recommandes': ['ail', 'fines herbes', 'poivre', 'noix', 'cumin'],
                        'aromates_deconseilles': [],
                        'raison': 'Neutre, s\'accommode de presque tout'
                    }
                ]
            },
            
            'type_pate_x_aromates': {
                'Fromage frais': {
                    'aromates_compatibles': ['herbes fraîches', 'ail frais', 'ciboulette', 'aneth', 'menthe'],
                    'aromates_incompatibles': ['épices chaudes fortes', 'curry', 'piment de Cayenne'],
                    'raison': 'Goût délicat, consommation rapide : herbes fraîches idéales'
                },
                'Pâte molle': {
                    'aromates_compatibles': ['herbes séchées', 'poivre', 'ail confit'],
                    'aromates_incompatibles': ['herbes fraîches'],
                    'raison': 'Affinage humide : herbes fraîches peuvent pourrir, préférer séchées'
                },
                'Pâte pressée non cuite': {
                    'aromates_compatibles': ['cumin', 'fenugrec', 'noix', 'fruits secs', 'épices en grains'],
                    'aromates_incompatibles': ['herbes fraîches délicates'],
                    'raison': 'Longue conservation : épices robustes et séchées résistent mieux'
                },
                'Pâte pressée cuite': {
                    'aromates_compatibles': ['cumin', 'noix', 'fruits secs'],
                    'aromates_incompatibles': ['herbes fraîches'],
                    'raison': 'Très long affinage : seules épices robustes survivent'
                },
                'Pâte persillée': {
                    'aromates_compatibles': ['noix', 'miel', 'fruits secs'],
                    'aromates_incompatibles': ['herbes fortes', 'épices puissantes'],
                    'raison': 'Goût déjà très prononcé : accompagnements doux uniquement'
                }
            },
            
            'exclusions_absolues': [
                {
                    'combinaison': 'lait:brebis + type_pate:Pâte molle',
                    'raison': 'Incompatibilité traditionnelle et technique. La brebis ne développe pas bien le Penicillium camemberti',
                    'severite': 'haute',
                    'alternatives': ['Pâte pressée non cuite', 'Pâte persillée']
                },
                {
                    'combinaison': 'lait:chevre + type_pate:Pâte molle',
                    'raison': 'Chèvre développe naturellement croûte cendrée, pas fleurie comme camembert',
                    'severite': 'haute',
                    'alternatives': ['Fromage frais', 'Pâte pressée non cuite']
                },
                {
                    'combinaison': 'type_pate:Fromage frais + aromate:herbes séchées fortes',
                    'raison': 'Déséquilibre gustatif - fromage frais trop délicat',
                    'severite': 'moyenne',
                    'alternatives': ['Herbes fraîches', 'herbes séchées douces']
                },
                {
                    'combinaison': 'affinage:long + aromate:herbes fraîches',
                    'raison': 'Risque sanitaire - les herbes fraîches moisissent pendant affinage humide',
                    'severite': 'haute',
                    'alternatives': ['Herbes séchées', 'aromates après affinage']
                }
            ]
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
            with open(self.recipes_file, 'w', encoding='utf-8') as f:json.dump([], f)

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

    def get_history(self):
        """Retourne l'historique complet"""
        try:
            if os.path.exists(self.recipes_file):
                with open(self.recipes_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"❌ Erreur get_history: {e}")
            return []

    def _save_to_history(self, ingredients, cheese_type, constraints, recipe):
        """Sauvegarde une recette dans l'historique"""
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

            # Sauvegarder localement
            with open(self.recipes_file, 'w', encoding='utf-8') as f:
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

                ing = entry['ingredients']
                if isinstance(ing, list):
                    ing = ', '.join(str(i) for i in ing)  # ✅ CORRECT !
                elif isinstance(ing, str):
                    ing = ing[:50]  # Limite si déjà string

                display += f"🥛 Ingrédients: {ing[:50]}...\n"

                if entry.get('constraints'):
                    display += f"⚙️ Contraintes: {entry['constraints']}\n"

                display += "\n---\n\n"

            return display

        except Exception as e:
            return f"❌ Erreur lecture historique: {e}"

    def clear_history(self):
        """Efface tout l'historique"""
        try:
            with open(self.recipes_file, 'w', encoding='utf-8') as f:
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
                 
    # vérification connexion internet dans ta classe AgentFromagerHF
    def test_internet(self):
        """Test si Internet fonctionne"""
        try:
            import requests
            response = requests.get("https://httpbin.org/get", timeout=10)
            return f"✅ Internet fonctionne !\n\nStatus: {response.status_code}\nURL testée: https://httpbin.org/get"
        except Exception as  e:
            return f"❌ Erreur d'accès Internet:\n{str(e)}"
    def search_web_recipes(self, ingredients: str, cheese_type: str, max_results: int = 6) -> list:
        """Scrape le web pour trouver des recettes de fromage - VERSION AMÉLIORÉE"""
    
        recipes = []
        
        try:
            from duckduckgo_search import DDGS
            
            # ===== 1. CONSTRUIRE DES REQUÊTES MULTIPLES =====
            # Au lieu d'une seule requête, en faire plusieurs pour plus de résultats
            
            ingredients_clean = ingredients.replace(',', ' ')
            
            queries = []
            
            # Requête principale
            if cheese_type and cheese_type != "Laissez l'IA choisir":
                queries.append(f"recette {cheese_type} {ingredients_clean}")
                queries.append(f"fabrication {cheese_type} maison")
            else:
                queries.append(f"recette fromage {ingredients_clean}")
            
            # Requêtes par ingrédient principal
            main_ingredients = [ing.strip() for ing in ingredients.split(',')[:2]]  # 2 premiers
            for ing in main_ingredients:
                if ing and len(ing) > 3:
                    queries.append(f"fromage {ing} recette")
            
            # Requête artisanale
            queries.append(f"fromage artisanal maison {ingredients_clean}")
            
            print(f"🔍 Recherche avec {len(queries)} requêtes différentes")
            
            # ===== 2. RECHERCHE MULTIPLE AVEC DUCKDUCKGO =====
            ddg = DDGS()
            seen_urls = set()  # Éviter les doublons
            seen_domains = set()
            
            for query in queries[:3]:  # Limiter à 3 requêtes pour ne pas spammer
                try:
                    print(f"   → Recherche : {query}")
                    
                    search_results = ddg.text(
                        keywords=query,
                        region='fr-fr',
                        safesearch='off',
                        max_results=10  # Plus de résultats par requête
                    )
                    
                    for result in search_results:
                        url = result.get('href') or result.get('link', '')
                        title = result.get('title', 'Sans titre')
                        description = result.get('body', '') or result.get('description', '')
                        
                        if not url or url in seen_urls:
                            continue
                        
                        # Extraire le domaine
                        domain = self._extract_domain(url)
                        
                        # ===== 3. FILTRAGE INTELLIGENT =====
                        
                        # Sites de recettes prioritaires (score élevé)
                        priority_sites = [
                            'marmiton', '750g', 'cuisineaz', 'ricardocuisine',
                            'ptitchef', 'cuisine-facile', 'chefsimon', 'hervecuisine',
                            'lasantedanslassiette', 'supertoinette', 'auxdelicesdupalais'
                        ]
                        
                        # Sites fromagers spécialisés (score très élevé)
                        cheese_sites = [
                            'fromage', 'fromagerie', 'laiterie', 'fermier',
                            'artisan', 'cheese', 'dairy'
                        ]
                        
                        # Sites à éviter
                        blocked_sites = [
                            'youtube', 'pinterest', 'instagram', 'facebook',
                            'amazon', 'ebay', 'shopping', 'pub', 'ad'
                        ]
                        
                        # Vérifier si le site est bloqué
                        if any(blocked in url.lower() or blocked in domain.lower() 
                            for blocked in blocked_sites):
                            continue
                        
                        # Vérifier pertinence du contenu
                        content_lower = (title + ' ' + description).lower()
                        
                        # Mots-clés fromagers obligatoires
                        cheese_keywords = ['fromage', 'cheese', 'lait', 'caillé', 'présure', 'affinage']
                        has_cheese_keyword = any(kw in content_lower for kw in cheese_keywords)
                        
                        if not has_cheese_keyword:
                            continue
                        
                        # ===== 4. SCORING DES RÉSULTATS =====
                        score = 0
                        
                        # Bonus pour sites prioritaires
                        if any(site in domain.lower() or site in url.lower() 
                            for site in priority_sites):
                            score += 10
                        
                        # Bonus énorme pour sites fromagers
                        if any(site in domain.lower() or site in url.lower() 
                            for site in cheese_sites):
                            score += 20
                        
                        # Bonus pour type de fromage dans le titre
                        if cheese_type and cheese_type.lower() in title.lower():
                            score += 15
                        
                        # Bonus pour ingrédients dans le titre
                        for ing in main_ingredients:
                            if ing.lower() in title.lower():
                                score += 5
                        
                        # Bonus pour mots-clés "maison", "artisan", "facile"
                        if any(kw in content_lower for kw in ['maison', 'artisan', 'facile', 'diy']):
                            score += 5
                        
                        # Éviter trop de résultats du même domaine
                        if domain in seen_domains:
                            score -= 10
                        
                        # ===== 5. AJOUTER SI SCORE SUFFISANT =====
                        if score >= 5:  # Seuil minimal
                            recipes.append({
                                'title': title,
                                'url': url,
                                'description': self._clean_description(description),
                                'source': domain,
                                'score': score  # Pour trier par pertinence
                            })
                            
                            seen_urls.add(url)
                            seen_domains.add(domain)
                            
                            print(f"   ✓ Ajouté : {title[:50]}... (score: {score})")
                        
                        # Arrêter si on a assez de résultats
                        if len(recipes) >= max_results * 2:
                            break
                    
                except Exception as e:
                    print(f"   ⚠️ Erreur sur requête '{query}': {e}")
                    continue
            
            # ===== 6. TRIER PAR SCORE ET LIMITER =====
            recipes.sort(key=lambda x: x['score'], reverse=True)
            recipes = recipes[:max_results]
            
            print(f"✅ {len(recipes)} recettes trouvées (sur {len(seen_urls)} résultats)")
            
            return recipes
        
        except Exception as e:
            print(f"❌ Erreur recherche web globale: {e}")
            import traceback
            traceback.print_exc()
            return []


    def _clean_description(self, description: str) -> str:
        """Nettoie et formate la description"""
        # Limiter la longueur
        if len(description) > 280:
            description = description[:280] + "..."
        
        # Supprimer les caractères bizarres
        description = description.replace('\n', ' ').replace('\r', ' ')
        description = ' '.join(description.split())  # Nettoyer espaces multiples
        
        return description

    def _extract_domain(self, url: str) -> str:
        """Extrait le nom de domaine d'une URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc
            # Retirer 'www.' et garder le domaine principal
            domain = domain.replace('www.', '')
            return domain
        except:
            return "web"

    # =====  MÉTHODE de validationICI =====
    def _validate_combination(self, lait: str, type_pate: str, aromates: list = None) -> tuple:
        """
        Valide une combinaison lait/pâte/aromates
        Returns: (bool, str) - (est_valide, raison)
        """
        rules = self.knowledge['regles_compatibilite']
        
        # Vérifier les exclusions absolues
        for exclusion in rules['exclusions_absolues']:
            combo = exclusion['combinaison']
            if f'lait:{lait}' in combo and f'type_pate:{type_pate}' in combo:
                alternatives = ', '.join(exclusion.get('alternatives', []))
                message = f"❌ {exclusion['raison']}\n\nAlternatives suggérées : {alternatives}"
                return False, message
        
        # Vérifier compatibilité lait/pâte
        for combo in rules['lait_x_type_pate']['combinaisons_valides']:
            if combo['lait'] == lait.lower():
                if type_pate in combo.get('types_pate_incompatibles', []):
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
    
    def _extract_lait_from_text(self, text: str) -> str:
        """Extrait le type de lait d'un texte"""
        if not text:
            return None
        
        text_lower = text.lower()
        
        lait_patterns = {
            'vache': ['vache', 'bovin', 'cow', 'lait de vache'],
            'chevre': ['chèvre', 'chevre', 'caprin', 'goat', 'lait de chèvre', 'lait de chevre'],
            'brebis': ['brebis', 'mouton', 'ovin', 'sheep', 'lait de brebis'],
            'bufflonne': ['bufflonne', 'buffle', 'buffalo', 'lait de bufflonne']
        }
        
        # Priorité aux patterns les plus spécifiques
        for lait_type, patterns in lait_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return lait_type
        
        return None
    
    def _validate_combination(self, lait: str, type_pate: str) -> tuple:
        """
        Valide une combinaison lait/pâte
        Returns: (bool, str) - (est_valide, message)
        """
        if not lait or not type_pate:
            return True, "OK"
        
        rules = self.knowledge_base['regles_compatibilite']
        lait_lower = lait.lower()
        
        # Vérifier les exclusions absolues
        for exclusion in rules['exclusions_absolues']:
            combo = exclusion['combinaison']
            if f'lait:{lait_lower}' in combo and f'type_pate:{type_pate}' in combo:
                alternatives = ', '.join(exclusion.get('alternatives', []))
                message = f"{exclusion['raison']}\n\n**Alternatives :** {alternatives}"
                return False, message
        
        # Vérifier compatibilité lait/pâte
        for combo in rules['lait_x_type_pate']['combinaisons_valides']:
            if combo['lait'] == lait_lower:
                if type_pate in combo.get('types_pate_incompatibles', []):
                    compatible = ', '.join(combo['types_pate_compatibles'])
                    message = f"{combo['raison']}\n\n**Types compatibles avec le lait de {lait} :** {compatible}"
                    return False, message
        
        return True, "✅ Combinaison valide"
    
    def _suggest_alternatives(self, lait: str, type_pate: str) -> str:
        """Suggère des alternatives compatibles"""
        rules = self.knowledge_base['regles_compatibilite']
        
        # Trouver les types compatibles pour ce lait
        for combo in rules['lait_x_type_pate']['combinaisons_valides']:
            if combo['lait'] == lait.lower():
                compatibles = combo['types_pate_compatibles']
                exemples = combo.get('exemples', [])
                
                result = f"**Pour du lait de {lait}, voici les types compatibles :**\n\n"
                for i, pate in enumerate(compatibles, 1):
                    result += f"{i}. {pate}\n"
                
                if exemples:
                    result += f"\n**Exemples :** {', '.join(exemples)}"
                
                return result
        
        return "Veuillez choisir une autre combinaison lait/type de pâte."
    
    def generate_recipe(self, ingredients, cheese_type, constraints):
        """Génère une recette de fromage détaillée avec validation"""
        
        # Validation des ingrédients
        valid, message = self.validate_ingredients(ingredients)
        if not valid:
            return message
        
        ingredients_list = [ing.strip() for ing in ingredients.split(',')]
        cheese_type_clean = cheese_type if cheese_type != "Laissez l'IA choisir" else "Fromage artisanal"
        
        # ===== VALIDATION DE LA COMPATIBILITÉ LAIT/PÂTE =====
        lait = self._extract_lait_from_text(ingredients)
        
        # Si un type de pâte spécifique est choisi, valider la compatibilité
        if lait and cheese_type_clean != "Fromage artisanal":
            is_valid, reason = self._validate_combination(lait, cheese_type_clean)
            if not is_valid:
                alternatives = self._suggest_alternatives(lait, cheese_type_clean)
                return f"**❌ Combinaison invalide détectée**\n\n{reason}\n\n**💡 Alternatives suggérées :**\n{alternatives}\n\nModifiez votre type de fromage pour continuer."
        
        # Générer la recette
        recipe = self._generate_detailed_recipe(ingredients_list, cheese_type_clean, constraints)
        
        # Sauvegarder dans l'historique
        self._save_to_history(ingredients_list, cheese_type_clean, constraints, recipe)
        
        return recipe
    
    def _generate_detailed_recipe(self, ingredients, cheese_type, constraints):
        """Génère une recette enrichie avec la base de connaissances"""
        
         # ===== DOUBLE VALIDATION POST-DÉTERMINATION =====
        # Extraire le lait des ingrédients
        ingredients_str = ' '.join(ingredients).lower()
        lait = self._extract_lait_from_text(ingredients_str)
        
        # Valider la combinaison finale
        if lait and cheese_type:
            is_valid, reason = self._validate_combination(lait, cheese_type)
            if not is_valid:
                # Forcer un type compatible
                rules = self.knowledge_base['regles_compatibilite']
                for combo in rules['lait_x_type_pate']['combinaisons_valides']:
                    if combo['lait'] == lait.lower():
                        compatibles = combo['types_pate_compatibles']
                        if compatibles:
                            cheese_type = compatibles[0]  # Utiliser le premier compatible
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
   
    def generate_recipe_creative(self, ingredients, cheese_type, constraints, 
                            creativity_level, texture_preference, 
                            affinage_duration, spice_intensity):
        """Génère une recette avec mode créatif et micro-choix"""
    
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
            cheese_type_clean = cheese_type if cheese_type != "Laissez l'IA choisir" else "Fromage artisanal"
        
            # Validation compatibilité lait/pâte
            lait = self._extract_lait_from_text(ingredients)
            if lait and cheese_type_clean != "Fromage artisanal":
                is_valid, reason = self._validate_combination(lait, cheese_type_clean)
                if not is_valid:
                    alternatives = self._suggest_alternatives(lait, cheese_type_clean)
                    return f"❌ Combinaison invalide\n\n{reason}\n\n{alternatives}"
        
            # ===== APPLIQUER LES MICRO-CHOIX =====
            # Adapter selon les préférences
            modified_ingredients = self._apply_micro_choices(
                ingredients_list, 
                texture_preference,
                spice_intensity,
                affinage_duration
            )
        
            # Générer recette de base
            recipe = self._generate_detailed_recipe(
                modified_ingredients, 
                cheese_type_clean, 
                constraints
            )
        
            # ===== MODE CRÉATIF =====
            if creativity_level > 0:
                recipe = self._add_creative_variations(
                    recipe, 
                    creativity_level,
                    cheese_type_clean,
                    lait
                )
        
            # Sauvegarder
            self._save_to_history(modified_ingredients, cheese_type_clean, constraints, recipe)
        
            return recipe
        
        except Exception as e:
            error_msg = f"❌ Erreur lors de la génération de la recette : {str(e)}"
            print(error_msg)
        
         # Retourner une recette de secours simple
            try:
                cheese_type_clean = cheese_type if cheese_type != "Laissez l'IA choisir" else "Fromage artisanal"
                return self._create_simple_fallback_recipe(ingredients, cheese_type_clean)
            except:
                return f"{error_msg}\n\nImpossible de générer une recette de secours."
        
    def _apply_micro_choices(self, ingredients, texture, spice_intensity, affinage):
        """Applique les micro-choix aux ingrédients"""
        modified = ingredients.copy()
    
        #  Texture : ajuster ferments/présure
        if texture == "Très crémeux":
            modified.append("crème fraîche (30ml)")
        elif texture == "Très ferme":
            modified.append("présure supplémentaire (+20%)")
    
        # Épices : ajouter selon intensité
        if spice_intensity == "Intense":
            spices = self.rng.choice([
                "poivre noir concassé (2 c.à.c)",
                "piment d'Espelette (1 c.à.c)",
                "ail confit (3 gousses)"
            ])
            modified.append(spices)
        
        elif spice_intensity == "Modéré":
            spices = self.rng.choice([
                "herbes de Provence (1 c.à.s)",
                "thym séché (1 c.à.c)",
                "basilic frais (quelques feuilles)"
            ])
            modified.append(spices)
    
        return modified

    def _add_creative_variations(self, recipe, creativity_level, cheese_type, lait):
        """Ajoute des variations créatives selon le niveau"""
    
        creative_section = "\n\n" + "="*70 + "\n"
        creative_section += "🎨 VARIATIONS CRÉATIVES\n"
        creative_section += "="*70 + "\n\n"
    
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
            creative_section += f"### Variation {i} : {var.get('title', 'Variation créative')}\n\n"
            creative_section += f"**Concept :** {var.get('concept', 'Création originale')}\n\n"
        
        # Ingrédients
        ingredients = var.get('ingredients', [])
        if ingredients:
            creative_section += f"**Ingrédients supplémentaires :**\n"
            for ing in ingredients:
                creative_section += f"- {ing}\n"
            creative_section += "\n"
        
        # Technique - AVEC .get() pour éviter l'erreur
        technique = var.get('technique', 'Incorporer selon votre méthode habituelle')
        creative_section += f"**Technique :** {technique}\n\n"
        creative_section += "---\n\n"
    
        return recipe + creative_section

    def _get_simple_variation(self, cheese_type, lait):
        """Variation simple : herbes et épices"""
    
        variations = {
            'Fromage frais': {
                'title': 'Fromage frais aux fleurs',
                'concept': 'Ajout de fleurs comestibles pour un fromage élégant',
                'ingredients': ['Pétales de rose séchés', 'Lavande culinaire', 'Bleuet'],
                'technique': 'Incorporer les fleurs lors du moulage, parsemer sur le dessus'
        },
            'Pâte molle': {
                'title': 'Pâte molle truffée',
                'concept': 'Infusion de truffe pour un fromage luxueux',
                'ingredients': ['Huile de truffe (5ml)', 'Copeaux de truffe'],
                'technique': 'Badigeonner la croûte avec l\'huile de truffe pendant l\'affinage'
        },
            'Pâte pressée non cuite': {
                'title': 'Tomme aux noix et miel',
                'concept': 'Enrobage sucré-salé original',
                'ingredients': ['Noix concassées', 'Miel de montagne', 'Thym'],
            '   technique': 'Enrober le fromage de noix et miel avant l\'affinage final'
        },
        'Pâte pressée cuite': {
            'title': 'Comté aux herbes de montagne',
            'concept': 'Fromage alpin aromatisé',
            'ingredients': ['Génépi', 'Fleurs de foin', 'Ail des ours'],
            'technique': 'Affiner sur une litière d\'herbes séchées'
        },
        'Pâte persillée': {
            'title': 'Bleu au miel et noix',
            'concept': 'Association sucrée-salée gourmande',
            'ingredients': ['Miel de châtaignier', 'Noix fraîches'],
            'technique': 'Servir avec un filet de miel et des noix concassées'
        }
    }
      # Variation par défaut si type non trouvé
        default = {
        'title': 'Variation classique',
        'concept': 'Fromage aromatisé aux herbes',
        'ingredients': ['Herbes de Provence', 'Ail séché'],
        'technique': 'Mélanger les herbes dans le caillé avant moulage'
        }
    
    
        return variations.get(cheese_type, variations['Fromage frais'])

    def _get_fusion_variation(self, cheese_type, lait):
        """Variation fusion : inspiration internationale"""
    
        fusions = [
            {
                'title': 'Inspiration méditerranéenne',
                'concept': 'Fromage aux saveurs du sud',
                'ingredients': ['Tomates séchées', 'Olives noires', 'Origan', 'Huile d\'olive'],
                'technique': 'Incorporer dans le caillé avant moulage'
            },
            {
                'title': 'Inspiration japonaise',
                'concept': 'Fromage au yuzu et sésame noir',
                'ingredients': ['Zeste de yuzu', 'Graines de sésame noir', 'Algue nori émincée'],
                'technique': 'Enrober le fromage de sésame et ajouter le yuzu en surface'
            },
            {
                'title': 'Inspiration indienne',
                'concept': 'Fromage aux épices chaudes',
                'ingredients': ['Curry doux', 'Gingembre frais râpé', 'Coriandre', 'Curcuma'],
                'technique': 'Mélanger les épices au sel de salage'
            },
            {
                'title': 'Inspiration mexicaine',
                'concept': 'Fromage piquant et fumé',
                'ingredients': ['Piment chipotle', 'Coriandre fraîche', 'Lime'],
                'technique': 'Incorporer le piment fumé dans le caillé'
            }
        ]   
    
        return self.rng.choice(fusions)

    def _get_experimental_variation(self, cheese_type, lait):
        """Variation expérimentale : très créatif"""
    
        experiments = [
        {
            'title': 'Fromage lacto-fermenté aux légumes',
            'concept': 'Double fermentation avec légumes crus',
            'ingredients': ['Carottes râpées', 'Betterave', 'Gingembre', 'Kombucha'],
            'technique': 'Ajouter les légumes lacto-fermentés pendant l\'égouttage'
        },
        {
            'title': 'Fromage aux algues et spiruline',
            'concept': 'Superfood fromager, riche en protéines',
            'ingredients': ['Spiruline en poudre', 'Wakame', 'Graines de chia'],
            'technique': 'Mélanger dans le lait avant caillage pour couleur verte'
        },
        {
            'title': 'Fromage au café et cacao',
            'concept': 'Dessert fromager original',
            'ingredients': ['Café espresso', 'Poudre de cacao', 'Vanille', 'Miel'],
            'technique': 'Infuser le lait avec café/cacao avant emprésurage'
        },
        {
            'title': 'Fromage fumé aux bois exotiques',
            'concept': 'Fumage à froid avec bois spéciaux',
            'ingredients': ['Copeaux de hêtre', 'Copeaux de pommier', 'Romarin séché'],
            'technique': 'Fumer à froid pendant 2-3 heures après séchage'
        },
        {
            'title': 'Fromage au thé matcha',
            'concept': 'Fusion franco-japonaise délicate',
            'ingredients': ['Thé matcha premium', 'Gingembre confit', 'Sésame blanc'],
            'technique': 'Infuser le lait avec matcha, parsemer de sésame'
        }
    ]
    
        return self.rng.choice(experiments)   
    
    def _determine_type(self, ingredients):
        """Détermine le type selon les ingrédients en respectant les compatibilités"""
        ingredients_str = ' '.join(ingredients).lower()
        
        # Extraire le type de lait
        lait = self._extract_lait_from_text(ingredients_str)
        
        # Détecter des indices sur le type souhaité
        if 'citron' in ingredients_str or 'vinaigre' in ingredients_str:
            return "Fromage frais"
        elif 'bleu' in ingredients_str or 'roquefort' in ingredients_str:
            return "Pâte persillée"
        
        # Sinon, choisir un type compatible avec le lait détecté
        if lait:
            rules = self.knowledge_base['regles_compatibilite']
            for combo in rules['lait_x_type_pate']['combinaisons_valides']:
                if combo['lait'] == lait.lower():
                    compatibles = combo['types_pate_compatibles']
                    
                    # Logique de choix selon les ingrédients
                    if any(x in ingredients_str for x in ['herbe', 'épice', 'aromate']):
                        # Si aromates : privilégier fromage frais ou pressée non cuite
                        if 'Fromage frais' in compatibles:
                            return "Fromage frais"
                        elif 'Pâte pressée non cuite' in compatibles:
                            return "Pâte pressée non cuite"
                    
                    # Par défaut : choisir le premier type compatible (généralement le plus simple)
                    if compatibles:
                        return compatibles[0]
        
        # Si pas de lait détecté, fromage frais par défaut (le plus simple et universel)
        return "Fromage frais"
    
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
        base = ["Velours", "Délice", "Nuage", "Trésor", "Secret", "Essence"]
        lieu = ["de Cave", "du Terroir", "des Prés", "Lacté", "Artisan"]
        style = ["Fondant", "Rustique", "Crémeux", "Affiné", "Doux"]

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
            base = base
            qualifier = ["Maison", "Artisanale", "Fromagère"]

        return f"{self.rng.choice(base)} {self.rng.choice(lieu)} {self.rng.choice(style)}"

    
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
    

# Initialiser l'agent
agent = AgentFromagerHF()

# CREATE INTERFACE GRADIO
def create_interface():
    """Interface avec génération simultanée"""
    
    import gradio as gr  # ✅ AJOUTER CET IMPORT ICI
    import json
    import os
    
    fromage_theme = gr.themes.Soft(
        primary_hue="amber",
        secondary_hue="orange",
        neutral_hue="stone"
    )
    
    custom_css = """
    ... (ton CSS)
    """
    
    with gr.Blocks(title="🧀 Agent Fromager") as demo:
        
        gr.Markdown("""
        # 🧀 Agent Fromager Intelligent
        ### Créez vos fromages avec l'IA + Recherche web automatique
        """)
        
        # ===== ZONE DE SAISIE COMMUNE EN HAUT =====
        with gr.Row():
            with gr.Column(scale=2):
                ingredients_input = gr.Textbox(
                    label="🥛 Ingrédients disponibles",
                    placeholder="Ex: lait de chèvre, présure, sel, herbes",
                    lines=3
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
                    label="🧀 Type de fromage",
                    value="Laissez l'IA choisir"
                )
                
                constraints_input = gr.Textbox(
                    label="⚙️ Contraintes",
                    placeholder="Ex: végétarien, rapide...",
                    lines=2
                )
                
                gr.Markdown("### 🎛️ Micro-choix")
                
                with gr.Row():
                    creativity_slider = gr.Slider(0, 3, value=0, step=1, label="🎨 Créativité")
                    texture_choice = gr.Radio(
                        ["Très crémeux", "Équilibré", "Très ferme"],
                        value="Équilibré",
                        label="🧈 Texture"
                    )
                
                with gr.Row():
                    affinage_slider = gr.Slider(0, 12, value=4, step=1, label="⏱️ Affinage (semaines)")
                    spice_choice = gr.Radio(
                        ["Neutre", "Modéré", "Intense"],
                        value="Neutre",
                        label="🌶️ Épices"
                    )
                
                generate_all_btn = gr.Button(
                    "✨ Générer la recette + Recherche web", 
                    variant="primary", 
                    size="lg"
                )
                
                gr.Markdown("⏳ *La génération + recherche web prend 10-15 secondes...*")
            
            with gr.Column(scale=1):
                gr.Markdown("""
                ### 💡 Comment ça marche ?
                
                1️⃣ Entrez vos ingrédients
                2️⃣ Ajustez les micro-choix
                3️⃣ Cliquez sur "Générer"
                
                **Résultat :**
                - Onglet 1 : Votre recette personnalisée
                - Onglet 2 : 6 recettes similaires du web
                
                **Tout se remplit automatiquement !**
                """)
        
        # ===== FONCTIONS LOCALES =====
        def load_history():
            """Charge l'historique avec résumé détaillé"""
            print("🔍 DEBUG: load_history() appelé")
            
            try:
                # Charger l'historique
                if hasattr(agent, 'history') and agent.history:
                    history = agent.history
                    print(f"   → Historique depuis agent.history: {len(history)} recettes")
                elif os.path.exists(agent.recipes_file):
                    with open(agent.recipes_file, 'r', encoding='utf-8') as f:
                        history = json.load(f)
                    print(f"   → Historique depuis fichier: {len(history)} recettes")
                else:
                    print("   → Aucun historique trouvé")
                    return "📭 Aucune recette sauvegardée", []
                
                if not history:
                    print("   → Historique vide")
                    return "📭 Aucune recette sauvegardée", []
                
                # Créer les choix pour le dropdown
                choices = []
                for entry in history[-20:][::-1]:  # 20 dernières, ordre inverse
                    cheese_name = entry.get('cheese_name', 'Sans nom')
                    id_num = entry.get('id', 0)
                    date = entry.get('date', '')[:10] if entry.get('date') else ''
                    
                    if date:
                        choice_text = f"#{id_num} - {cheese_name} ({date})"
                    else:
                        choice_text = f"#{id_num} - {cheese_name}"
                    
                    choices.append(choice_text)
                
                print(f"   ✅ Choices créés: {len(choices)} recettes")
                
                # ✅ CRÉER UN RÉSUMÉ DÉTAILLÉ
                summary = f"📚 {len(history)} recette(s) sauvegardée(s)\n"
                summary += "═" * 60 + "\n\n"
                summary += "🧀 DERNIÈRES RECETTES :\n\n"
                
                # Afficher les 10 dernières recettes
                for entry in history[-10:][::-1]:
                    try:
                        cheese_name = entry.get('cheese_name', 'Sans nom')
                        id_num = entry.get('id', 0)
                        date = entry.get('date', '')[:16] if entry.get('date') else 'Date inconnue'
                        ingredients = entry.get('ingredients', [])
                        cheese_type = entry.get('type', 'Type inconnu')
                        
                        summary += f"#{id_num} - {cheese_name}\n"
                        summary += f"   📅 {date}\n"
                        summary += f"   🧀 Type: {cheese_type}\n"
                        
                        # Afficher les 3 premiers ingrédients
                        if ingredients:
                            ing_preview = ', '.join(ingredients[:3])
                            if len(ingredients) > 3:
                                ing_preview += f"... (+{len(ingredients)-3})"
                            summary += f"   🥛 {ing_preview}\n"
                        
                        summary += "─" * 60 + "\n\n"
                    except Exception as e:
                        print(f"   ⚠️ Erreur sur une entrée: {e}")
                        continue
                
                return summary, choices
                
            except Exception as e:
                print(f"❌ Erreur load_history: {e}")
                import traceback
                traceback.print_exc()
                return f"❌ Erreur: {str(e)}", []
        def show_recipe_select(choice):
            """Affiche la recette sélectionnée"""
            if not choice:
                return ""
            try:
                id_num = int(choice.split('#')[1].split('-')[0])
                return agent.get_recipe_by_id(id_num)
            except:
                return "❌ Erreur chargement recette"

        def agent_clear_history():
            """Efface l'historique"""
            try:
                import json
                import os
                
                # Effacer le fichier
                recipes_file = "recipes_history.json"
                with open(recipes_file, 'w', encoding='utf-8') as f:
                    json.dump([], f)
                
                # Effacer en mémoire
                if hasattr(agent, 'history'):
                    agent.history = []
                
                print("✅ Historique effacé")
                
                return (
                    "✅ Historique effacé avec succès",
                    gr.update(choices=[], value=None),
                    ""
                )
            except Exception as e:
                print(f"❌ Erreur clear: {e}")
                return (
                    f"❌ Erreur: {str(e)}",
                    gr.update(choices=[], value=None),
                    ""
                )

        def generate_all(ingredients, cheese_type, constraints, creativity, texture, affinage, spice):
            """Génère recette + recherche web"""
            try:
                # Générer la recette
                recipe = agent.generate_recipe_creative(
                    ingredients, cheese_type, constraints, 
                    creativity, texture, affinage, spice
                )
                
                # Sauvegarder dans l'historique
                ingredients_list = [ing.strip() for ing in ingredients.split(',')]
                agent._save_to_history(ingredients_list, cheese_type, constraints, recipe)
                
                # Rechercher sur le web
                try:
                    web_recipes = agent.search_web_recipes(ingredients, cheese_type, max_results=6)
                except Exception as e:
                    print(f"⚠️ Erreur recherche web: {e}")
                    web_recipes = []
                
                # Construire HTML
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
                
                print("✅ Génération terminée avec succès")
                return recipe, "", cards_html
                
            except Exception as e:
                print(f"❌ Erreur generate_all: {e}")
                import traceback
                traceback.print_exc()
                return f"❌ Erreur: {str(e)}", "❌ Erreur", "<div class='no-recipes'>❌ Erreur technique</div>"
     
        # ✅ AJOUTER CES DEUX FONCTIONS ICI
        def load_and_populate():
            """Charge ET met à jour le dropdown"""
            summary, choices = load_history()
            print(f"🔄 Wrapper: summary={len(summary)} chars, choices={choices}")
            return summary, gr.Dropdown(choices=choices, value=None)
        
        def clear_and_reset():
            """Efface et reset"""
            result = agent_clear_history()
            # agent_clear_history retourne déjà 3 valeurs
            return result
        
        # ===== ONGLETS =====
        with gr.Tabs():
            # ONGLET 1
            with gr.Tab("📖 Ma Recette"):
                recipe_output = gr.Textbox(
                    label="Votre recette complète",
                    lines=30,
                    max_lines=50,
                    placeholder="Votre recette apparaîtra ici après génération..."
                )
            
            # ONGLET 2
            with gr.Tab("🌐 Recettes Web"):
                search_status = gr.HTML(label="Statut", value="")
                web_results = gr.HTML(
                    label="Résultats",
                    value="<div class='no-recipes'>Cliquez sur 'Générer' pour lancer la recherche web...</div>"
                )
            
            # ONGLET 3
            with gr.Tab("📚 Base de connaissances"):
                with gr.Row():
                    knowledge_btn = gr.Button("📖 Charger résumé COMPLET", variant="primary")
                
                knowledge_output = gr.Textbox(
                    label="🧀 SAVOIR FROMAGÈRE COMPLET", 
                    lines=45, 
                    max_lines=60,
                    placeholder="Cliquez pour charger TOUS les types, épices, dosages..."
                )
                
                knowledge_btn.click(
                    fn=agent.get_knowledge_summary,
                    outputs=knowledge_output
                )
            
            # ONGLET 4 : Historique
            with gr.Tab("🕒 Historique"):
                gr.Markdown("### 📚 Historique de vos recettes")
                
                with gr.Row():
                    history_btn = gr.Button("📋 Charger mes recettes", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Effacer tout", variant="stop", size="lg")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        history_summary = gr.Textbox(
                            label="📊 Résumé",
                            lines=10,
                            interactive=False,
                            placeholder="Cliquez sur 'Charger mes recettes' pour voir le résumé..."
                        )
                    
                    with gr.Column(scale=2):
                        recipe_dropdown = gr.Dropdown(
                            label="🍽️ Sélectionner une recette",
                            choices=[],
                            interactive=True,
                            value=None
                        )
                        
                        recipe_display = gr.Textbox(
                            label="📖 Recette complète",
                            lines=25,
                            interactive=False,
                            placeholder="Sélectionnez une recette dans la liste..."
                        )
                
                # === CONNEXIONS ===
                history_btn.click(
                    fn=load_and_populate,
                    inputs=[],
                    outputs=[history_summary, recipe_dropdown]
                )
                
                recipe_dropdown.select(
                    fn=show_recipe_select,
                    inputs=[recipe_dropdown],
                    outputs=[recipe_display]
                )
                
                # ✅ FONCTION POUR EFFACER
                def clear_and_reset():
                    """Efface et reset"""
                    result = agent_clear_history()
                    return "✅ Historique effacé", gr.Dropdown(choices=[], value=None), ""
                
                # ✅ CONNEXION DU BOUTON EFFACER
                clear_btn.click(
                    fn=clear_and_reset,
                    inputs=[],
                    outputs=[history_summary, recipe_dropdown, recipe_display]
                )
            
            # ONGLET 5
            with gr.Tab("🧪 Test Internet"):
                test_btn = gr.Button("🔍 Tester")
                test_output = gr.Textbox(lines=5)
                test_btn.click(fn=agent.test_internet, outputs=test_output)
        
        # ===== CONNEXION BOUTON PRINCIPAL =====
        generate_all_btn.click(
            fn=generate_all,
            inputs=[
                ingredients_input,
                cheese_type_input,
                constraints_input,
                creativity_slider,
                texture_choice,
                affinage_slider,
                spice_choice
            ],
            outputs=[recipe_output, search_status, web_results]
        )
        
        gr.Markdown("""
        ---
        <center>
        Fait avec 🧀 et 🤖 | Hugging Face Spaces | © 2026 Braconier
        </center>
        """)
    
    return demo
def generate_all(ingredients, cheese_type, constraints, creativity, texture, affinage, spice):
    """Génère + FORCE historique + recherche web"""
    try:
        # 1. GÉNÉRATION
        recipe = agent.generate_recipe_creative(
            ingredients, cheese_type, constraints, creativity, texture, affinage, spice
        )
        
        # 2. FORCE HISTORIQUE (AVANT web)
        ingredients_list = [ing.strip() for ing in ingredients.split(',')]
        agent._save_to_history(ingredients_list, cheese_type, constraints, recipe)
        
        # 3. RECHERCHE WEB (fallback si erreur)
        try:
            web_recipes = agent.search_web_recipes(ingredients, cheese_type, max_results=6)
        except:
            web_recipes = []
        
        # 4. CARDS HTML
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
                ✅ {len(web_recipes)} recettes web trouvées !
            </div>
            """
            for i, r in enumerate(web_recipes[:6], 1):
                cards_html += f"""
                <div class="recipe-card">
                    <b>{i}. {r.get('title', 'Recette')}</b><br>
                    📍 {r.get('source', 'Web')}<br>
                    {r.get('description', '')[:200]}...
                    <br><a href="{r.get('url', '#')}" target="_blank">🔗 Voir</a>
                </div>
                """
        
        print("✅ Génération + historique OK")
        return recipe, "", cards_html
        
    except Exception as e:
        print(f"❌ Erreur generate_all: {e}")
        return "❌ Erreur génération", "Erreur", "Erreur technique"

    # ===== CONNECTER LE BOUTON =====
    generate_all_btn.click(
        fn=generate_all,
        inputs=[
            ingredients_input,
            cheese_type_input,
            constraints_input,
            creativity_slider,
            texture_choice,
            affinage_slider,
            spice_choice
        ],
        outputs=[recipe_output, search_status, web_results]
    )
    
    gr.Markdown("""
    ---
    <center>
    Fait avec 🧀 et 🤖 | Hugging Face Spaces | © 2026 Braconier
    </center>
    """)

    return demo

# ========================================
# LANCEMENT DE L'APPLICATION
# ========================================
if __name__ == "__main__":
    # 🧀 THÈME FROMAGER - Couleurs chaudes et gourmandes
    fromage_theme = gr.themes.Soft(
        primary_hue="amber",      # Jaune doré comme un fromage affiné
        secondary_hue="orange",   # Orange crémeux
        neutral_hue="stone",      # Beige pierre comme une cave à fromage
        font=gr.themes.GoogleFont("Quicksand"),  # Police ronde et douce
    ).set(
        # Couleurs primaires
        body_background_fill="#FFF9E6",           # Crème légère
        body_background_fill_dark="#2C2416",      # Marron cave sombre
        
        # Boutons
        button_primary_background_fill="#FF8F00",      # Orange fromage
        button_primary_background_fill_hover="#FF6F00", # Orange plus foncé
        button_primary_text_color="#FFFFFF",
        
        # Inputs
        input_background_fill="#FFFBF0",          # Blanc crémeux
        input_border_color="#FFB74D",             # Bordure orange douce
        
        # Tabs
        block_label_text_color="#E65100",         # Orange foncé
        block_title_text_color="#BF360C",         # Marron fromage affiné
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
            background: linear-gradient(135deg, #FFF9E6 0%, #FFE5B4 100%) !important;
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
        
        .tab-nav button {
            background: #FFF3E0 !important;
            color: #5D4037 !important;
            border: 2px solid #FFE0B2 !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            padding: 12px 24px !important;
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
    interface.launch(
        theme=fromage_theme,
        css=custom_css
    )
    