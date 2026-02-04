import gradio as gr
import json
import os
from datetime import datetime

class AgentFromagerHF:
    """Agent fromager pour Hugging Face Spaces"""
    
    def __init__(self):
        self.knowledge_base = self._init_knowledge()
        self.recipes_file = 'recipes_history.json'
    
    def _init_knowledge(self):
        """Base de connaissances fromage"""
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
            }
        }
    
    def validate_ingredients(self, ingredients_text):
        """Valide les ingrédients"""
        if not ingredients_text or not ingredients_text.strip():
            return False, "⚠️ Vous devez entrer au moins un ingrédient !"
        
        ingredients_lower = ingredients_text.lower()
        
        # Vérifier présence de lait
        has_milk = any(word in ingredients_lower for word in 
                      ['lait', 'milk', 'vache', 'chèvre', 'brebis', 'bufflonne'])
        
        if not has_milk:
            return False, "❌ Il faut du lait pour faire du fromage !\n💡 Ajoutez : lait de vache, chèvre, brebis..."
        
        # Vérifier présence de coagulant
        has_coagulant = any(word in ingredients_lower for word in 
                           ['présure', 'presure', 'citron', 'vinaigre', 'acide'])
        
        if not has_coagulant:
            return True, "⚠️ Aucun coagulant détecté. Je suggérerai présure ou citron dans la recette.\n✅ Validation OK, génération possible."
        
        return True, "✅ Ingrédients parfaits pour faire du fromage ! Tous les éléments essentiels sont présents."
    
    def generate_recipe(self, ingredients, cheese_type, constraints):
        """Génère une recette de fromage détaillée"""
        
        # Validation
        valid, message = self.validate_ingredients(ingredients)
        if not valid:
            return message
        
        # Parser les ingrédients
        ingredients_list = [ing.strip() for ing in ingredients.split(',')]
        
        # Construction de la recette basée sur les templates
        cheese_type_clean = cheese_type if cheese_type != "Laissez l'IA choisir" else "Fromage artisanal"
        
        # Template de recette détaillée
        recipe = self._generate_detailed_recipe(
            ingredients_list, 
            cheese_type_clean, 
            constraints
        )
        
        # Sauvegarder dans l'historique
        self._save_to_history(ingredients_list, cheese_type_clean, constraints, recipe)
        
        return recipe
    
    def _generate_detailed_recipe(self, ingredients, cheese_type, constraints):
        """Génère une recette détaillée basée sur templates"""
        
        # Déterminer le type si "artisanal"
        if cheese_type == "Fromage artisanal":
            ingredients_str = ' '.join(ingredients).lower()
            if 'citron' in ingredients_str or 'vinaigre' in ingredients_str:
                cheese_type = "Fromage frais"
            elif any(x in ingredients_str for x in ['herbe', 'épice', 'cendr']):
                cheese_type = "Pâte molle aromatisée"
            else:
                cheese_type = "Pâte molle"
        
        # Obtenir les infos du type
        type_info = None
        for key, value in self.knowledge_base['types_pate'].items():
            if key.lower() in cheese_type.lower():
                type_info = value
                break
        
        if not type_info:
            type_info = self.knowledge_base['types_pate']['Fromage frais']
        
        # Générer nom créatif
        cheese_name = self._generate_name(ingredients, cheese_type)
        
        # Construction de la recette
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
- 2 litres de lait ({ingredients[0] if 'lait' in ingredients[0].lower() else 'lait entier'})
- 2ml de présure liquide (ou 1/4 de comprimé)
  Alternative : 60ml de jus de citron frais
- 10g de sel de mer fin
- Ferments lactiques (1 sachet) - optionnel
{self._format_additional_ingredients(ingredients)}


🔧 MATÉRIEL NÉCESSAIRE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Grande casserole inox (3L minimum)
- Thermomètre de cuisson (précision ±1°C)
- Moule à fromage perforé (ou passoire + étamine)
- Toile à fromage (étamine/mousseline)
- Louche et couteau long
- Récipient pour égouttage
- Cave d'affinage ou frigo (10-14°C, 80-90% humidité)


📝 ÉTAPES DE FABRICATION DÉTAILLÉES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PHASE 1 : PRÉPARATION DU LAIT (20 min)
──────────────────────────────────────
1. Verser le lait dans la casserole bien propre
2. Chauffer doucement à 32°C (température du corps)
   ⚠️ Ne JAMAIS dépasser 35°C au risque de tuer les ferments
3. Maintenir cette température pendant 10 minutes
4. Si utilisation de ferments : les ajouter maintenant et mélanger 1 minute


PHASE 2 : CAILLAGE (45-90 min)
────────────────────────────────
5. Ajouter la présure (ou le citron) en mélangeant délicatement 30 secondes
6. Couvrir et laisser reposer SANS BOUGER pendant :
   - Présure : 45-60 minutes
   - Citron : 20-30 minutes (caillage plus rapide mais moins stable)
7. Test de caillage : le caillé doit se briser net, comme du tofu
   Si encore liquide → attendre 15 min supplémentaires


PHASE 3 : DÉCOUPAGE ET BRASSAGE (15 min)
─────────────────────────────────────────
8. Découper le caillé en cubes de 1cm avec un couteau long
   Faire un quadrillage vertical puis horizontal
9. Laisser reposer 5 minutes (le petit-lait commence à sortir)
10. Brasser TRÈS doucement pendant 10 minutes
    Le caillé doit se raffermir sans se désintégrer


PHASE 4 : MOULAGE ET ÉGOUTTAGE (4-12h)
───────────────────────────────────────
11. Disposer l'étamine dans le moule perforé
12. Transférer le caillé à la louche (garder le petit-lait !)
13. Laisser égoutter naturellement :
    - Fromage frais : 2-4 heures à température ambiante
    - Autres : 12-24 heures au frais (12°C)
14. Retourner le fromage toutes les 4 heures pour égouttage uniforme


PHASE 5 : SALAGE (Variable selon type)
───────────────────────────────────────
15. Méthode au sel sec :
    - Démouler délicatement
    - Frotter toutes les faces avec le sel
    - Quantité : 2% du poids du fromage (env. 10g)
16. Ou méthode en saumure :
    - Immerger dans eau saturée en sel (300g/L) pendant 2-4h


PHASE 6 : AFFINAGE (Selon type choisi)
───────────────────────────────────────
{self._get_affinage_instructions(cheese_type)}


⚠️ POINTS DE VIGILANCE - ERREURS FRÉQUENTES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ Lait UHT : Préférer du lait cru ou pasteurisé (pas stérilisé)
❌ Température trop haute : Détruit les ferments, pas de caillage
❌ Caillage incomplet : Perte de rendement, texture granuleuse
❌ Sel trop tôt : Bloque l'acidification
❌ Affinage trop sec : Le fromage craque
❌ Affinage trop humide : Moisissures indésirables


🍷 DÉGUSTATION ET ACCORDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Moment idéal : {self._get_tasting_time(cheese_type)}

Température de service : 18-20°C (sortir 1h avant)

Accords parfaits :
- Pain de campagne au levain (croûte croustillante)
- Fruits frais selon saison (raisin, pomme, figue)
- Vin : {self._get_wine_pairing(cheese_type)}
- Miel de châtaignier ou confiture de figues


🎨 VARIANTES CRÉATIVES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Version aux herbes : Ajouter thym, romarin, basilic dans le caillé
2. Version poivrée : Enrober de poivre noir concassé après salage
3. Version cendrée : Saupoudrer de cendres végétales alimentaires
4. Version aillée : Incorporer ail des ours haché finement


💡 CONSEILS DU MAÎTRE FROMAGER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✨ Le petit-lait (lactosérum) est précieux ! Utilisez-le pour :
   - Faire du pain (remplace l'eau)
   - Arroser les plantes (riche en nutriments)
   - Base de smoothies protéinés

✨ Hygiène irréprochable : Stériliser tout le matériel à l'eau bouillante

✨ Patience : Un bon fromage ne se précipite pas. Respectez les temps.

✨ Carnet de bord : Notez températures, durées, résultats pour progresser

✨ Cave d'affinage DIY : Une glacière avec bol d'eau + hygromètre fait l'affaire


📊 VALEURS NUTRITIONNELLES (Pour 100g)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Calories : 250-350 kcal
- Protéines : 18-25g
- Lipides : 20-30g
- Calcium : 600-800mg
- Sodium : Variable selon salage


🔬 SCIENCE DU FROMAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Le caillage : La présure (enzyme) coupe les protéines du lait (caséines),
qui s'agglomèrent en réseau 3D emprisonnant eau et matières grasses.

L'affinage : Les bactéries et levures transforment protéines et graisses
en molécules aromatiques complexes. Plus long = goût plus prononcé.


{self._add_constraints_note(constraints)}

╔══════════════════════════════════════════════════════════════╗
║  Recette générée le {datetime.now().strftime('%d/%m/%Y à %H:%M')}           
║  Bonne fabrication ! 🧀                                       
╚══════════════════════════════════════════════════════════════╝
"""
        return recipe
    
    def _generate_name(self, ingredients, cheese_type):
        """Génère un nom créatif"""
        names = {
            'frais': ['Blanc Nuage', 'Fraîcheur Lactée', 'Douceur Matinale'],
            'molle': ['Velours de Cave', 'Crème d\'Artisan', 'Délice Fondant'],
            'pressée': ['Roc du Terroir', 'Tradition Pressée', 'Meule d\'Or'],
            'persillée': ['Bleu des Monts', 'Marbré Mystère', 'Azur Intense']
        }
        
        for key in names:
            if key in cheese_type.lower():
                import random
                return random.choice(names[key])
        
        return "Fromage Maison"
    
    def _format_additional_ingredients(self, ingredients):
        """Formate les ingrédients additionnels"""
        additional = [ing for ing in ingredients if 'lait' not in ing.lower()]
        if not additional:
            return ""
        
        result = "\n• Ingrédients spéciaux fournis :\n"
        for ing in additional:
            result += f"  - {ing.capitalize()}\n"
        return result
    
    def _get_affinage_instructions(self, cheese_type):
        """Instructions d'affinage selon le type"""
        instructions = {
            'frais': """
17. FROMAGE FRAIS - Pas d'affinage nécessaire !
    ✅ Consommer immédiatement ou dans les 3-5 jours
    Conservation : Au frigo (4°C) dans boîte hermétique
    Astuce : Ajouter herbes fraîches juste avant service
""",
            'molle': """
17. Placer en cave d'affinage (10-12°C, 90% humidité)
18. Retourner tous les 2 jours pendant 2 semaines
19. Brosser délicatement si croûte blanche apparaît (normal !)
20. Surveiller : odeur de champignon = bon signe
21. Durée minimale : 2 semaines
    Durée optimale : 4-6 semaines
""",
            'pressée': """
17. Affinage en cave fraîche (12-14°C, 85% humidité)
18. Retourner tous les jours la première semaine
19. Puis 2 fois par semaine ensuite
20. Frotter avec saumure 1x/semaine pour développer la croûte
21. Durée minimale : 4 semaines
    Durée optimale : 2-3 mois pour texture ferme
""",
            'persillée': """
17. Percer le fromage avec une aiguille stérilisée (20 trous)
    → Permet à l'air d'entrer pour développer le bleu
18. Affinage en cave humide (10°C, 95% humidité)
19. Retourner tous les 3 jours
20. Les veines bleues apparaissent après 2-3 semaines
21. Durée minimale : 6 semaines
    Durée optimale : 2-3 mois
"""
        }
        
        for key, value in instructions.items():
            if key in cheese_type.lower():
                return value
        
        return instructions['molle']  # Par défaut
    
    def _get_tasting_time(self, cheese_type):
        """Moment optimal de dégustation"""
        if 'frais' in cheese_type.lower():
            return "Immédiatement après fabrication"
        elif 'molle' in cheese_type.lower():
            return "Après 3-6 semaines d'affinage"
        elif 'persillée' in cheese_type.lower():
            return "Après 2-3 mois minimum"
        else:
            return "Après 1-2 mois d'affinage"
    
    def _get_wine_pairing(self, cheese_type):
        """Accord vin selon le fromage"""
        pairings = {
            'frais': 'Blanc sec et vif (Sauvignon, Muscadet)',
            'molle': 'Rouge léger ou Champagne (Pinot Noir, Crémant)',
            'pressée': 'Rouge charpenté (Côtes du Rhône, Bordeaux)',
            'persillée': 'Blanc doux ou rouge puissant (Sauternes, Porto)'
        }
        
        for key, value in pairings.items():
            if key in cheese_type.lower():
                return value
        
        return 'Vin rouge de caractère'
    
    def _add_constraints_note(self, constraints):
        """Ajoute une note sur les contraintes"""
        if not constraints or constraints.strip() == "":
            return ""
        
        return f"""

⚙️ ADAPTATION AUX CONTRAINTES : {constraints.upper()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{self._generate_constraint_advice(constraints)}
"""
    
    def _generate_constraint_advice(self, constraints):
        """Conseils selon contraintes"""
        constraints_lower = constraints.lower()
        advice = []
        
        if 'végétarien' in constraints_lower or 'vegetarien' in constraints_lower:
            advice.append("✓ Utiliser de la présure végétale (extraite de chardon ou figuier)")
            advice.append("✓ Vérifier que les ferments sont d'origine non-animale")
        
        if 'rapide' in constraints_lower:
            advice.append("✓ Privilégier un fromage frais (prêt en 4-6h)")
            advice.append("✓ Utiliser du citron pour caillage accéléré (20 min)")
        
        if 'lactose' in constraints_lower:
            advice.append("✓ Les fromages affinés contiennent peu de lactose (consommé par bactéries)")
            advice.append("✓ Utiliser du lait délactosé ou lait de chèvre (plus digeste)")
        
        if 'vegan' in constraints_lower or 'végétalien' in constraints_lower:
            advice.append("✓ Utiliser lait végétal (soja, cajou) + agar-agar ou tapioca")
            advice.append("✓ Ferments : probiotiques en poudre ou rejuvelac")
        
        if not advice:
            advice.append("Aucune adaptation spécifique nécessaire.")
        
        return '\n'.join(advice)
    
    def _save_to_history(self, ingredients, cheese_type, constraints, recipe):
        """Sauvegarde dans l'historique"""
        try:
            history = []
            if os.path.exists(self.recipes_file):
                with open(self.recipes_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            
            history.append({
                'date': datetime.now().isoformat(),
                'ingredients': ingredients,
                'type': cheese_type,
                'constraints': constraints,
                'recipe_preview': recipe[:500] + "..."
            })
            
            # Garder seulement les 50 dernières
            history = history[-50:]
            
            with open(self.recipes_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except:
            pass  # Pas critique si ça échoue
    
    def get_knowledge_summary(self):
        """Retourne un résumé de la base de connaissances"""
        summary = "📚 BASE DE CONNAISSANCES FROMAGE\n\n"
        
        summary += "🧀 TYPES DE PÂTE :\n"
        for name, info in self.knowledge_base['types_pate'].items():
            summary += f"\n• {name}\n"
            summary += f"  {info['description']}\n"
            summary += f"  Exemples : {info['exemples']}\n"
            summary += f"  Durée : {info['duree']} | {info['difficulte']}\n"
        
        summary += "\n\n🥛 INGRÉDIENTS ESSENTIELS :\n"
        for category, items in self.knowledge_base['ingredients_base'].items():
            summary += f"\n• {category} :\n"
            for item in items:
                summary += f"  - {item}\n"
        
        return summary

# Initialiser l'agent
agent = AgentFromagerHF()

# Interface Gradio
def create_interface():
    """Crée l'interface Gradio"""
    
    with gr.Blocks(title="🧀 Agent Fromager") as demo:
        
        gr.Markdown("""
        # 🧀 Agent Fromager Intelligent
        ### Créez vos fromages artisanaux avec l'IA
        
        Entrez vos ingrédients et laissez l'intelligence artificielle vous guider pas à pas dans la fabrication de fromages maison de qualité professionnelle.
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
                            value="Laissez l'IA choisir",
                            info="L'IA suggérera le meilleur type selon vos ingrédients"
                        )
                        
                        constraints_input = gr.Textbox(
                            label="⚙️ Contraintes ou préférences (optionnel)",
                            placeholder="Ex: végétarien, rapide, sans lactose, vegan...",
                            lines=2
                        )
                        
                        generate_btn = gr.Button(
                            "✨ Générer la recette complète",
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
                        
                        **Optionnels :**
                        - Ferments lactiques
                        - Herbes, épices
                        - Cendres, poivre
                        
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
            
            # TAB 3 : À propos
            with gr.Tab("ℹ️ À propos"):
                gr.Markdown("""
                ## 🧀 Agent Fromager Intelligent
                
                ### Créé par Myriam avec ❤️
                
                **Technologies utilisées :**
                - 🤖 Intelligence Artificielle pour génération de recettes
                - 📚 Base de connaissances fromagère professionnelle
                - 🎨 Interface Gradio sur Hugging Face Spaces
                
                **Fonctionnalités :**
                - ✅ Recettes détaillées étape par étape
                - ✅ Adaptation aux contraintes (végétarien, vegan, rapide...)
                - ✅ Conseils de maître fromager
                - ✅ Accords mets et vins
                - ✅ Explications scientifiques
                - ✅ Variantes créatives
                
                **Sources d'inspiration :**
                - Techniques fromagères traditionnelles françaises
                - Connaissances AOP/IGP
                - Artisanat fromager moderne
                
                ---
                
                ### 🌟 Remerciements
                
                Merci à la communauté fromagère pour le partage de savoir-faire ancestral.
                
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

# Lancement
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=False)
