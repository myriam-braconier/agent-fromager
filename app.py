import gradio as gr
import json
import os
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download

class AgentFromagerHF:
    """Agent fromager avec persistance HF Dataset"""
    
    def __init__(self):
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
        """Génère une recette détaillée basée sur templates"""
        
        if cheese_type == "Fromage artisanal":
            ingredients_str = ' '.join(ingredients).lower()
            if 'citron' in ingredients_str or 'vinaigre' in ingredients_str:
                cheese_type = "Fromage frais"
            elif any(x in ingredients_str for x in ['herbe', 'épice', 'cendr']):
                cheese_type = "Pâte molle aromatisée"
            else:
                cheese_type = "Pâte molle"
        
        type_info = None
        for key, value in self.knowledge_base['types_pate'].items():
            if key.lower() in cheese_type.lower():
                type_info = value
                break
        
        if not type_info:
            type_info = self.knowledge_base['types_pate']['Fromage frais']
        
        cheese_name = f"Fromage {cheese_type}"
        
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
• 2 litres de lait entier
• 2ml de présure liquide (ou 1/4 de comprimé)
• 10g de sel de mer fin
• Ferments lactiques (optionnel)
Vos ingrédients : {', '.join(ingredients)}


🔧 MATÉRIEL NÉCESSAIRE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Grande casserole inox (3L minimum)
• Thermomètre de cuisson
• Moule à fromage perforé
• Toile à fromage (étamine)
• Louche et couteau long


📝 ÉTAPES DE FABRICATION DÉTAILLÉES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PHASE 1 : PRÉPARATION DU LAIT (20 min)
──────────────────────────────────────
1. Verser le lait dans la casserole bien propre
2. Chauffer doucement à 32°C
3. Maintenir cette température pendant 10 minutes


PHASE 2 : CAILLAGE (45-90 min)
────────────────────────────────
4. Ajouter la présure et mélanger délicatement
5. Couvrir et laisser reposer 45-60 minutes
6. Test : le caillé doit se briser net


PHASE 3 : DÉCOUPAGE ET BRASSAGE (15 min)
─────────────────────────────────────────
7. Découper le caillé en cubes de 1cm
8. Laisser reposer 5 minutes
9. Brasser doucement pendant 10 minutes


PHASE 4 : MOULAGE ET ÉGOUTTAGE (4-12h)
───────────────────────────────────────
10. Disposer l'étamine dans le moule
11. Transférer le caillé à la louche
12. Laisser égoutter 12-24 heures au frais


PHASE 5 : SALAGE
───────────────────────────────────────
13. Démouler et frotter avec le sel
14. Quantité : 2% du poids du fromage


PHASE 6 : AFFINAGE
───────────────────────────────────────
15. Placer en cave (10-14°C, 85-90% humidité)
16. Durée selon type : {type_info['duree']}


🍷 DÉGUSTATION ET ACCORDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Servir à température ambiante (18-20°C)
Accords : Pain au levain, fruits frais, vin rouge


💡 CONSEILS DU MAÎTRE FROMAGER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✨ Hygiène irréprochable : stériliser tout le matériel
✨ Patience : respecter les temps de repos
✨ Température précise : ±2°C peut changer le résultat
✨ Le petit-lait est précieux : l'utiliser pour le pain


{self._add_constraints_note(constraints) if constraints else ''}

╔══════════════════════════════════════════════════════════════╗
║  Recette générée le {datetime.now().strftime('%d/%m/%Y à %H:%M')}           
║  Bonne fabrication ! 🧀                                       
╚══════════════════════════════════════════════════════════════╝
"""
        return recipe
    
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
        """Retourne un résumé de la base de connaissances"""
        summary = "📚 BASE DE CONNAISSANCES FROMAGE\n\n"
        summary += "🧀 TYPES DE PÂTE :\n"
        summary += "="*70 + "\n\n"
        
        for name, info in self.knowledge_base['types_pate'].items():
            summary += f"• {name.upper()}\n"
            summary += f"  {info['description']}\n"
            summary += f"  Exemples : {info['exemples']}\n"
            summary += f"  Durée : {info['duree']} | Difficulté : {info['difficulte']}\n\n"
        
        summary += "\n" + "="*70 + "\n"
        summary += "🥛 INGRÉDIENTS ESSENTIELS :\n"
        summary += "="*70 + "\n\n"
        
        for category, items in self.knowledge_base['ingredients_base'].items():
            summary += f"\n• {category.upper()} :\n"
            for item in items:
                summary += f"  - {item}\n"
        
        summary += "\n" + "="*70 + "\n"
        summary += "💡 CONSEILS GÉNÉRAUX :\n"
        summary += "="*70 + "\n\n"
        summary += "✓ Hygiène irréprochable : stériliser tout le matériel\n"
        summary += "✓ Température précise : ±2°C peut changer le résultat\n"
        summary += "✓ Patience : un bon fromage ne se précipite pas\n"
        summary += "✓ Tenir un carnet : noter températures et durées\n"
        summary += "✓ Commencer simple : fromage frais avant pâtes pressées\n\n"
        
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