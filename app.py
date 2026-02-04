import gradio as gr
import json
import os
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download


class AgentFromagerHF:
    """Agent fromager avec base de connaissances complète"""

    def __init__(self):
        self.knowledge_base = self._init_knowledge()
        self.recipes_file = "recipes_history.json"
        self.hf_repo = "volubyl/fromager-recipes"
        self.hf_token = os.environ.get("HF_TOKEN")
        self.api = HfApi(token=self.hf_token) if self.hf_token else None
        self._ensure_local_history()

    # ======================================================================
    # BASE DE CONNAISSANCES — INCHANGÉE (INTÉGRALE)
    # ======================================================================

    def _init_knowledge(self):
        return {
            "types_pate": {
                "Fromage frais": {
                    "description": "Non affiné, humide, à consommer rapidement",
                    "exemples": "Fromage blanc, faisselle, ricotta, cottage cheese",
                    "duree": "0-3 jours",
                    "difficulte": "Facile - Idéal débutants"
                },
                "Pâte molle": {
                    "description": "Croûte fleurie ou lavée, texture crémeuse",
                    "exemples": "Camembert, brie, munster, reblochon",
                    "duree": "2-8 semaines",
                    "difficulte": "Moyenne - Nécessite une cave"
                },
                "Pâte pressée non cuite": {
                    "description": "Pressée sans cuisson, texture ferme",
                    "exemples": "Cantal, saint-nectaire, morbier, tomme",
                    "duree": "1-6 mois",
                    "difficulte": "Moyenne - Matériel spécifique"
                },
                "Pâte pressée cuite": {
                    "description": "Caillé chauffé puis pressé, longue conservation",
                    "exemples": "Comté, gruyère, beaufort, parmesan",
                    "duree": "3-36 mois",
                    "difficulte": "Difficile - Expertise requise"
                },
                "Pâte persillée": {
                    "description": "Avec moisissures bleues, goût prononcé",
                    "exemples": "Roquefort, bleu d'Auvergne, gorgonzola, stilton",
                    "duree": "2-6 mois",
                    "difficulte": "Difficile - Contrôle précis"
                }
            },

            "ingredients_base": {
                "Lait": [
                    "Vache (doux)", "Chèvre (acidulé)", "Brebis (riche)",
                    "Bufflonne (crémeux)", "Mélange"
                ],
                "Coagulant": [
                    "Présure animale", "Présure végétale",
                    "Jus de citron", "Vinaigre blanc"
                ],
                "Ferments": [
                    "Lactiques", "Mésophiles", "Thermophiles"
                ],
                "Sel": [
                    "Sel fin", "Gros sel", "Sel de mer", "Saumure"
                ]
            },

            "epices_et_aromates": {
                "Herbes fraîches": [
                    "Basilic", "Ciboulette", "Thym", "Romarin",
                    "Persil", "Aneth", "Menthe", "Coriandre"
                ],
                "Épices": [
                    "Poivre", "Paprika", "Cumin", "Curry",
                    "Piment", "Fenugrec", "Nigelle"
                ],
                "Aromates spéciaux": [
                    "Ail", "Échalote", "Zeste d'agrumes",
                    "Gingembre", "Citronnelle"
                ]
            },

            "techniques_aromatisation": {
                "Incorporation dans le caillé":
                    "Ajouter les épices au moulage",
                "Enrobage externe":
                    "Rouler le fromage après salage",
                "Saumure parfumée":
                    "Infuser herbes et épices dans la saumure"
            },

            "dosages_recommandes": {
                "Herbes fraîches": "2-3 c. à soupe / kg",
                "Herbes séchées": "1-2 c. à soupe / kg",
                "Épices moulues": "1-2 c. à café / kg",
                "Sel": "1,5 à 2 % du poids"
            },

            "problemes_courants": {
                "Caillé trop mou":
                    "Pas assez de présure ou température trop basse",
                "Fromage trop acide":
                    "Fermentation trop longue",
                "Moisissures indésirables":
                    "Humidité excessive ou hygiène insuffisante",
                "Fromage trop sec":
                    "Égouttage excessif"
            },

            "temperatures_affinage": {
                "Fromage frais": "4-6°C",
                "Pâte molle": "10-12°C, 90% humidité",
                "Pâte pressée non cuite": "12-14°C",
                "Pâte pressée cuite": "14-18°C",
                "Pâte persillée": "8-10°C, 95% humidité"
            },

            "conservation": {
                "Fromage frais": "3-5 jours au réfrigérateur",
                "Pâte molle": "2-3 semaines",
                "Pâte pressée": "1-6 mois",
                "Pâte persillée": "3-4 semaines"
            },

            "accords_vins": {
                "Fromage frais": "Vin blanc sec",
                "Pâte molle": "Champagne ou rouge léger",
                "Pâte pressée": "Vin rouge structuré",
                "Pâte persillée": "Vin doux (Sauternes)"
            }
        }

    # ======================================================================
    # ACCÈS STRUCTURÉ À LA BASE
    # ======================================================================

    def _get_type_info(self, cheese_type):
        return self.knowledge_base["types_pate"].get(
            cheese_type,
            self.knowledge_base["types_pate"]["Fromage frais"]
        )

    def _get_temperature_affinage(self, cheese_type):
        return self.knowledge_base["temperatures_affinage"].get(
            cheese_type, "10-12°C"
        )

    def _get_conservation_info(self, cheese_type):
        return self.knowledge_base["conservation"].get(
            cheese_type, "Consommation rapide"
        )

    def _get_accord_vin(self, cheese_type):
        return self.knowledge_base["accords_vins"].get(
            cheese_type, "Vin au choix"
        )

    # ======================================================================
    # RECETTE (UTILISE RÉELLEMENT LA BASE)
    # ======================================================================

    def generate_recipe(self, ingredients, cheese_type, constraints):
        ingredients_list = [i.strip() for i in ingredients.split(",") if i.strip()]
        if not ingredients_list:
            return "❌ Aucun ingrédient fourni"

        if cheese_type == "Laissez l'IA choisir":
            cheese_type = "Fromage frais"

        info = self._get_type_info(cheese_type)

        recipe = f"""
🧀 {cheese_type.upper()}

📋 Description
{info['description']}

🕒 Durée
{info['duree']}

⚙️ Difficulté
{info['difficulte']}

🌡️ Température d'affinage
{self._get_temperature_affinage(cheese_type)}

📦 Conservation
{self._get_conservation_info(cheese_type)}

🍷 Accord vin
{self._get_accord_vin(cheese_type)}

🥛 Ingrédients fournis
- """ + "\n- ".join(ingredients_list)

        if constraints:
            recipe += f"\n\n⚙️ Contraintes prises en compte : {constraints}"

        self._save_history(recipe)
        return recipe

    # ======================================================================
    # HISTORIQUE
    # ======================================================================

    def _ensure_local_history(self):
        if not os.path.exists(self.recipes_file):
            with open(self.recipes_file, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _save_history(self, recipe):
        with open(self.recipes_file, "r", encoding="utf-8") as f:
            history = json.load(f)

        history.append({
            "date": datetime.now().isoformat(),
            "recipe": recipe
        })

        with open(self.recipes_file, "w", encoding="utf-8") as f:
            json.dump(history[-100:], f, indent=2, ensure_ascii=False)

    def get_history_display(self):
        with open(self.recipes_file, "r", encoding="utf-8") as f:
            history = json.load(f)

        if not history:
            return "📭 Aucun historique"

        return "\n\n".join(
            f"🧀 {i+1} — {h['date']}" for i, h in enumerate(history[::-1])
        )

    # ======================================================================
    # AFFICHAGE COMPLET DE LA BASE
    # ======================================================================

    def get_knowledge_summary(self):
        txt = "📚 BASE DE CONNAISSANCES FROMAGÈRES\n\n"
        for section, content in self.knowledge_base.items():
            txt += f"\n=== {section.upper()} ===\n"
            if isinstance(content, dict):
                for k, v in content.items():
                    txt += f"\n• {k}\n"
                    if isinstance(v, dict):
                        for sk, sv in v.items():
                            txt += f"  - {sk} : {sv}\n"
                    elif isinstance(v, list):
                        for item in v:
                            txt += f"  - {item}\n"
                    else:
                        txt += f"  {v}\n"
        return txt


# ======================================================================
# INTERFACE
# ======================================================================

agent = AgentFromagerHF()

def create_interface():
    with gr.Blocks(title="🧀 Agent Fromager") as demo:
        gr.Markdown("# 🧀 Agent Fromager Intelligent")

        with gr.Tab("Créer une recette"):
            ing = gr.Textbox(label="Ingrédients", lines=3)
            typ = gr.Dropdown(
                ["Laissez l'IA choisir", "Fromage frais", "Pâte molle",
                 "Pâte pressée non cuite", "Pâte pressée cuite", "Pâte persillée"],
                value="Laissez l'IA choisir"
            )
            cons = gr.Textbox(label="Contraintes", lines=2)
            btn = gr.Button("Générer")
            out = gr.Textbox(lines=25)
            btn.click(agent.generate_recipe, [ing, typ, cons], out)

        with gr.Tab("📚 Base de connaissances"):
            gr.Textbox(value=agent.get_knowledge_summary(), lines=40)

        with gr.Tab("🕒 Historique"):
            gr.Textbox(value=agent.get_history_display(), lines=20)

    return demo


if __name__ == "__main__":
    create_interface().launch()
