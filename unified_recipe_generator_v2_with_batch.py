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
    
    def __init__(self, knowledge_base=None, agent=None):
        """
        Initialise le générateur unifié V2
        
        Args:
            knowledge_base: Base de connaissances statique (dict)
            agent: Agent avec la méthode chat_with_llm() (optionnel)
        """
        # Priorité : knowledge_base passé en paramètre, sinon depuis l'agent
        if knowledge_base is not None:
            self.knowledge_base = knowledge_base
        elif agent is not None and hasattr(agent, 'knowledge_base'):
            self.knowledge_base = agent.knowledge_base
        else:
            self.knowledge_base = {}
        
        # Stocker l'agent
        self.agent = agent
        
        # Cache et historique
        self.cache = {}
        self.history_file = "unified_recipes_history.json"
        
        # Debug
        print(f"🔍 UnifiedRecipeGeneratorV2 initialisé:")
        print(f"   - knowledge_base: {len(self.knowledge_base)} clés")
        print(f"   - agent: {type(self.agent)}")
        if self.agent:
            print(f"   - agent a chat_with_llm: {hasattr(self.agent, 'chat_with_llm')}")
               
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
        import random
        
        # Seed basé sur les ingrédients
        ingredients_str = ",".join(sorted(ingredients))
        seed = int(hashlib.md5(ingredients_str.encode()).hexdigest()[:8], 16) % 1000
        
        # ✅ Générer le nom créatif du fromage
        try:
            cheese_name = self._generate_creative_name(cheese_type, ingredients)
            print(f"🧀 Nom créatif généré: {cheese_name}")
        except Exception as e:
            # Fallback si la génération échoue
            print(f"⚠️ Erreur génération nom: {e}, utilisation nom par défaut")
            cheese_name = cheese_type.replace("_", " ").title()
            
            random.seed(seed)
        
        # Contexte profil
        profile_context = self._get_profile_context(profile)
        
        # Récupérer les infos du type de fromage depuis la base statique
        type_info = self._get_type_info_from_knowledge(cheese_type)
        
        # ========== NOM CRÉATIF ==========
        prefixes = ["Artisanal", "Fermier", "Maison", "du Terroir", "Authentique", "Rustique"]
        suffixes = ["Frais", "Traditionnel", "Rustique", "Nature", "Gourmand", "Parfumé"]
        
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        
        # Ajouter le lait au nom si spécifié
        lait_name = f"de {lait}" if lait and lait != "vache" else ""
        title = f"{prefix} {cheese_type} {lait_name} {suffix}".strip()
        
        # ========== INGRÉDIENTS AVEC QUANTITÉS ADAPTÉES ==========
        quantite_lait = profile_context['quantite_lait']
        ingredients_list = [
            f"{quantite_lait} de lait {lait or 'de vache'} (entier, pasteurisé ou cru)"
        ]
        
        # Coagulant depuis la base statique
        if 'ingredients_base' in self.knowledge_base and 'Coagulant' in self.knowledge_base['ingredients_base']:
            coagulant_options = self.knowledge_base['ingredients_base']['Coagulant']
            coagulant = random.choice(coagulant_options)
            # Dosage adapté selon quantité
            if "1L" in quantite_lait or "1 L" in quantite_lait:
                dosage_presure = "5 ml (ou 3 gouttes)"
            elif "2L" in quantite_lait or "2 L" in quantite_lait:
                dosage_presure = "10 ml (ou 6 gouttes)"
            else:
                dosage_presure = "Selon indications fabricant"
            ingredients_list.append(f"{dosage_presure} de {coagulant.lower()}")
        else:
            ingredients_list.append("5 ml de présure liquide")
        
        # Ferments depuis la base statique
        if 'ingredients_base' in self.knowledge_base and 'Ferments' in self.knowledge_base['ingredients_base']:
            ferments_options = self.knowledge_base['ingredients_base']['Ferments']
            ferment = random.choice(ferments_options)
            ingredients_list.append(f"2 g de ferments {ferment.lower()}")
        else:
            ingredients_list.append("2 g de ferments lactiques")
        
        # Sel depuis la base statique
        if 'ingredients_base' in self.knowledge_base and 'Sel' in self.knowledge_base['ingredients_base']:
            sel_options = self.knowledge_base['ingredients_base']['Sel']
            sel = random.choice(sel_options)
            ingredients_list.append(f"{profile_context['sel']} de {sel.lower()}")
        else:
            ingredients_list.append(f"{profile_context['sel']} de sel fin non iodé")
        
        # ========== AROMATES ET ÉPICES DEPUIS LA BASE ==========
        aromates = self._extract_aromates(ingredients)
        aromates_utilises = []
        
        if aromates and 'epices_et_aromates' in self.knowledge_base:
            for aromate in aromates:
                # Vérifier compatibilité avec type de fromage
                if self._check_aromate_compatibility(aromate, cheese_type, lait):
                    dosage = self._get_dosage_from_knowledge(aromate, quantite_lait)
                    ingredients_list.append(f"{dosage} de {aromate}")
                    aromates_utilises.append(aromate)
        
        # ========== ÉTAPES DÉTAILLÉES ==========
        etapes = self._generate_steps_from_knowledge(
            cheese_type,
            quantite_lait,
            type_info,
            profile_context,
            aromates_utilises,
            lait
        )
        
        # ========== TEMPÉRATURE ET CONDITIONS D'AFFINAGE ==========
        temp_affinage = self._get_temperature_affinage_from_knowledge(cheese_type)
        
        # ========== CONSEILS PERSONNALISÉS ==========
        conseils_sections = []
        
        # Conseils du profil
        conseils_sections.append(f"**{profile_context['conseil']}**")
        
        # Conseils spécifiques au type de fromage
        conseils_base = self._get_conseils_from_knowledge(cheese_type)
        if conseils_base:
            conseils_sections.append(f"\n**Spécificités du {cheese_type} :**\n{conseils_base}")
        
        # Problèmes courants
        problemes = self._get_problemes_courants_from_knowledge(cheese_type)
        if problemes:
            conseils_sections.append(f"\n**⚠️ Problèmes courants à éviter :**\n{problemes}")
        
        # Conservation
        conservation = self._get_conservation_from_knowledge(cheese_type)
        if conservation:
            conseils_sections.append(f"\n**📦 Conservation :**\n{conservation}")
        
        # Accords
        accords = self._get_accords_from_knowledge(cheese_type, lait)
        if accords:
            conseils_sections.append(f"\n**🍷 Accords recommandés :**\n{accords}")
        
        conseils = "\n".join(conseils_sections)
        
        # ========== MATÉRIEL NÉCESSAIRE ==========
        materiel = self._get_materiel_from_knowledge(profile, cheese_type)
        
        # ========== CONSTRUIRE LA RECETTE COMPLÈTE ==========
        recipe = {
            'title': title,
            'description': f"{type_info.get('description', f'Fromage {cheese_type.lower()}')} - {profile_context['description']}",
            'lait': lait or 'vache',
            'type_pate': cheese_type,
            'ingredients': ingredients_list,
            'etapes': etapes,
            'duree_totale': type_info.get('duree', profile_context['duree_totale']),
            'difficulte': type_info.get('difficulte', profile_context['difficulte']),
            'temperature_affinage': temp_affinage,
            'materiel_necessaire': materiel,
            'conseils': conseils,
            'aromates': aromates_utilises,
            'technique_aromatisation': self._get_technique_aromatisation(aromates_utilises, cheese_type) if aromates_utilises else None,
            'score': 7,
            'seed': seed,
            'profile': profile,
            'exemples_fromages': type_info.get('exemples', '')
        }
        
        print(f"   📝 Recette générée : {title}")
        print(f"   🧀 Type: {cheese_type} | Lait: {lait or 'vache'} | Profil: {profile}")
        
        return recipe

    def generate_recipe_pdf(self, recipe: Dict, output_path: str = None) -> str:
        """
        Génère un PDF professionnel de la recette de fromage
        
        Args:
            recipe: Dictionnaire contenant les données de la recette
            output_path: Chemin de sortie (optionnel)
        
        Returns:
            Chemin du fichier PDF généré
        """
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            PageBreak, Image, KeepTogether
        )
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        import os
        from datetime import datetime
        
        # Définir le chemin de sortie
        if output_path is None:
            safe_title = "".join(c for c in recipe['title'] if c.isalnum() or c in (' ', '-', '_')).strip()
            output_path = f"/mnt/user-data/outputs/Recette_{safe_title}_{datetime.now().strftime('%Y%m%d')}.pdf"
        
        # Créer le document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm,
            title=recipe['title'],
            author="Agent Fromager"
        )
        
        # Conteneur des éléments
        story = []
        
        # Styles
        styles = getSampleStyleSheet()
        
        # Style titre principal
        style_title = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2C5F2D'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        # Style sous-titre
        style_subtitle = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#666666'),
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica-Oblique'
        )
        
        # Style section
        style_section = ParagraphStyle(
            'SectionTitle',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2C5F2D'),
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold',
            borderPadding=5,
            borderColor=colors.HexColor('#2C5F2D'),
            borderWidth=0,
            leftIndent=0
        )
        
        # Style corps de texte
        style_body = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#333333'),
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            leading=16
        )
        
        # Style liste
        style_list = ParagraphStyle(
            'CustomList',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#333333'),
            leftIndent=20,
            spaceAfter=6,
            leading=14
        )
        
        # ========== EN-TÊTE ==========
        # Titre
        story.append(Paragraph(f"🧀 {recipe['title']}", style_title))
        
        # Description
        if recipe.get('description'):
            story.append(Paragraph(recipe['description'], style_subtitle))
        
        story.append(Spacer(1, 0.5*cm))
        
        # ========== INFORMATIONS CLÉS ==========
        info_data = [
            ['🥛 Type de lait', recipe.get('lait', 'Non spécifié').capitalize()],
            ['🧀 Catégorie', recipe.get('type_pate', 'Non spécifié')],
            ['⏱️ Durée totale', recipe.get('duree_totale', 'Variable')],
            ['📊 Difficulté', recipe.get('difficulte', 'Moyenne')],
            ['🌡️ Affinage', recipe.get('temperature_affinage', 'Selon type')],
        ]
        
        if recipe.get('profile'):
            info_data.append(['👤 Profil', recipe['profile']])
        
        info_table = Table(info_data, colWidths=[6*cm, 11*cm])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E8F5E9')),
            ('BACKGROUND', (1, 0), (1, -1), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#333333')),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ]))
        
        story.append(info_table)
        story.append(Spacer(1, 0.8*cm))
        
        # ========== MATÉRIEL NÉCESSAIRE ==========
        if recipe.get('materiel_necessaire'):
            story.append(Paragraph("🔧 Matériel nécessaire", style_section))
            
            for item in recipe['materiel_necessaire']:
                story.append(Paragraph(f"• {item}", style_list))
            
            story.append(Spacer(1, 0.5*cm))
        
        # ========== INGRÉDIENTS ==========
        story.append(Paragraph("🛒 Ingrédients", style_section))
        
        ingredients_data = [[Paragraph(f"<b>{ing}</b>", style_body)] for ing in recipe.get('ingredients', [])]
        
        ingredients_table = Table(ingredients_data, colWidths=[17*cm])
        ingredients_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FFF9E6')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#333333')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E6D8A3')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 15),
        ]))
        
        story.append(ingredients_table)
        story.append(Spacer(1, 0.8*cm))
        
        # ========== ÉTAPES DE FABRICATION ==========
        story.append(Paragraph("👨‍🍳 Étapes de fabrication", style_section))
        
        for i, etape in enumerate(recipe.get('etapes', []), 1):
            # Nettoyer les marqueurs markdown
            etape_clean = etape.replace('**', '').replace('*', '')
            
            # Créer un tableau pour chaque étape
            etape_data = [[
                Paragraph(f"<b>Étape {i}</b>", style_body),
                Paragraph(etape_clean, style_body)
            ]]
            
            etape_table = Table(etape_data, colWidths=[2.5*cm, 14.5*cm])
            etape_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, 0), colors.HexColor('#2C5F2D')),
                ('BACKGROUND', (1, 0), (1, 0), colors.HexColor('#F5F5F5')),
                ('TEXTCOLOR', (0, 0), (0, 0), colors.white),
                ('TEXTCOLOR', (1, 0), (1, 0), colors.HexColor('#333333')),
                ('ALIGN', (0, 0), (0, 0), 'CENTER'),
                ('ALIGN', (1, 0), (1, 0), 'LEFT'),
                ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (0, 0), 11),
                ('FONTSIZE', (1, 0), (1, 0), 10),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
                ('TOPPADDING', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ('LEFTPADDING', (1, 0), (1, 0), 12),
                ('RIGHTPADDING', (1, 0), (1, 0), 12),
            ]))
            
            story.append(etape_table)
            story.append(Spacer(1, 0.3*cm))
        
        story.append(Spacer(1, 0.5*cm))
        
        # ========== TECHNIQUE D'AROMATISATION ==========
        if recipe.get('technique_aromatisation') and recipe.get('aromates'):
            story.append(Paragraph("🌿 Aromatisation", style_section))
            
            aromates_text = ", ".join(recipe['aromates'])
            story.append(Paragraph(f"<b>Aromates utilisés :</b> {aromates_text}", style_body))
            story.append(Spacer(1, 0.2*cm))
            
            technique_clean = recipe['technique_aromatisation'].replace('**', '').replace('*', '')
            story.append(Paragraph(f"<b>Technique :</b> {technique_clean}", style_body))
            story.append(Spacer(1, 0.5*cm))
        
        # ========== CONSEILS ==========
        if recipe.get('conseils'):
            story.append(PageBreak())
            story.append(Paragraph("💡 Conseils et recommandations", style_section))
            
            conseils_clean = recipe['conseils'].replace('**', '<b>').replace('**', '</b>')
            conseils_paragraphs = conseils_clean.split('\n\n')
            
            for para in conseils_paragraphs:
                if para.strip():
                    # Gérer les listes à puces
                    if para.strip().startswith('•') or para.strip().startswith('-'):
                        lines = para.split('\n')
                        for line in lines:
                            if line.strip():
                                story.append(Paragraph(line.strip(), style_list))
                    else:
                        story.append(Paragraph(para.strip(), style_body))
                    story.append(Spacer(1, 0.3*cm))
        
        # ========== EXEMPLES DE FROMAGES ==========
        if recipe.get('exemples_fromages'):
            story.append(Spacer(1, 0.5*cm))
            story.append(Paragraph("🧀 Exemples de fromages de cette catégorie", style_section))
            story.append(Paragraph(recipe['exemples_fromages'], style_body))
        
        # ========== PIED DE PAGE ==========
        story.append(Spacer(1, 1*cm))
        
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#999999'),
            alignment=TA_CENTER,
            spaceAfter=5
        )
        
        story.append(Paragraph("─" * 80, footer_style))
        story.append(Paragraph(
            f"📅 Recette générée le {datetime.now().strftime('%d/%m/%Y à %H:%M')} par <b>Agent Fromager</b>",
            footer_style
        ))
        story.append(Paragraph(
            "🧀 Fromagerie artisanale et transmission du savoir-faire fromager",
            footer_style
        ))
        
        if recipe.get('seed'):
            story.append(Paragraph(
                f"<i>Seed de recette : {recipe['seed']}</i>",
                footer_style
            ))
        
        # ========== GÉNÉRER LE PDF ==========
        try:
            doc.build(story)
            print(f"✅ PDF généré avec succès : {output_path}")
            return output_path
        except Exception as e:
            print(f"❌ Erreur lors de la génération du PDF : {e}")
            raise


    def batch_generate_pdfs(self, recipes: List[Dict], output_dir: str = "/mnt/user-data/outputs") -> List[str]:
        """
        Génère des PDFs pour plusieurs recettes
        
        Args:
            recipes: Liste de dictionnaires de recettes
            output_dir: Répertoire de sortie
        
        Returns:
            Liste des chemins des PDFs générés
        """
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        pdf_paths = []
        
        for i, recipe in enumerate(recipes, 1):
            print(f"📄 Génération PDF {i}/{len(recipes)} : {recipe['title']}")
            
            try:
                safe_title = "".join(c for c in recipe['title'] if c.isalnum() or c in (' ', '-', '_')).strip()
                output_path = os.path.join(output_dir, f"Recette_{i:02d}_{safe_title}.pdf")
                
                pdf_path = self.generate_recipe_pdf(recipe, output_path)
                pdf_paths.append(pdf_path)
                
            except Exception as e:
                print(f"⚠️ Échec pour '{recipe['title']}' : {e}")
                continue
        
        print(f"\n✅ {len(pdf_paths)}/{len(recipes)} PDFs générés avec succès")
        return pdf_paths


    # ========== FONCTION D'EXPORT AVEC GÉNÉRATION PDF ==========
    def export_recipe_with_pdf(self, recipe: Dict, format: str = 'both') -> Dict[str, str]:
        """
        Exporte une recette en JSON et/ou PDF
        
        Args:
            recipe: Dictionnaire de la recette
            format: 'json', 'pdf', ou 'both'
        
        Returns:
            Dictionnaire avec les chemins des fichiers générés
        """
        import json
        from datetime import datetime
        
        safe_title = "".join(c for c in recipe['title'] if c.isalnum() or c in (' ', '-', '_')).strip()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        outputs = {}
        
        # Export JSON
        if format in ['json', 'both']:
            json_path = f"/mnt/user-data/outputs/Recette_{safe_title}_{timestamp}.json"
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(recipe, f, ensure_ascii=False, indent=2)
            
            outputs['json'] = json_path
            print(f"✅ JSON exporté : {json_path}")
        
        # Export PDF
        if format in ['pdf', 'both']:
            pdf_path = f"/mnt/user-data/outputs/Recette_{safe_title}_{timestamp}.pdf"
            
            try:
                self.generate_recipe_pdf(recipe, pdf_path)
                outputs['pdf'] = pdf_path
            except Exception as e:
                print(f"❌ Erreur PDF : {e}")
        
        return outputs

    # ========== MÉTHODES AUXILIAIRES ==========

    def _get_type_info_from_knowledge(self, cheese_type: str) -> Dict:
        """Récupère les infos d'un type de fromage depuis la base"""
        if 'types_pate' in self.knowledge_base:
            return self.knowledge_base['types_pate'].get(cheese_type, {})
        return {}


    def _check_aromate_compatibility(self, aromate: str, cheese_type: str, lait: Optional[str]) -> bool:
        """Vérifie la compatibilité aromate/fromage depuis regles_compatibilite"""
        if 'regles_compatibilite' not in self.knowledge_base:
            return True
        
        # Vérifier exclusions absolues
        if 'exclusions_absolues' in self.knowledge_base['regles_compatibilite']:
            for exclusion in self.knowledge_base['regles_compatibilite']['exclusions_absolues']:
                if f"type_pate:{cheese_type}" in exclusion['combinaison'] and aromate.lower() in exclusion['combinaison'].lower():
                    print(f"   ⚠️ Exclusion : {aromate} incompatible avec {cheese_type}")
                    return False
        
        # Vérifier compatibilité type_pate x aromates
        if 'type_pate_x_aromates' in self.knowledge_base['regles_compatibilite']:
            if cheese_type in self.knowledge_base['regles_compatibilite']['type_pate_x_aromates']:
                infos = self.knowledge_base['regles_compatibilite']['type_pate_x_aromates'][cheese_type]
                
                # Vérifier incompatibilités
                if 'aromates_incompatibles' in infos:
                    for incompatible in infos['aromates_incompatibles']:
                        if incompatible.lower() in aromate.lower():
                            print(f"   ⚠️ {aromate} déconseillé pour {cheese_type}")
                            return False
        
        return True


    def _get_dosage_from_knowledge(self, aromate: str, quantite_lait: str = "1L") -> str:
        """
        Récupère le dosage recommandé depuis la base de connaissances
        
        Args:
            aromate: Nom de l'aromate/épice
            quantite_lait: Quantité de lait utilisée (ex: "1L", "2L", "10L")
        
        Returns:
            Dosage recommandé avec unité
        """
        if 'dosages_recommandes' not in self.knowledge_base:
            return "1 cuillère à café"
        
        dosages = self.knowledge_base['dosages_recommandes']
        aromate_lower = aromate.lower()
        
        # Extraire le coefficient multiplicateur selon quantité de lait
        coef = 1.0
        try:
            # Chercher un nombre suivi de L ou l
            import re
            match = re.search(r'(\d+(?:\.\d+)?)\s*[Ll]', quantite_lait)
            if match:
                coef = float(match.group(1))
        except:
            coef = 1.0
        
        # Identifier la catégorie et appliquer le dosage
        
        # Herbes fraîches
        if any(herb in aromate_lower for herb in ['basilic', 'thym', 'romarin', 'persil', 'menthe', 'ciboulette', 'aneth', 'coriandre']):
            if 'frais' in aromate_lower or 'fraîche' in aromate_lower:
                base = dosages.get('Herbes fraîches', "2-3 cuillères à soupe")
                if coef > 1:
                    # Extraire les nombres du dosage
                    try:
                        nums = re.findall(r'\d+', base)
                        if len(nums) >= 2:
                            min_val = int(nums[0]) * coef
                            max_val = int(nums[1]) * coef
                            return f"{int(min_val)}-{int(max_val)} cuillères à soupe"
                    except:
                        pass
                return base
            else:
                # Herbes séchées
                base = dosages.get('Herbes séchées', "1-2 cuillères à soupe")
                if coef > 1:
                    try:
                        nums = re.findall(r'\d+', base)
                        if len(nums) >= 2:
                            min_val = int(nums[0]) * coef
                            max_val = int(nums[1]) * coef
                            return f"{int(min_val)}-{int(max_val)} cuillères à soupe"
                    except:
                        pass
                return base
        
        # Épices moulues
        elif any(spice in aromate_lower for spice in ['poivre', 'paprika', 'curry', 'cumin', 'piment', 'cayenne', 'espelette']):
            base = dosages.get('Épices moulues', "1-2 cuillères à café")
            if coef > 1:
                try:
                    nums = re.findall(r'\d+', base)
                    if len(nums) >= 2:
                        min_val = int(nums[0]) * coef
                        max_val = int(nums[1]) * coef
                        return f"{int(min_val)}-{int(max_val)} cuillères à café"
                except:
                    pass
            return base
        
        # Graines et épices en grains
        elif any(grain in aromate_lower for grain in ['graines', 'grain', 'fenouil', 'carvi', 'nigelle', 'coriandre en graines']):
            base = dosages.get('Épices en grains', "1 cuillère à soupe concassée")
            if coef > 1:
                return f"{int(coef)} cuillères à soupe concassées"
            return base
        
        # Ail
        elif 'ail' in aromate_lower:
            if coef > 1:
                min_val = int(1 * coef)
                max_val = int(2 * coef)
                return f"{min_val}-{max_val} gousses"
            return "1-2 gousses"
        
        # Gingembre
        elif 'gingembre' in aromate_lower:
            if coef > 1:
                min_val = int(1 * coef)
                max_val = int(2 * coef)
                return f"{min_val}-{max_val} morceaux de 2cm"
            return "1-2 morceaux de 2cm"
        
        # Zestes
        elif any(zest in aromate_lower for zest in ['zeste', 'citron', 'orange', 'bergamote', 'lime']):
            if coef > 1:
                return f"{int(coef)} agrume(s) entier(s)"
            return "1 agrume entier"
        
        # Cendres
        elif 'cendre' in aromate_lower or 'charbon' in aromate_lower:
            return "Fine couche sur la croûte"
        
        # Noix, noisettes, pistaches
        elif any(nut in aromate_lower for nut in ['noix', 'noisette', 'pistache', 'amande']):
            if coef > 1:
                return f"{int(30 * coef)}g concassées"
            return "30g concassées (2-3 cuillères à soupe)"
        
        # Olives
        elif 'olive' in aromate_lower:
            if coef > 1:
                return f"{int(50 * coef)}g dénoyautées et coupées"
            return "50g dénoyautées et coupées"
        
        # Tomates séchées
        elif 'tomate' in aromate_lower and ('séchée' in aromate_lower or 'sechee' in aromate_lower):
            if coef > 1:
                return f"{int(30 * coef)}g hachées"
            return "30g hachées"
        
        # Fruits secs
        elif any(fruit in aromate_lower for fruit in ['abricot', 'figue', 'raisin', 'datte', 'pruneau']):
            if coef > 1:
                return f"{int(40 * coef)}g hachés"
            return "40g hachés (environ 4-5 pièces)"
        
        # Fleurs et pollen
        elif any(fleur in aromate_lower for fleur in ['lavande', 'safran', 'rose', 'bleuet', 'pollen']):
            if 'safran' in aromate_lower:
                if coef > 1:
                    return f"{int(0.2 * coef * 10) / 10}g (quelques pistils)"
                return "0.2g (quelques pistils)"
            else:
                if coef > 1:
                    return f"{int(1 * coef)} cuillère à café"
                return "1 cuillère à café"
        
        # Truffe
        elif 'truffe' in aromate_lower:
            if coef > 1:
                return f"{int(10 * coef)}g râpée"
            return "10g râpée (environ 1 petite truffe)"
        
        # Champignons séchés
        elif 'champignon' in aromate_lower and 'séché' in aromate_lower:
            if coef > 1:
                return f"{int(20 * coef)}g réhydratés et hachés"
            return "20g réhydratés et hachés"
        
        # Dosage par défaut
        return "Selon goût (1-2 cuillères à café)"

    def _generate_steps_from_knowledge(
        self,
        cheese_type: str,
        quantite_lait: str,
        type_info: Dict,
        profile_context: Dict,
        aromates: List[str],
        lait: Optional[str]
    ) -> List[str]:
        """Génère les étapes de fabrication adaptées au profil"""
        
        etapes = []
        
        # Intro adaptée au profil
        ton = profile_context.get('ton', 'Encourageant')
        
        if ton == 'Encourageant, pédagogique, rassurant' or ton == 'Encourageant':
            etapes.append("**🌟 Préparation (pas de panique !)** : Rassemblez tout votre matériel et vos ingrédients. Lisez la recette en entier avant de commencer - c'est le secret d'une première fois réussie !")
        elif ton == 'Technique, précis, professionnel' or ton == 'Technique':
            etapes.append("**📋 Mise en place** : Préparer et peser tous les ingrédients. Stériliser le matériel à l'eau bouillante. Vérifier la température ambiante (20-22°C optimal).")
        else:
            etapes.append("**🎓 Préparation pédagogique** : Avant de commencer avec vos apprenants, vérifiez que chaque poste dispose du matériel nécessaire. Préparez vos supports visuels.")
        
        # Chauffage du lait
        if "Fromage frais" in cheese_type:
            temp_cible = "35-37°C"
        elif "Pâte pressée cuite" in cheese_type:
            temp_cible = "52-54°C"
        else:
            temp_cible = "30-32°C"
        
        etapes.append(f"**🌡️ Chauffage du lait** : Versez le lait dans une grande casserole. Chauffez doucement à feu moyen en remuant régulièrement jusqu'à atteindre {temp_cible}. Utilisez un thermomètre - la précision est importante !")
        
        # Ajout ferments
        etapes.append("**🦠 Ajout des ferments** : Retirez du feu. Saupoudrez les ferments à la surface, attendez 2 minutes qu'ils se réhydratent, puis mélangez délicatement. Laissez reposer 30-45 minutes à température ambiante (cette étape s'appelle la maturation).")
        
        # Emprésurage
        etapes.append(f"**💧 Emprésurage** : Diluez la présure dans 2 cuillères à soupe d'eau froide. Versez dans le lait en mélangeant doucement pendant 30 secondes. Couvrez et laissez reposer {self._get_caillage_time(cheese_type)} sans bouger la casserole.")
        
        # Découpe du caillé
        if "Fromage frais" in cheese_type:
            etapes.append("**🔪 Test du caillé** : Le caillé est prêt quand il se détache net sur les bords. Versez délicatement dans une étamine ou un moule perforé pour l'égouttage.")
        else:
            etapes.append("**🔪 Découpe du caillé** : Vérifiez la 'cassure nette' (le caillé doit se fendre proprement). Découpez en cubes de 1-2 cm avec un couteau long en faisant des lignes verticales puis horizontales.")
        
        # Brassage si nécessaire
        if "pressée" in cheese_type.lower():
            etapes.append("**🌀 Brassage et chauffage** : Remuez doucement les cubes de caillé pendant 10-15 minutes en chauffant progressivement à 38-40°C. Le petit-lait va se séparer, les grains vont se raffermir.")
        
        # Moulage avec aromates
        if aromates:
            technique = self._get_technique_aromatisation(aromates, cheese_type)
            etapes.append(f"**🌿 Moulage avec aromates** : {technique}. Versez le caillé dans le moule en tassant légèrement.")
        else:
            etapes.append("**🧈 Moulage** : Transférez le caillé dans le(s) moule(s) perforé(s). Tassez légèrement avec le dos d'une cuillère.")
        
        # Égouttage
        egouttage_time = self._get_egouttage_time(cheese_type, profile_context)
        etapes.append(f"**💧 Égouttage** : Laissez égoutter {egouttage_time} en retournant le fromage {self._get_retournement(cheese_type)}. Le petit-lait va s'écouler naturellement.")
        
        # Salage
        etapes.append(f"**🧂 Salage** : {self._get_salage_method(cheese_type)}. Le sel parfume et favorise la formation de la croûte.")
        
        # Affinage
        temp_affinage = self._get_temperature_affinage_from_knowledge(cheese_type)
        if "Fromage frais" in cheese_type:
            etapes.append(f"**❄️ Conservation** : Votre fromage frais est prêt ! Conservez-le au réfrigérateur dans une boîte hermétique et consommez sous 3-5 jours.")
        else:
            etapes.append(f"**🏺 Affinage** : Placez le fromage dans votre cave d'affinage ou une pièce fraîche à {temp_affinage}. Retournez-le tous les 2 jours. {self._get_affinage_specifics(cheese_type)}")
        
        return etapes

    def _get_caillage_time(self, cheese_type: str) -> str:
        """Temps de caillage selon type"""
        times = {
            "Fromage frais": "45 minutes à 1h",
            "Pâte molle": "1h à 1h30",
            "Pâte pressée non cuite": "30-45 minutes",
            "Pâte pressée cuite": "30-40 minutes",
            "Pâte persillée": "1h30 à 2h"
        }
        return times.get(cheese_type, "1 heure")


    def _get_egouttage_time(self, cheese_type: str, profile_context: Dict) -> str:
        """Temps d'égouttage adapté"""
        if "Fromage frais" in cheese_type:
            return "4-6 heures au frais" if profile_context['niveau'] == 'débutant' else "6-12 heures"
        elif "Pâte molle" in cheese_type:
            return "12-18 heures à température ambiante"
        else:
            return "6-8 heures avec poids de 500g-1kg"


    def _get_retournement(self, cheese_type: str) -> str:
        """Fréquence de retournement"""
        if "Fromage frais" in cheese_type:
            return "pas nécessaire"
        elif "Pâte molle" in cheese_type:
            return "toutes les 6 heures"
        else:
            return "toutes les 2-3 heures"


    def _get_salage_method(self, cheese_type: str) -> str:
        """Méthode de salage"""
        if "Fromage frais" in cheese_type:
            return "Saupoudrez de sel fin sur toutes les faces, ou mélangez directement dans la pâte"
        elif "persillée" in cheese_type.lower():
            return "Frottez toutes les faces avec du gros sel, puis bain de saumure 24h"
        else:
            return "Frottez généreusement toutes les faces avec du sel fin ou gros sel"


    def _get_affinage_specifics(self, cheese_type: str) -> str:
        """Spécificités d'affinage"""
        specs = {
            "Pâte molle": "Une croûte blanche fleurie va apparaître après 5-7 jours. Durée totale : 2-4 semaines.",
            "Pâte pressée non cuite": "La croûte va se former et durcir. Frottez-la avec un linge humide chaque semaine. Durée : 1-3 mois minimum.",
            "Pâte pressée cuite": "Patience ! L'affinage peut durer 3-12 mois selon le résultat souhaité.",
            "Pâte persillée": "Les veines bleues vont se développer après 2-3 semaines. Piquez avec une aiguille stérile pour favoriser l'aération."
        }
        return specs.get(cheese_type, "Suivez l'évolution de votre fromage semaine après semaine.")


    def _get_temperature_affinage_from_knowledge(self, cheese_type: str) -> str:
        """Récupère température d'affinage depuis la base"""
        if 'temperatures_affinage' in self.knowledge_base:
            return self.knowledge_base['temperatures_affinage'].get(cheese_type, "10-14°C, 85% humidité")
        return "10-14°C, 85% humidité"


    def _get_technique_aromatisation(self, aromates: List[str], cheese_type: str) -> str:
        """Récupère la meilleure technique d'aromatisation"""
        if 'techniques_aromatisation' not in self.knowledge_base:
            return "Incorporez les aromates au moment du moulage"
        
        techniques = self.knowledge_base['techniques_aromatisation']
        
        if "Fromage frais" in cheese_type:
            return techniques.get('Incorporation dans le caillé', '') + " - mélangez délicatement les herbes fraîches dans le caillé égoutté"
        elif any('cendr' in a.lower() for a in aromates):
            return techniques.get('Enrobage externe', '') + " - roulez le fromage démoulé dans les cendres"
        else:
            return techniques.get('Couche intermédiaire', '') + " - créez des strates aromates/caillé dans le moule"


    def _get_conseils_from_knowledge(self, cheese_type: str) -> str:
        """Récupère conseils spécifiques au type"""
        if 'types_pate' in self.knowledge_base and cheese_type in self.knowledge_base['types_pate']:
            info = self.knowledge_base['types_pate'][cheese_type]
            return f"Difficulté : {info.get('difficulte', '')}. Durée d'affinage typique : {info.get('duree', '')}."
        return ""


    def _get_problemes_courants_from_knowledge(self, cheese_type: str) -> str:
        """Sélectionne 2-3 problèmes pertinents"""
        if 'problemes_courants' not in self.knowledge_base:
            return ""
        
        problemes = self.knowledge_base['problemes_courants']
        
        # Sélection contextuelle
        if "Fromage frais" in cheese_type:
            keys = ['Caillé trop dur', 'Pas de caillage', 'Fromage trop acide']
        elif "Pâte molle" in cheese_type:
            keys = ['Moisissures indésirables', 'Croûte craquelée', 'Fromage coule']
        elif "pressée" in cheese_type.lower():
            keys = ['Texture granuleuse', 'Fromage trop sec', 'Yeux (trous) non désirés']
        else:
            keys = list(problemes.keys())[:3]
        
        result = []
        for key in keys:
            if key in problemes:
                result.append(f"• {key} : {problemes[key]}")
        
        return "\n".join(result)


    def _get_conservation_from_knowledge(self, cheese_type: str) -> str:
        """Récupère infos de conservation"""
        if 'conservation' in self.knowledge_base:
            for key, value in self.knowledge_base['conservation'].items():
                if cheese_type in key:
                    return value
        return "Conservez au frais dans du papier sulfurisé ou une boîte hermétique."


    def _get_accords_from_knowledge(self, cheese_type: str, lait: Optional[str]) -> str:
        """Récupère accords vins et mets"""
        accords = []
        
        # Accords vins
        if 'accords_vins' in self.knowledge_base:
            # Chercher par type ou par lait
            for key, value in self.knowledge_base['accords_vins'].items():
                if cheese_type in key or (lait and lait.lower() in key.lower()):
                    accords.append(f"🍷 {value}")
                    break
        
        # Accords mets
        if 'accords_mets' in self.knowledge_base:
            for key, value in self.knowledge_base['accords_mets'].items():
                if cheese_type in key:
                    accords.append(f"🍽️ {value}")
                    break
        
        return "\n".join(accords) if accords else ""


    def _get_materiel_from_knowledge(self, profile: str, cheese_type: str) -> List[str]:
        """Liste du matériel nécessaire selon profil"""
        if 'materiel_indispensable' not in self.knowledge_base:
            return []
        
        materiel = self.knowledge_base['materiel_indispensable']
        
        if profile == "🧀 Amateur":
            return materiel.get('Pour débuter', [])
        elif profile == "🏭 Producteur":
            return materiel.get('Pour expert', [])
        else:
            return materiel.get('Pour progresser', [])


    def _extract_aromates(self, ingredients: List[str]) -> List[str]:
        """Extrait les aromates de la liste d'ingrédients"""
        aromates = []
        
        # Liste des mots-clés d'aromates
        aromates_keywords = [
            'thym', 'romarin', 'basilic', 'menthe', 'persil', 'ciboulette', 'aneth',
            'poivre', 'paprika', 'curry', 'cumin', 'piment', 'ail', 'herbes',
            'lavande', 'noix', 'olive', 'tomate séchée', 'cendre', 'truffe'
        ]
        
        for ing in ingredients:
            ing_lower = ing.lower()
            for keyword in aromates_keywords:
                if keyword in ing_lower:
                    aromates.append(ing)
                    break
        
        return aromates


    def _get_profile_context(self, profile: str) -> Dict:
        """Récupère le contexte du profil depuis la base"""
        if 'profils_utilisateurs' in self.knowledge_base:
            profile_data = self.knowledge_base['profils_utilisateurs'].get(profile, {})
            
            # Adapter les quantités selon profil
            if profile == "🧀 Amateur":
                quantite = "1L"
                sel = "1 cuillère à café"
                duree = "24-48h"
                difficulte = "Facile"
                conseil = "Prenez votre temps et suivez chaque étape tranquillement !"
                description = profile_data.get('description', 'Débutant, usage familial, matériel limité')  # ✅
            elif profile == "🏭 Producteur":
                quantite = "10L"
                sel = "2% du poids total"
                duree = "Selon cahier des charges"
                difficulte = "Technique"
                conseil = "Documentez température, pH et temps à chaque étape."
                description = profile_data.get('description', 'Professionnel ou semi-pro, recherche de qualité')  # ✅
            else:  # Formateur
                quantite = "2L"
                sel = "15g"
                duree = "Variable selon session"
                difficulte = "Moyenne"
                conseil = "Préparez des échantillons à différents stades pour la démonstration."
                description = profile_data.get('description', 'Enseignant, animateur, partage de savoir')  # ✅
            
            return {
                'quantite_lait': quantite,
                'sel': sel,
                'duree_totale': duree,
                'difficulte': difficulte,
                'conseil': conseil,
                'ton': profile_data.get('ton', 'neutre'),
                'niveau': profile_data.get('niveau', 'intermédiaire'),
                'description': description  # ✅ Toujours une vraie description
            }
        
        # Fallback
        return {
            'quantite_lait': '1L',
            'sel': '1 cuillère à café',
            'duree_totale': '48h',
            'difficulte': 'Facile',
            'conseil': 'Suivez les étapes avec attention !',
            'ton': 'Encourageant',
            'niveau': 'débutant',
            'description': 'Recette adaptée pour débutants'  # ✅ Fallback avec vraie description
        }
    
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
        """Génère avec LLM en utilisant le contexte complet de la base statique"""
        
        
        
        print("=" * 80)
        print("🚨 FONCTION _generate_with_llm_and_knowledge() APPELÉE !")
        print(f"🚨 Ingrédients reçus : {ingredients}")
        print(f"🚨 Type de fromage : {cheese_type}")
        print("=" * 80)
        
        import json
        import time
        import random
        
        seed = int(time.time() * 1000 + random.randint(1, 999))
        
        # ========== RÉCUPÉRER LE CONTEXTE DEPUIS LA BASE STATIQUE ==========
        type_info = self._get_type_info_from_knowledge(cheese_type)
        aromates = self._extract_aromates(ingredients)
        profile_context = self._get_profile_context(profile)
        
        # ========== CONSTRUIRE UN CONTEXTE ENRICHI POUR LE LLM ==========
        knowledge_context = f"""
    **📚 CONTEXTE DEPUIS LA BASE DE CONNAISSANCES:**

    **Type de fromage : {cheese_type}**
    - Description : {type_info.get('description', 'N/A')}
    - Exemples similaires : {type_info.get('exemples', 'N/A')}
    - Durée typique d'affinage : {type_info.get('duree', 'N/A')}
    - Niveau de difficulté : {type_info.get('difficulte', 'N/A')}

    **Conditions d'affinage recommandées :**
    {self._get_temperature_affinage_from_knowledge(cheese_type)}

    **Profil utilisateur : {profile}**
    - Niveau : {profile_context.get('niveau', 'intermédiaire')}
    - Ton à adopter : {profile_context.get('ton', 'neutre')}
    - Quantité de lait : {profile_context.get('quantite_lait', '1L')}
    - Description : {profile_context.get('description', '')}
    """

        # Ajouter les aromates et dosages
        if aromates:
            knowledge_context += f"\n**🌿 Aromates détectés : {', '.join(aromates)}**\n"
            knowledge_context += "\n**⚠️ IMPORTANT : Utilise UNIQUEMENT ces aromates, n'en ajoute pas d'autres !**\n"
            
            if 'dosages_recommandes' in self.knowledge_base:
                knowledge_context += "\n**Dosages recommandés pour 1kg de fromage :**\n"
                for aromate in aromates:
                    dosage = self._get_dosage_from_knowledge(aromate, profile_context.get('quantite_lait', '1L'))
                    knowledge_context += f"- {aromate} : {dosage}\n"
            
            # Vérifier compatibilités
            if 'regles_compatibilite' in self.knowledge_base:
                knowledge_context += "\n**⚠️ Règles de compatibilité :**\n"
                for aromate in aromates:
                    if not self._check_aromate_compatibility(aromate, cheese_type, lait):
                        knowledge_context += f"- ⚠️ {aromate} peut être incompatible avec {cheese_type}\n"
            
            # Techniques d'aromatisation
            if 'techniques_aromatisation' in self.knowledge_base:
                technique = self._get_technique_aromatisation(aromates, cheese_type)
                knowledge_context += f"\n**Technique d'aromatisation suggérée :**\n{technique}\n"
        else:
            knowledge_context += "\n**Aromates : Aucun aromate spécifié**\n"
        
        # Ajouter les problèmes courants à anticiper
        if 'problemes_courants' in self.knowledge_base:
            problemes = self._get_problemes_courants_from_knowledge(cheese_type)
            if problemes:
                knowledge_context += f"\n**⚠️ Problèmes courants à anticiper :**\n{problemes}\n"
        
        # Ajouter les associations classiques
        if 'associations_classiques' in self.knowledge_base:
            assoc_key = None
            if lait and lait.lower() in ['chèvre', 'chevre']:
                assoc_key = 'Fromage de chèvre'
            elif lait and lait.lower() == 'brebis':
                assoc_key = 'Brebis'
            elif 'molle' in cheese_type.lower():
                assoc_key = 'Pâte molle'
            elif 'pressée' in cheese_type.lower():
                assoc_key = 'Pâte pressée'
            elif 'persillée' in cheese_type.lower() or 'bleu' in cheese_type.lower():
                assoc_key = 'Pâte persillée'
            elif 'frais' in cheese_type.lower():
                assoc_key = 'Fromage frais'
            
            if assoc_key and assoc_key in self.knowledge_base['associations_classiques']:
                assoc = self.knowledge_base['associations_classiques'][assoc_key]
                knowledge_context += f"\n**🎨 Associations classiques pour ce type :**\n{assoc}\n"
        
        # Ajouter conservation et accords
        conservation = self._get_conservation_from_knowledge(cheese_type)
        if conservation:
            knowledge_context += f"\n**📦 Conservation :**\n{conservation}\n"
        
        accords = self._get_accords_from_knowledge(cheese_type, lait)
        if accords:
            knowledge_context += f"\n**🍷 Accords recommandés :**\n{accords}\n"
        
        # Ajouter le matériel nécessaire
        materiel = self._get_materiel_from_knowledge(profile, cheese_type)
        if materiel:
            knowledge_context += f"\n**🔧 Matériel nécessaire pour ce profil :**\n"
            for item in materiel[:5]:  # Limiter à 5 pour ne pas surcharger
                knowledge_context += f"- {item}\n"
        
        # ========== CONSTRUIRE LE PROMPT ==========
        prompt = f"""Tu es un maître fromager expert avec des décennies d'expérience. Génère UNE recette UNIQUE au format JSON STRICT et ULTRA-DÉTAILLÉE.

INTERDICTIONS ABSOLUES:
❌ PAS de texte explicatif avant le JSON
❌ PAS de markdown (pas de ```)
❌ PAS de commentaires
❌ PAS de titres ou sections
✅ COMMENCE DIRECTEMENT PAR {{
✅ TERMINE DIRECTEMENT PAR }}
    
INGRÉDIENTS DISPONIBLES: {', '.join(ingredients)}
TYPE DE LAIT: {lait or "vache"}
TYPE DE FROMAGE: {cheese_type}
AROMATES: {', '.join(aromates) if aromates else "AUCUN"}
PROFIL: {profile}

{knowledge_context[:1500] if knowledge_context else ""}

RÈGLES JSON ABSOLUES (NON-NÉGOCIABLES):
1. ✅ JSON VALIDE uniquement - commence par {{ et termine par }}
2. ✅ Chaque accolade ouvrante {{ DOIT avoir sa fermante }}
3. ✅ Chaque crochet ouvrant [ DOIT avoir son fermant ]
4. ✅ Virgules ENTRE les éléments, JAMAIS avant ] ou }}
5. ✅ Guillemets doubles " pour TOUTES les clés et valeurs string
6. ✅ Pas de virgule après le dernier élément d'un tableau ou objet
7. ✅ N'utilise QUE les aromates listés ci-dessus
8. ✅ Inclus obligatoirement: présure, ferments lactiques, sel
9. ✅ Minimum 6 étapes détaillées
10. ✅ Structure SIMPLE et PLATE - pas d'objets imbriqués complexes
11. ⚠️ AUCUN astérisque * dans le JSON (pas de markdown italique !)
12. ⚠️ AUCUN underscore _ pour le markdown (pas de __gras__)
13. ⚠️ Texte brut uniquement dans les strings

EXIGENCE DE LONGUEUR OPTIMALE:
- Chaque étape doit contenir 150-250 caractères (pas plus !)
- Description du fromage: 100-150 caractères
- Conseils: 200-300 caractères
- TOTAL VISÉ: 4000-6000 caractères (pas plus de 8000)

⚠️ IMPÉRATIF: FERME TOUTES LES ACCOLADES ET CROCHETS !
⚠️ VÉRIFIE que ton JSON se termine par }} avant d'envoyer
⚠️ PAS d'objets imbriqués comme {{"type": "...", "origine": "..."}} !

⚠️⚠️⚠️ RÈGLES ANTI-MARKDOWN (CRITIQUE) ⚠️⚠️⚠️
- INTERDICTION ABSOLUE d'utiliser * (astérisque) dans le JSON
- INTERDICTION ABSOLUE d'utiliser _ (underscore) dans le JSON  
- INTERDICTION ABSOLUE de tout formatage markdown
- Si tu veux mettre en valeur un mot, utilise MAJUSCULES, pas markdown
- Exemple CORRECT: "ferments lactiques Lactococcus lactis"
- Exemple INTERDIT: "ferments lactiques *Lactococcus lactis*"

Le JSON doit contenir UNIQUEMENT du texte brut, des virgules, des accolades, des crochets et des guillemets.
AUCUN autre caractère spécial de formatage n'est autorisé.

FORMAT JSON ULTRA-SIMPLIFIÉ (COPIE EXACTEMENT CETTE STRUCTURE):

{{
    "title": "Nom du fromage",
    "description": "Description en une phrase courte",
    "lait": "Type de lait et température",
    "type_pate": "Type de pâte",
    "ingredients": [
        "1L de lait entier",
        "5ml de presure liquide",
        "2g de ferments lactiques",
        "10g de sel fin",
        "Aromates doses precises"
    ],
    "materiel": [
        "thermometre",
        "casserole",
        "moule",
        "etamine"
    ],
    "etapes": [
        "Etape 1 - Steriliser tout le materiel 10 minutes dans eau bouillante",
        "Etape 2 - Chauffer lait a 32 degres en remuant doucement",
        "Etape 3 - Ajouter ferments lactiques et melanger 2 minutes",
        "Etape 4 - Ajouter presure diluee et attendre 45 minutes",
        "Etape 5 - Decouper caille en cubes de 2cm et brasser",
        "Etape 6 - Mouler et egoutter 12 heures"
    ],
    "duree_totale": "24-48h",
    "difficulte": "Moyenne",
    "temperature_affinage": "12-14 degres avec 85% humidite",
    "conseils": "Conseils pratiques en une ou deux phrases courtes",
    "aromates": {json.dumps(aromates, ensure_ascii=False)},
    "technique_aromatisation": "Technique incorporation aromates",
    "score": 8.0,
    "seed": {seed},
    "profile": "{profile}"
}}

REGLES ULTRA-STRICTES:
- PAS d'accents dans le JSON (utilise e au lieu de é, a au lieu de à)
- PAS de caracteres speciaux (* _ # etc)
- PAS d'apostrophes (utilise espaces)
- Texte simple sans formatage
- Maximum 150 caracteres par etape
- Virgules ENTRE elements, JAMAIS avant ] ou }}

⚠️ INSTRUCTIONS FINALES CRITIQUES:
- COMMENCE DIRECTEMENT PAR LA PREMIÈRE ACCOLADE {{
- TERMINE DIRECTEMENT PAR LA DERNIÈRE ACCOLADE }}
- AUCUN texte avant ou après
- AUCUN formatage markdown (* pour italique, ** pour gras, _ pour souligné)
- Texte brut UNIQUEMENT dans toutes les valeurs
- VALIDE ton JSON mentalement avant d'envoyer
- Structure PLATE uniquement (pas d'objets dans les objets)

GÉNÈRE MAINTENANT LE JSON COMPLET ET ULTRA-DÉTAILLÉ:"""

        # ========== APPEL AU LLM ==========
        try:
            print("🔍 DEBUG: Envoi du prompt au LLM...")
            print(f"🔍 Longueur du prompt: {len(prompt)} caractères")
            
            
            response = self.agent.chat_with_llm(
                prompt,
                max_tokens=8192,  # Augmentez cette valeur si nécessaire
                temperature=0.8
            )
            print(f"🔍 DEBUG: Réponse LLM reçue ({len(response)} caractères)")
            print(f"🔍 DEBUG: Premiers 500 caractères: {response[:500]}")
            
           
            # Nettoyage de la réponse
            cleaned = response.strip()
            
            # Retirer les blocs markdown
            if cleaned.startswith('```'):
                first_newline = cleaned.find('\n')
                if first_newline != -1:
                    cleaned = cleaned[first_newline + 1:]
                else:
                    cleaned = cleaned[3:]
            
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3].rstrip()
            
            cleaned = cleaned.strip()
            
            print(f"🔍 DEBUG: Après retrait markdown ({len(cleaned)} caractères)")
            print(f"🔍 DEBUG: Premiers 300 caractères:")
            print(cleaned[:300])
            print(f"🔍 DEBUG: Derniers 200 caractères:")
            print(cleaned[-200:])
            
            # ========== EXTRACTION DU JSON ==========
            start_idx = cleaned.find('{')
            
            if start_idx == -1:
                print("❌ DEBUG: Aucune accolade ouvrante trouvée dans la réponse")
                print(f"❌ DEBUG: Contenu complet de 'cleaned':")
                print(cleaned)
                raise ValueError("Aucune accolade ouvrante trouvée dans la réponse")
            
            print(f"✅ DEBUG: Première accolade trouvée à l'index {start_idx}")
            
            # Compter les accolades pour trouver la fin
            brace_count = 0
            end_idx = -1
            in_string = False
            escape_next = False
            
            for i in range(start_idx, len(cleaned)):
                char = cleaned[i]
                
                if escape_next:
                    escape_next = False
                    continue
                
                if char == '\\':
                    escape_next = True
                    continue
                
                if char == '"':
                    in_string = not in_string
                    continue
                
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i
                            break
            
            # Extraire ou compléter le JSON
            if end_idx == -1:
                print(f"⚠️ DEBUG: JSON incomplet (accolades restantes: {brace_count})")
                json_str = cleaned[start_idx:]
                json_str += "\n" + ("}" * brace_count)
                print(f"🔧 DEBUG: JSON complété avec {brace_count} accolade(s)")
            else:
                json_str = cleaned[start_idx:end_idx + 1]
                print(f"✅ DEBUG: JSON complet trouvé")
            
            # ✅ VÉRIFICATION CRITIQUE
            print(f"🔍 DEBUG: Longueur de json_str = {len(json_str)}")
            print(f"🔍 DEBUG: Type de json_str = {type(json_str)}")
            print(f"🔍 DEBUG: json_str vide ? {len(json_str.strip()) == 0}")
            
            # ✅ NETTOYAGE AGRESSIF DU JSON
            import re
            
            print("🧹 Nettoyage agressif du JSON...")
            json_original_length = len(json_str)
            
            # Supprimer TOUS les caractères de formatage markdown
            json_str = json_str.replace('*', '')
            json_str = json_str.replace('_', '')
            json_str = json_str.replace('#', '')
            json_str = json_str.replace('`', '')
            
            # Normaliser les guillemets typographiques
            json_str = json_str.replace(''', "'")
            json_str = json_str.replace(''', "'")
            json_str = json_str.replace('"', '"')
            json_str = json_str.replace('"', '"')
            
            # Supprimer les virgules avant ] ou }
            json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)
            
            # Supprimer les doubles virgules
            json_str = re.sub(r',,+', ',', json_str)
            
            # Réparer les deux-points manquants après les clés
            json_str = re.sub(r'"\s+"', '": "', json_str)
            json_str = re.sub(r'"\s+\[', '": [', json_str)
            json_str = re.sub(r'"\s+\{', '": {', json_str)
            
            # Normaliser les espaces multiples (mais garder les sauts de ligne)
            json_str = re.sub(r'  +', ' ', json_str)
            
            chars_removed = json_original_length - len(json_str)
            print(f"🧹 Nettoyage terminé: {chars_removed} caractères supprimés/modifiés")
            print(f"🔍 JSON nettoyé ({len(json_str)} caractères)")
            
            # Afficher un extrait du JSON nettoyé
            print(f"🔍 DEBUG: Premiers 300 caractères après nettoyage:")
            print(json_str[:300])
            print(f"🔍 DEBUG: Derniers 200 caractères après nettoyage:")
            print(json_str[-200:])
            
            
            if not json_str or len(json_str.strip()) == 0:
                print("❌ ERREUR CRITIQUE: json_str est vide !")
                print(f"❌ start_idx = {start_idx}")
                print(f"❌ end_idx = {end_idx}")
                print(f"❌ brace_count = {brace_count}")
                print(f"❌ Contenu de 'cleaned' autour de start_idx:")
                print(cleaned[max(0, start_idx-50):min(len(cleaned), start_idx+200)])
                raise ValueError("JSON extrait est vide")
            
            print(f"🔍 DEBUG: Premiers 500 caractères de json_str:")
            print(json_str[:500])
            print(f"🔍 DEBUG: Derniers 300 caractères de json_str:")
            print(json_str[-300:])
            
            # Parser le JSON
            try:
                recipe_data = json.loads(json_str)
                print("✅ DEBUG: JSON parsé avec succès !")
                
                # ✅ NORMALISER LE SCORE ICI
                if 'score' in recipe_data:
                    score = recipe_data['score']
                    if isinstance(score, (int, float)) and score > 10:
                        recipe_data['score'] = round(score / 10, 1)
                        print(f"🔧 Score normalisé: {score} → {recipe_data['score']}")  # ✅ Guillemet ajouté
                
                # Validation
                required_fields = ['title', 'etapes', 'ingredients']
                for field in required_fields:
                    if not recipe_data.get(field):
                        print(f"⚠️ Champ manquant: {field}")
                
                print(f"   ✅ Recette générée: {recipe_data.get('title', 'Sans titre')}")
                print(f"   🔢 {len(recipe_data.get('etapes', []))} étapes")
                
                return recipe_data
            
            except json.JSONDecodeError as e:
                print(f"⚠️ DEBUG: Erreur parsing JSON: {e}")
                print(f"⚠️ Position: ligne {e.lineno}, col {e.colno}, pos {e.pos}")
                print("🔧 Tentative de réparation automatique...")
                
                # Tentative 1 : Nettoyage et réparation
                try:
                    json_cleaned = json_str
                    json_cleaned = json_cleaned.replace('*', '')
                    json_cleaned = json_cleaned.replace('_', '')
                    json_cleaned = json_cleaned.replace('#', '')
                    json_cleaned = json_cleaned.replace('`', '')
                    json_cleaned = json_cleaned.replace(''', "'")
                    json_cleaned = json_cleaned.replace(''', "'")
                    json_cleaned = json_cleaned.replace('"', '"')
                    json_cleaned = json_cleaned.replace('"', '"')
                    
                    # Essayer json-repair si disponible
                    try:
                        from json_repair import repair_json
                        json_repaired = repair_json(json_cleaned)
                        recipe_data = json.loads(json_repaired)
                        print("✅ JSON réparé avec json-repair !")
                        
                        # ✅ NORMALISER LE SCORE ICI (bien indenté maintenant)
                        if 'score' in recipe_data:
                            score = recipe_data['score']
                            if isinstance(score, (int, float)) and score > 10:
                                recipe_data['score'] = round(score / 10, 1)
                                print(f"🔧 Score normalisé: {score} → {recipe_data['score']}")
                        
                        # Validation
                        required_fields = ['title', 'etapes', 'ingredients']
                        for field in required_fields:
                            if not recipe_data.get(field):
                                print(f"⚠️ Champ manquant: {field}")
                        
                        return recipe_data
                        
                    except ImportError:
                        print("⚠️ json-repair non disponible")
                        
                    except ImportError:
                        print("⚠️ json-repair non disponible")
                        
                        # ✅ DEBUG : Afficher le contexte de l'erreur
                        if e.pos and e.pos < len(json_cleaned):
                            start = max(0, e.pos - 200)
                            end = min(len(json_cleaned), e.pos + 200)
                            print(f"\n🔍 CONTEXTE DE L'ERREUR (pos {e.pos}):")
                            print("=" * 80)
                            context = json_cleaned[start:end]
                            marker_pos = e.pos - start
                            print(context[:marker_pos] + " <<<ERREUR_ICI>>> " + context[marker_pos:])
                            print("=" * 80)
                    except Exception as repair_err:
                        print(f"⚠️ json-repair a échoué: {repair_err}")
                    
                    # Tentative 2 : Réparation manuelle
                    import re
                    json_repaired = json_cleaned
                    json_repaired = re.sub(r'"\s+"', '": "', json_repaired)
                    json_repaired = re.sub(r'"\s+\[', '": [', json_repaired)
                    json_repaired = re.sub(r'"\s+\{', '": {', json_repaired)
                    json_repaired = re.sub(r',(\s*[\]}])', r'\1', json_repaired)
                    json_repaired = re.sub(r',,+', ',', json_repaired)
                    
                    recipe_data = json.loads(json_repaired)
                    print("✅ JSON réparé manuellement !")
                    
                    # ✅ NORMALISER LE SCORE ICI AUSSI
                    if 'score' in recipe_data:
                        score = recipe_data['score']
                        if isinstance(score, (int, float)) and score > 10:
                            recipe_data['score'] = round(score / 10, 1)
                            print(f"🔧 Score normalisé: {score} → {recipe_data['score']}")
                    
                    return recipe_data
                    
                except Exception as repair_error:
                    print(f"❌ Toutes les réparations ont échoué: {repair_error}")
                    
                    # Tentative 2 : Réparation manuelle
                    import re
                    json_repaired = json_cleaned
                    json_repaired = re.sub(r'"\s+"', '": "', json_repaired)
                    json_repaired = re.sub(r'"\s+\[', '": [', json_repaired)
                    json_repaired = re.sub(r'"\s+\{', '": {', json_repaired)
                    json_repaired = re.sub(r',(\s*[\]}])', r'\1', json_repaired)
                    json_repaired = re.sub(r',,+', ',', json_repaired)
                    
                    recipe_data = json.loads(json_repaired)
                    print("✅ JSON réparé manuellement !")
                    return recipe_data
                    
                except Exception as repair_error:
                    print(f"❌ Toutes les réparations ont échoué: {repair_error}")
                
                # Tentative 3 : FALLBACK - Template statique garanti
                print("🆘 Utilisation du template de secours...")
                
                # Construire une recette minimale mais valide
                recipe_data = {
                    "title": cheese_name if 'cheese_name' in locals() else f"Fromage {cheese_type}",
                    "description": f"Fromage artisanal de type {cheese_type} au lait de {lait or 'vache'}",
                    "lait": lait or "vache",
                    "type_pate": cheese_type,
                    "ingredients": [
                        f"1L de lait {lait or 'vache'} entier",
                        "5ml de presure liquide",
                        "2g de ferments lactiques mesophiles",
                        "10g de sel fin non iode"
                    ],
                    "materiel": [
                        "Thermometre de cuisine",
                        "Grande casserole inox",
                        "Moule a fromage",
                        "Etamine ou tissu fromager",
                        "Louche",
                        "Couteau long"
                    ],
                    "etapes": [
                        "Steriliser tout le materiel en le plongeant 10 minutes dans eau bouillante puis laisser secher",
                        "Chauffer le lait doucement a 32 degres en remuant regulierement pour eviter que ca accroche",
                        "Ajouter les ferments lactiques et melanger delicatement pendant 2 minutes puis laisser reposer 30 minutes",
                        "Incorporer la presure diluee dans 50ml eau tiede melanger 1 minute puis laisser coaguler 45 a 60 minutes",
                        "Decouper le caille en cubes de 2cm avec un couteau stérilise puis brasser delicatement 10 minutes",
                        "Mouler le caille presser legerement et egoutter 12 a 24 heures en retournant toutes les 6 heures"
                    ],
                    "duree_totale": "24 a 48 heures",
                    "difficulte": "Moyenne",
                    "temperature_affinage": "12 a 14 degres avec 85 a 90% humidite",
                    "conseils": "Respecter scrupuleusement les temperatures et durees. Utiliser du lait cru pour plus de saveur. Patience essentielle pendant affinage.",
                    "score": 7.5,
                    "seed": seed,
                    "profile": profile
                }
                
                # Ajouter les aromates si présents
                if aromates:
                    for aromate in aromates[:3]:  # Max 3 aromates
                        recipe_data["ingredients"].append(f"{aromate} en quantite appropriee")
                    recipe_data["aromates"] = aromates
                    recipe_data["technique_aromatisation"] = f"Incorporer {', '.join(aromates)} pendant le brassage du caille"
                
                print(f"✅ Template de secours généré: {recipe_data['title']}")
                print(f"   🔢 {len(recipe_data['etapes'])} étapes")
                
                # Sauvegarder le JSON problématique pour analyse
                try:
                    import os
                    save_path = '/tmp/json_error.txt' if os.path.exists('/tmp') else 'json_error.txt'
                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write(f"=== ERREUR ===\n{e}\n\n")
                        f.write(f"=== JSON ORIGINAL ===\n{json_str}\n")
                    print(f"💾 JSON problématique sauvegardé: {save_path}")
                except:
                    pass
                
                # ✅ NORMALISER LE SCORE ICI (au cas où)
                if 'score' in recipe_data:
                    score = recipe_data['score']
                    if isinstance(score, (int, float)) and score > 10:
                        recipe_data['score'] = round(score / 10, 1)
                        print(f"🔧 Score normalisé: {score} → {recipe_data['score']}")
                        
                print(f"✅ Template de secours généré: {recipe_data['title']}")
                
                return recipe_data
                
            # # Validation des champs essentiels
            # required_fields = ['title', 'etapes', 'ingredients']
            # for field in required_fields:
            #     if not data.get(field):
            #         raise ValueError(f"Champ requis manquant : {field}")
            
            # # Validation du nombre d'étapes
            # if len(data.get('etapes', [])) < 6:
            #     print(f"   ⚠️ Seulement {len(data['etapes'])} étapes générées (minimum 6 recommandé)")
            
            # print(f"   ✅ LLM a généré : {data['title']}")
            # print(f"   🔢 {len(data.get('etapes', []))} étapes détaillées")
            # print(f"   🧀 Type : {data.get('type_pate', 'N/A')}")
            # print(f"   ⭐ Score : {data.get('score', 'N/A')}")
            
            
            
            # return data
            
        except json.JSONDecodeError as e:
            print(f"   ❌ Erreur de parsing JSON : {e}")
            print(f"   📄 Réponse reçue : {response[:200]}...")
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
    
       
    def _get_temp_caillage(self, cheese_type: str) -> str:
        """Température de caillage selon type"""
        if "Fromage frais" in cheese_type:
            return "35-37°C"
        elif "Pâte pressée cuite" in cheese_type:
            return "52-54°C"
        elif "Pâte persillée" in cheese_type:
            return "30-32°C"
        else:
            return "30-32°C"

    def _get_test_maturite(self, cheese_type: str) -> str:
        """Test de maturité selon type"""
        tests = {
            "Fromage frais": "il est ferme au toucher",
            "Pâte molle": "une croûte blanche fleurie s'est formée et qu'il est souple au centre",
            "Pâte pressée non cuite": "la croûte est sèche et que le fromage résiste à la pression du doigt",
            "Pâte pressée cuite": "il est dur et que la croûte est bien formée",
            "Pâte persillée": "les veines bleues sont bien développées et réparties"
        }
        return tests.get(cheese_type, "il est ferme et la croûte est formée")
       
    def _get_conseils_from_knowledge(self, cheese_type: str) -> str:
        """Récupère les conseils depuis la base (problèmes courants, etc.)"""
        
        conseils = []
        
        if self.knowledge_base and 'problemes_courants' in self.knowledge_base:
            # Prendre 2-3 problèmes courants pertinents
            problemes = list(self.knowledge_base['problemes_courants'].items())[:3]
            for probleme, solution in problemes:
                conseils.append(f"❌ {probleme}\n   ✅ {solution}")
        
        return "\n".join(conseils) if conseils else "Respectez les températures et l'hygiène."
    
    def _get_astuce_profile(self, profile: str) -> str:
        """Astuce spécifique au profil"""
        astuces = {
            "🧀 Amateur": "Pour votre première fois, divisez les quantités par deux (500ml de lait) - c'est plus facile à gérer et moins décourageant en cas d'erreur. Le fromage raté peut toujours servir en cuisine !",
            "🏭 Producteur": "Tenez un cahier de fabrication avec pH, température, durée exacte à chaque étape et résultat final. Cette traçabilité vous permettra de reproduire ou d'ajuster vos meilleures recettes.",
            "🎓 Formateur": "Préparez 3 échantillons à différents stades (caillé frais, fromage à 1 semaine, fromage affiné) pour montrer l'évolution. C'est très pédagogique de faire goûter le petit-lait - les gens sont souvent surpris de son goût légèrement sucré !"
        }
        return astuces.get(profile, "Notez vos observations à chaque étape pour progresser rapidement !")
        

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
                        # ===== DEBUG COMPLET =====
            print("=" * 80)
            print("🔍 DEBUG AGENT:")
            print(f"Type de self.agent: {type(self.agent)}")
            print(f"Contenu de self.agent: {self.agent}")
            print(f"Attributs de self.agent: {dir(self.agent)}")

            # Vérifier si c'est un dict
            if isinstance(self.agent, dict):
                print("⚠️ PROBLÈME: self.agent est un dictionnaire, pas un objet Agent !")
                print(f"Clés du dictionnaire: {list(self.agent.keys())}")
                return None

            # Vérifier si la méthode existe
            if not hasattr(self.agent, 'chat_with_llm'):
                print("❌ ERREUR: self.agent n'a pas de méthode 'chat_with_llm'")
                print(f"Méthodes disponibles: {[m for m in dir(self.agent) if not m.startswith('_')]}")
                return None

            print("✅ self.agent a la méthode chat_with_llm")
            print("=" * 80)

            # Maintenant l'appel
            response = self.agent.chat_with_llm(
                prompt,
                max_tokens=20000,  # ✅ DOUBLÉ pour avoir le JSON complet
                temperature=0.8
            )
            
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