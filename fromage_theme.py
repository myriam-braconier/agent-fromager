"""
THÈME GRADIO FROMAGE - Remplace le CSS custom
==============================================

Utilise l'API de thème Gradio au lieu de CSS custom pour éviter
les problèmes de Content Security Policy sur Hugging Face Spaces.
"""

import gradio as gr

def create_fromage_theme():
    """Crée un thème fromage complet avec tous les styles"""
    
    theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="amber",
        neutral_hue="stone",
        font=gr.themes.GoogleFont("Quicksand"),
        radius_size=gr.themes.sizes.radius_lg,
    )
    
    # Appliquer seulement les styles supportés par Gradio
    theme.set(
        # ===== FOND GÉNÉRAL =====
        body_background_fill="url(https://unsplash.com/fr/photos/une-assiette-de-nourriture-rOdM3LQRd50) center/cover, linear-gradient(135deg, #FFF9E6 0%, #FFE5B4 100%)",
        body_background_fill_dark="url('https://unsplash.com/fr/photos/une-assiette-de-nourriture-rOdM3LQRd50') center/cover, linear-gradient(135deg, #2C1810 0%, #3E2723 100%)",
        
        # ===== TEXTE =====
        body_text_color="#3E2723",
        
        # ===== TITRES =====
        block_title_text_color="#BF360C",
        
        # ===== LABELS =====
        block_label_text_color="#4E342E",
        
        # ===== BOUTONS PRIMAIRES =====
        button_primary_background_fill="linear-gradient(45deg, #667eea 0%, #764ba2 100%)",
        button_primary_background_fill_hover="linear-gradient(45deg, #5568d3 0%, #63408d 100%)",
        button_primary_text_color="white",
        
        # ===== BOUTONS SECONDAIRES =====
        button_secondary_background_fill="#FFE5B4",
        button_secondary_background_fill_hover="#FFDAB9",
        button_secondary_text_color="#3E2723",
        
        # ===== INPUTS / TEXTBOX =====
        input_background_fill="#FFFEF5",  # Fond crème fromage        
        input_border_color="#FFE5B4",
        
        # ===== BLOCKS =====
        block_background_fill="rgba(255, 255, 255, 0.9)",
        
        # ===== PANEL =====
        panel_background_fill="#FFFEF5",
    )
    
    return theme


# ===== CSS MINIMAL pour les cas spéciaux =====
# (Seulement ce qui ne peut pas être fait avec le thème)

minimal_css = """
/* FORCER ABSOLUMENT TOUT LE TEXTE EN NOIR */
body, body *, 
.gradio-container, .gradio-container *,
div, p, li, ul, ol, h1, h2, h3, h4, h5, h6, span,
.markdown-body, .prose, .gr-markdown, .gr-html,
.recipe-card, #recipe-scroll {
    color: #3E2723 !important;
}

/* Exceptions pour garder certains éléments blancs */
button.primary, button.primary *{
    color: white !important;
}

/* INPUTS */
input, textarea, .gr-text-input, .gr-textbox {
    background: #FFFEF5 !important;
    color: #3E2723 !important;
}

/* SCROLL */
#recipe-scroll {
    max-height: 800px !important;
    overflow-y: auto !important;
}

#chat-display {
    max-height: 500px !important;
    overflow-y: auto !important;
}

/* LOGIN BOX */
.login-box {
    max-width: 400px !important;
    margin: 100px auto !important;
}

/* CARTES DE RECETTES */
.recipe-card {
    background: #f9f9f9 !important;
    padding: 15px !important;
    margin: 10px 0 !important;
    border-radius: 8px !important;
    border: 1px solid #e0e0e0 !important;
}

.no-recipes {
    text-align: center !important;
    padding: 40px !important;
    color: #666 !important;
    font-size: 1.2em !important;
}

/* SCROLLBARS */
#recipe-scroll::-webkit-scrollbar,
#chat-display::-webkit-scrollbar {
    width: 8px;
}

#recipe-scroll::-webkit-scrollbar-track,
#chat-display::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 4px;
}

#recipe-scroll::-webkit-scrollbar-thumb,
#chat-display::-webkit-scrollbar-thumb {
    background: #FFE5B4;
    border-radius: 4px;
}

#recipe-scroll::-webkit-scrollbar-thumb:hover,
#chat-display::-webkit-scrollbar-thumb:hover {
    background: #FFDAB9;
}

.recipe-card, .recipe-card * {
    color: #3E2723 !important;
}

#recipe-scroll, #recipe-scroll * {
    color: #3E2723 !important;
}

.markdown-body, .prose, .gr-markdown, .gr-html {
    color: #3E2723 !important;
}

ul, ol, li, p {
    color: #3E2723 !important;
}

/* INPUTS SPÉCIFIQUES */
#date_str, #lait_emoji, #type_pate, #duree_affinage, #temperature {
    color: black !important;
    background: #FFFEF5 !important;
}

/* TOUS LES INPUTS - Fond crème et texte marron */
input, textarea, .gr-text-input, .gr-textbox {
    background: #FFFEF5 !important;
    color: #3E2723 !important;
}

/* SCROLL PERSONNALISÉ */
#recipe-scroll {
    max-height: 800px !important;
    overflow-y: auto !important;
}

#chat-display {
    max-height: 500px !important;
    overflow-y: auto !important;
}

/* LOGIN BOX */
.login-box {
    max-width: 400px !important;
    margin: 100px auto !important;
}

/* CARTES DE RECETTES - SANS color: #FFFFFF ! */
.recipe-card {
    background: #f9f9f9 !important;
    padding: 15px !important;
    margin: 10px 0 !important;
    border-radius: 8px !important;
    border: 1px solid #e0e0e0 !important;
}

/* MESSAGE "PAS DE RECETTES" */
.no-recipes {
    text-align: center !important;
    padding: 40px !important;
    color: #666 !important;
    font-size: 1.2em !important;
}

/* SCROLLBARS WEBKIT */
#recipe-scroll::-webkit-scrollbar,
#chat-display::-webkit-scrollbar {
    width: 8px;
}

#recipe-scroll::-webkit-scrollbar-track,
#chat-display::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 4px;
}

#recipe-scroll::-webkit-scrollbar-thumb,
#chat-display::-webkit-scrollbar-thumb {
    background: #FFE5B4;
    border-radius: 4px;
}

#recipe-scroll::-webkit-scrollbar-thumb:hover,
#chat-display::-webkit-scrollbar-thumb:hover {
    background: #FFDAB9;
}
"""

# ===== EXEMPLE D'UTILISATION =====

if __name__ == "__main__":
    """
    Utilise ce thème dans ton app.py comme ceci:
    
    from fromage_theme import create_fromage_theme, minimal_css
    
    fromage_theme = create_fromage_theme()
    
    with gr.Blocks(
        title="🧀 Agent Fromager",
        theme=fromage_theme,
        css=minimal_css,
        head='<link rel="icon" type="image/png" href="https://em-content.zobj.net/source/apple/391/cheese-wedge_1f9c0.png">',
    ) as demo:
        # Ton interface...
    """
    pass