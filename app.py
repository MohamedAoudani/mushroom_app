import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://img.goodfon.com/original/3840x2160/0/c7/grib-mukhomor-chernyi-fon.jpg");
        background-size: cover;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# 1. Load Model and Encoders
@st.cache_resource
def load_model():
    try:
        # Load model
        with open('modele_tree.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load encoders (ensure this is a dictionary of LabelEncoders)
        with open('mushroom_artifacts.pkl', 'rb') as f:
            encoders = pickle.load(f)
            
        return model, encoders
    except Exception as e:
        st.error(f"Error loading files: {str(e)}")
        return None, None

model, encoders = load_model()

# 2. Only show UI if model loaded successfully
if model and encoders:
    st.title('🍄 Mushroom Classification')
    
    # Define all 21 features (removed 'veil-type')
    features = {
        'cap-shape': ['bell', 'conical', 'convex', 'flat', 'knobbed', 'sunken'],
        'cap-surface': ['fibrous', 'grooves', 'scaly', 'smooth'],
        'cap-color': ['brown', 'buff', 'cinnamon', 'gray', 'green', 'pink', 'purple', 'red', 'white', 'yellow'],
        'bruises': ['bruises', 'no'],
        'odor': ['almond', 'anise', 'creosote', 'fishy', 'foul', 'musty', 'none', 'pungent', 'spicy'],
        'gill-attachment': ['attached', 'free'],
        'gill-spacing': ['close', 'crowded'],
        'gill-size': ['broad', 'narrow'],
        'gill-color': ['black', 'brown', 'buff', 'chocolate', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow'],
        'stalk-shape': ['enlarging', 'tapering'],
        'stalk-root': ['bulbous', 'club', 'equal', 'rhizomorphs', 'rooted', 'missing'],
        'stalk-surface-above-ring': ['fibrous', 'scaly', 'silky', 'smooth'],
        'stalk-surface-below-ring': ['fibrous', 'scaly', 'silky', 'smooth'],
        'stalk-color-above-ring': ['brown', 'buff', 'cinnamon', 'gray', 'orange', 'pink', 'red', 'white', 'yellow'],
        'stalk-color-below-ring': ['brown', 'buff', 'cinnamon', 'gray', 'orange', 'pink', 'red', 'white', 'yellow'],
        'veil-color': ['brown', 'orange', 'white', 'yellow'],
        'ring-number': ['none', 'one', 'two'],
        'ring-type': ['cobwebby', 'evanescent', 'flaring', 'large', 'none', 'pendant', 'sheathing', 'zone'],
        'spore-print-color': ['black', 'brown', 'buff', 'chocolate', 'green', 'orange', 'purple', 'white', 'yellow'],
        'population': ['abundant', 'clustered', 'numerous', 'scattered', 'several', 'solitary'],
        'habitat': ['grasses', 'leaves', 'meadows', 'paths', 'urban', 'waste', 'woods']
    }
    
    # User input for all features
    user_inputs = {}
    for feature, options in features.items():
        user_inputs[feature] = st.selectbox(
            f"{feature.replace('-', ' ').title()}",
            options=options
        )
    
    # 3. Prediction function
    def predict():
        try:
            # Prepare features in correct order (ensure you have all 21 features)
            encoded_features = []
            for feature in features.keys():
                if feature in encoders:  # Use encoder if available
                    encoded_val = encoders[feature].transform([user_inputs[feature]])[0]
                else:  # Manual mapping as fallback
                    encoded_val = features[feature].index(user_inputs[feature])
                encoded_features.append(encoded_val)
            
            # Debugging: Check the length of the encoded feature vector
            st.write(f"Encoded features length: {len(encoded_features)}")
            
            # Ensure that the features are in the same order as the model expects
            if len(encoded_features) != 21:
                st.error("Error: Feature vector does not match expected size (21 features expected).")
                return None

            # Make prediction
            prediction = model.predict([encoded_features])[0]
            return 'e' if prediction == 0 else 'p'
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            return None
    
    # 4. Run prediction
    if st.button('Predict'):
        result = predict()
        if result == 'e':
            st.success('✅ Edible')
        elif result == 'p':
            st.error('☠️ Poisonous')


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 5. Add analysis and visualizations button# app.py
import streamlit as st
from PIL import Image

if st.button('Analyser les champignons'):
    st.subheader("Analyse du Dataset de Champignons")
    
    try:
        # === Distribution des classes et caractéristiques ===
        st.write("### 1. Distribution des classes (Comestible vs Vénéneux)")
        class_img = Image.open('class_distribution.png')
        st.image(class_img, use_container_width=True)
        st.markdown("""
        **Analyse** :  
        Ce graphique montre la proportion de champignons comestibles (e) vs vénéneux (p).  
        On observe que le dataset est équilibré, ce qui est idéal pour l'entraînement du modèle.
        """)

        # === Forme du chapeau ===
        st.write("### 2. Forme du chapeau (Cap Shape)")
        cap_img = Image.open('cap_shape.png')
        st.image(cap_img,  use_container_width=True)
        st.markdown("""
        **Signification** :  
        - `x` = convexe  
        - `b` = en cloche  
        - `s` = plat  
        **Insight** : Les formes convexes sont les plus courantes dans ce dataset.
        """)

        # === Odeur ===
        st.write("### 3. Caractéristiques d'odeur")
        odor_img = Image.open('odor_distribution.png')
        st.image(odor_img,  use_container_width=True)
        st.markdown("""
        **Clé des odeurs** :  
        - `n` = nauséabonde  
        - `l` = amer  
        - `a` = anisé  
        **Observation** : L'absence d'odeur (none) est la plus fréquente.
        """)

        # === Habitat vs Population ===
        st.write("### 4. Habitat vs Population")
        habitat_img = Image.open('habitat_population.png')
        st.image(habitat_img,  use_container_width=True)
        st.markdown("""
        **Légende** :  
        - `u` = urbain  
        - `g` = prairie  
        - `d` = forêt  
        **Tendance** : La plupart des champignons poussent en forêt.
        """)

        # === Analyses avancées ===
        st.write("## Corrélations avec la toxicité")

        # 1. Odeur vs Classe
        st.write("### 5. Odeur vs Toxicité")
        col1, col2 = st.columns(2)
        with col1:
            odor_class_img = Image.open('odor_vs_class.png')
            st.image(odor_class_img,  use_container_width=True)
        with col2:
            st.markdown("""
            **Conclusions** :  
            - Les odeurs `n` (nauséabonde), `f` (poisson) et `c` (créosote) sont des indicateurs forts de toxicité.  
            - Les champignons sans odeur (`none`) sont majoritairement comestibles.
            """)

        # 2. Couleur des spores
        st.write("### 6. Couleur des spores vs Toxicité")
        col1, col2 = st.columns(2)
        with col1:
            spore_class_img = Image.open('spore_color_vs_class.png')
            st.image(spore_class_img,  use_container_width=True)
        with col2:
            st.markdown("""
            **Points clés** :  
            - Les spores de couleur `r` (marron) et `w` (blanc) sont plus fréquentes chez les comestibles.  
            - Les couleurs `k` (noir) et `h` (chocolat) apparaissent surtout chez les vénéneux.
            """)

        # 3. Habitat vs Classe
        st.write("### 7. Habitat vs Toxicité")
        habitat_class_img = Image.open('habitat_vs_class.png')
        st.image(habitat_class_img,  use_container_width=True)
        st.markdown("""
        **Insights** :  
        - Les champignons en prairie (`g`) sont souvent comestibles.  
        - Ceux en zone urbaine (`u`) sont majoritairement vénéneux.  
        - Les forêts (`d`) contiennent les deux types en proportions similaires.
        """)

        # 4. Bleus (Bruises)
        st.write("### 8. Présence de bleus vs Toxicité")
        bruises_img = Image.open('bruises_vs_class.png')
        st.image(bruises_img,  use_container_width=True)
        st.markdown("""
        **Résultat surprenant** :  
        Contrairement à une croyance populaire, les champignons comestibles développent plus souvent des bleus (`t`) que les vénéneux.
        """)

        # 5. Couleur des lamelles
        st.write("### 9. Couleur des lamelles vs Toxicité")
        gill_color_img = Image.open('gill_color_vs_class.png')
        st.image(gill_color_img,  use_container_width=True)
        st.markdown("""
        **Indicateurs forts** :  
        - Les lamelles blanches (`w`) sont associées aux comestibles.  
        - Les couleurs vertes (`g`) et grises (`k`) apparaissent surtout chez les vénéneux.
        """)

    except FileNotFoundError as e:
        st.error(f"Fichier image manquant : {str(e)}")
    except Exception as e:
        st.error(f"Erreur inattendue : {str(e)}")
