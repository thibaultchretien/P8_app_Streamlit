import os
import streamlit as st
import requests
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt

# Assurer que l'application se lie correctement au port Heroku
port = os.getenv('PORT', 8501)  # Utiliser 8501 en local si Heroku ne fournit pas le PORT
st.set_option('server.port', port)

# Fonction pour encoder l'image en base64
def image_to_base64(image):
    with io.BytesIO() as img_byte_array:
        image.save(img_byte_array, format='PNG')
        return base64.b64encode(img_byte_array.getvalue()).decode('utf-8')

# Fonction pour envoyer l'image à l'API déployée sur Heroku et obtenir la réponse
def get_segmented_image(image_base64):
    url = os.getenv('API_URL', 'https://app8oc-1fbc73130596.herokuapp.com/predict')  # URL API
    headers = {'Content-Type': 'application/json'}
    data = {'image': image_base64}
    
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code == 200:
        return response.json()['segmented_image']
    else:
        raise Exception(f"Erreur lors de la demande à l'API: {response.status_code}")

# Fonction pour convertir base64 en image
def base64_to_image(base64_str):
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data))

# Répertoires contenant les images et masques
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Assure l'accès au bon chemin
image_dir = os.path.join(BASE_DIR, 'test_images')
mask_dir = os.path.join(BASE_DIR, 'test_masks')

# Récupérer la liste des fichiers image et mask
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

# Interface utilisateur Streamlit
st.title("Segmentation d'Image avec API")

# Afficher la liste des IDs des images disponibles (sans l'extension .png)
image_ids = [f.split('.')[0] for f in image_files]  # Utiliser le nom de fichier sans l'extension comme ID
image_id = st.selectbox("Sélectionne un ID d'image", image_ids)

# Construire les chemins pour l'image et le mask en utilisant l'ID sélectionné
image_path = os.path.join(image_dir, f"{image_id}.png")
mask_path = os.path.join(mask_dir, f"{image_id}_mask.png")  # Nom du fichier mask correspond au format image_0_mask.png

# Vérifier si les fichiers existent
if os.path.exists(image_path) and os.path.exists(mask_path):
    # Charger l'image et le mask réel selon l'ID sélectionné
    real_image = Image.open(image_path)
    real_mask = Image.open(mask_path)

    # Afficher l'image réelle et le masque réel
    st.image(real_image, caption="Image Réelle", use_column_width=True)
    st.image(real_mask, caption="Mask Réel", use_column_width=True)

    # Ajouter un bouton pour lancer la prédiction
    if st.button('Lancer la prédiction'):
        # Convertir l'image réelle en base64 et obtenir le masque prédit depuis l'API
        image_base64 = image_to_base64(real_image)
        try:
            segmented_image_base64 = get_segmented_image(image_base64)

            # Décoder le masque prédit
            segmented_image = base64_to_image(segmented_image_base64)

            # Afficher l'image réelle, le mask réel et le mask prédit côte à côte
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Créer un graphique avec 3 sous-graphes (côte à côte)

            # Afficher l'image réelle
            axes[0].imshow(real_image)
            axes[0].set_title("Image Réelle")
            axes[0].axis('off')  # Masquer les axes

            # Afficher le mask réel
            axes[1].imshow(real_mask)
            axes[1].set_title("Mask Réel")
            axes[1].axis('off')  # Masquer les axes

            # Afficher le mask prédit
            axes[2].imshow(segmented_image)
            axes[2].set_title("Mask Prédit")
            axes[2].axis('off')  # Masquer les axes

            # Afficher le graphique
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {str(e)}")
else:
    st.error("Le fichier d'image ou de mask n'existe pas pour l'ID sélectionné.")
