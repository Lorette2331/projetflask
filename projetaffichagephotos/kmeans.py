import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

def calculer_clusters(dossier_images, n_clusters=3):
    donnees = []
    noms_fichiers = []

    # Vérifier si le dossier existe
    if not os.path.exists(dossier_images):
        return {}

    for fichier in os.listdir(dossier_images):
        if fichier.lower().endswith(('.png', '.jpg', '.jpeg')):
            chemin = os.path.join(dossier_images, fichier)
            try:
                with Image.open(chemin) as img:
                    # 1. Récupérer Largeur (W) et Hauteur (L)
                    W, L = img.size
                    
                    # 2. Récupérer les moyennes R, G, B
                    img_rgb = img.convert('RGB')
                    pixels = np.array(img_rgb)
                    # axis=(0,1) calcule la moyenne sur toute la surface de l'image
                    R, G, B = np.mean(pixels, axis=(0, 1))
                    
                    donnees.append([L, W, R, G, B])
                    noms_fichiers.append(fichier)
            except Exception as e:
                print(f"Erreur sur {fichier}: {e}")

    # Si on n'a pas assez d'images pour faire des groupes
    if len(donnees) < n_clusters:
        return {0: noms_fichiers}

    # L'algorithme K-Means : il crée les "tas"
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(donnees)

    # On range les noms de fichiers dans leurs groupes respectifs
    groupes = {}
    for i in range(n_clusters):
        groupes[i] = [noms_fichiers[j] for j in range(len(noms_fichiers)) if labels[j] == i]
    
    return groupes