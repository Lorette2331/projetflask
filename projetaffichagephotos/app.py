import os
import uuid
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from sklearn.cluster import KMeans, AgglomerativeClustering

# Dendrogramme (pas de fenêtre graphique)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

app = Flask(__name__)

# Dossiers
UPLOAD_FOLDER = os.path.join(app.root_path, "static", "uploads")
PROCESSED_FOLDER = os.path.join(app.root_path, "static", "processed")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg", "gif"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def safe_int(value, default=6, min_v=2, max_v=32) -> int:
    try:
        v = int(value)
    except Exception:
        v = default
    return max(min_v, min(v, max_v))


def kmeans_colors(input_path: str, output_path: str, k: int, max_size: int = 900) -> None:
    """Réduit les couleurs de l'image avec KMeans (pixels RGB)."""
    img = Image.open(input_path).convert("RGB")

    # Réduire la taille pour éviter que ça soit trop lent
    w, h = img.size
    scale = min(1.0, max_size / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))

    arr = np.array(img, dtype=np.uint8)
    pixels = arr.reshape(-1, 3)

    k = max(2, min(int(k), 32))
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(pixels)

    centers = np.clip(km.cluster_centers_, 0, 255).astype(np.uint8)
    new_pixels = centers[labels]
    new_arr = new_pixels.reshape(arr.shape)

    Image.fromarray(new_arr).save(output_path)


def hc_colors(
    input_path: str,
    output_path: str,
    dendro_path: str,
    k: int,
    max_size: int = 900,
    sample_pixels: int = 5000,
) -> None:
    """
    Réduction de couleurs par clustering hiérarchique + dendrogramme.
    On travaille sur un échantillon de pixels pour le dendrogramme (sinon trop lourd).
    """
    img = Image.open(input_path).convert("RGB")

    # Réduire la taille pour éviter que ça soit trop lent
    w, h = img.size
    scale = min(1.0, max_size / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))

    arr = np.array(img, dtype=np.uint8)
    pixels = arr.reshape(-1, 3).astype(np.float32)

    k = max(2, min(int(k), 32))

    # Échantillonnage de pixels
    n = pixels.shape[0]
    m = min(sample_pixels, n)
    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=m, replace=False)
    sample = pixels[idx]

    # Clustering hiérarchique sur l'échantillon
    agg = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="complete")
    labels = agg.fit_predict(sample)

    # Centres = moyenne RGB par cluster
    centers = np.zeros((k, 3), dtype=np.float32)
    for c in range(k):
        pts = sample[labels == c]
        centers[c] = pts.mean(axis=0) if len(pts) else sample.mean(axis=0)

    # Affecter tous les pixels au centre le plus proche
    dists = ((pixels[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    assign = np.argmin(dists, axis=1)
    new_pixels = centers[assign]
    new_arr = new_pixels.reshape(arr.shape).clip(0, 255).astype(np.uint8)

    Image.fromarray(new_arr).save(output_path)

    # Dendrogramme sur l'échantillon
    Z = linkage(sample, method="complete", metric="euclidean")
    plt.figure(figsize=(9, 4))
    dendrogram(Z, no_labels=True)
    plt.tight_layout()
    plt.savefig(dendro_path)
    plt.close()


@app.route("/", methods=["GET"])
def home():
    img = request.args.get("img", "")
    result = request.args.get("result", "")
    dendro = request.args.get("dendro", "")
    algo = request.args.get("algo", "original")
    k = request.args.get("k", "6")
    return render_template("home.html", img=img, result=result, dendro=dendro, algo=algo, k=k)


@app.route("/process", methods=["POST"])
def process():
    algo = request.form.get("algo", "original")
    k_int = safe_int(request.form.get("k", "6"), default=6)

    if "file" not in request.files:
        return redirect(url_for("home"))

    file = request.files["file"]
    if not file or file.filename == "":
        return redirect(url_for("home"))

    if not allowed_file(file.filename):
        return "Format non autorisé (png/jpg/jpeg/gif)", 400

    # Sauvegarde upload
    ext = file.filename.rsplit(".", 1)[1].lower()
    base_name = secure_filename(file.filename.rsplit(".", 1)[0])
    unique = uuid.uuid4().hex[:10]
    upload_name = f"{base_name}_{unique}.{ext}"
    upload_path = os.path.join(UPLOAD_FOLDER, upload_name)
    file.save(upload_path)

    # Original
    if algo == "original":
        return redirect(url_for("home", img=upload_name, algo=algo, k=k_int))

    # KMeans
    if algo == "kmeans":
        out_name = f"kmeans_{base_name}_{unique}_k{k_int}.png"
        out_path = os.path.join(PROCESSED_FOLDER, out_name)
        kmeans_colors(upload_path, out_path, k_int)
        return redirect(url_for("home", img=upload_name, result=out_name, algo=algo, k=k_int))

    # HC + dendrogramme
    if algo == "hc":
        out_name = f"hc_{base_name}_{unique}_k{k_int}.png"
        out_path = os.path.join(PROCESSED_FOLDER, out_name)

        dendro_name = f"dendro_{base_name}_{unique}_k{k_int}.png"
        dendro_path = os.path.join(PROCESSED_FOLDER, dendro_name)

        hc_colors(upload_path, out_path, dendro_path, k_int)

        return redirect(
            url_for("home", img=upload_name, result=out_name, dendro=dendro_name, algo=algo, k=k_int)
        )

    # algo inconnu -> fallback
    return redirect(url_for("home", img=upload_name, algo="original", k=k_int))


if __name__ == "__main__":
    app.run(debug=True, port=5002)
