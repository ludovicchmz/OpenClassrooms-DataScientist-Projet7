# Utilise une image Python officielle (version 3.9)
FROM python:3.9-slim

# Installe les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \  # Pour compiler certaines dépendances
    && rm -rf /var/lib/apt/lists/*  # Nettoie le cache pour réduire la taille de l'image

# Définit le répertoire de travail dans le conteneur
WORKDIR /app

# Copie les fichiers nécessaires depuis ton projet local vers le conteneur
COPY requirements.txt .
COPY app.py .
COPY sample_input.json .

# Installe les dépendances Python (sans cache pour réduire la taille)
RUN pip install --no-cache-dir -r requirements.txt

# Expose le port sur lequel ton API écoute (10000 dans ton cas)
EXPOSE 10000

# Commande pour lancer ton API quand le conteneur démarre
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
