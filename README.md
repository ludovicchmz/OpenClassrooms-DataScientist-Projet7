
Ce repository a pour but de répondre à la problématique du Projet 7 de la formation de Data Scientist de la plateforme de cours en ligne OpenClassrooms.
Ce projet est une mise en situation professionnelle où nous travaillons pour la société financière "Prêt à dépenser". Notre but est de mettre en place un outil de scoring crédit pour calculer la probabilité qu'un client rembourse son crédit, puis accorde ou non le crédit en fonction de données variées.
Nous devons donc ici :
- Construire le modèle
- Analyser les features (locales et globales)
- Déployer le modèle sous forme d'API et permettre de tester cette API
- En bref, mettre en oeuvre une approche globale MLOps de bout en bout, du tracking des expérimentations à l'analyse en production du data drift

Voici un petit guide pour vous aider à vous retrouver dans les différents éléments du repository :
- /workflows/python-app.yml : automatise les tests unitaires
- API.py : code de l'API
- Chaumaz_Ludovic_1_API_102025.docx : lien de l'API
- Chaumaz_Ludovic_2_notebook_drift_102025.ipynb : notebook de l'étude du data drift
- Chaumaz_Ludovic_2_notebook_modélisation_102025.ipynb : notebook de la construction et de la sélection du modèle
- Chaumaz_Ludovic_2_notebook_préparation_102025.ipynb : notebook de préparation des données pour le modèle
- Chaumaz_Ludovic_5_notebook_test_API_102025.ipynb : notebook de test de l'API
- Chaumaz_Ludovic_6_presentation_102025.pdf : support de présentation pour la soutenance du projet
- data_drift.html : Rapport de data drift par evidently
- explainer.sav : SHAP values du modèle
- feature_names.sav : nom des features du modèle
- for_unit_testing.csv : ligne unique des données pour les tests unitaires
- install_dependencies.sh : fichier exécutable lors du déploiement de l'API, faisant les installations nécessaires
- requirements.txt : liste des packages nécessaires pour tout le repository
- selected_model.sav : modèle utilisé
- smallest_test_8.csv : échantillon des données tests utilisé pour tester l'API
- test_api.py : code des tests unitaires
- useful_saved_parameters.csv : seuil de décision du modèle
