import streamlit as st
import numpy as np
import pandas as pd
import requests
import pickle
import shap
import plotly.graph_objects as go
import json
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

st.title("Informations sur la décision d'accord du prêt au client")

clients_list = requests.get("https://openclassrooms-datascientist-projet7.onrender.com/clients_list").json()

## Pour le menu déroulant permettant de sélectionner l'identifiant voulu
option = st.sidebar.selectbox(
    "Sélectionner l'identifiant du client",
     clients_list)

st.sidebar.write(f"Client sélectionné : :blue[{option}]")

## Pour la probabilité de faire défaut :
params = {'id' : option} # Ceci servira de nouveau
url_proba = "https://openclassrooms-datascientist-projet7.onrender.com/predict_proba"
client_proba = np.round(requests.get(url_proba, params = params).json()*100,1)


## Graphique affichant la position du client par rapport au seuil de décision
#
# On récupère la valeur du seuil
for_threshold = pd.read_csv("useful_saved_parameters.csv")
threshold =  for_threshold["Model threshold"][0]

# Affichage par une phrase :
if client_proba > threshold:
    st.header(f"Prêt :red[refusé] \
              - Probabilité estimée de défaut : {client_proba}%")
else:
    st.header(f"Prêt :green[accepté] \
              - Probabilité estimée de défaut : {client_proba}%")
    
# Affichage de la jauge
fig = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = client_proba,
    mode = "gauge+number+delta",
    title = {'text': f"Probabilité de faire défaut (seuil à {threshold}%)"},
    delta = {'reference' : threshold,
             'increasing' : {'color':'red'},
             'decreasing' : {'color' : 'green'}},
    gauge = {'axis': {'range': [None, 100]},
             'steps' : [
                 {'range': [0, threshold], 'color': "green"},
                 {'range': [threshold, 100], 'color': "red"}],
             'threshold' : {'line': {'color': "orange", 'width': 3},
                            'thickness': 1,
                            'value': threshold},
             'bar' : {'color':'darkblue'}}))

st.plotly_chart(fig)

## Pour le Waterfall plot
#
st.header("1. Explication (optionnelle) de la prise de décision du modèle")
# Récupération de l'objet "Explainer"
explainer = pickle.load(open('explainer.sav', 'rb'))

# Récupération des features de notre client préféré
url_features = "https://openclassrooms-datascientist-projet7.onrender.com/client_features_prep"
client_feats = requests.get(url_features, params = params).json()

# On utilise l'explainer pour obtenir un waterfall plot
shap_values = explainer(np.array(client_feats))

# Nom des features
feature_names = pickle.load(open('feature_names.sav', 'rb'))
shap_values.feature_names = feature_names

waterfall_box = st.checkbox(f'Afficher les paramètres les plus importants pour\
                            le client :blue[{option}]')

if waterfall_box:
    # Affichage
    st.set_option('deprecation.showPyplotGlobalUse', False)
    waterfall = shap.waterfall_plot(shap_values[0])
    st.pyplot(waterfall)
    
## Pour l'affichage d'une sélection de features
#
# Récupération des features non scalées
url_client = "https://openclassrooms-datascientist-projet7.onrender.com/client_features"
my_client = requests.get(url_client, params = params).json()

data_client = pd.DataFrame()
data_client["feature_name"] = feature_names
data_client["feature_value"] = my_client[0]

# Construction de l'objet de sélection des features
st.header("2. Affichage (optionnel) des caractéristiques du client")
options = st.multiselect('Quelles caractéristiques du client afficher ?',
                         data_client["feature_name"])

data_to_show = data_client.loc[data_client["feature_name"].isin(options), :]
st.write(data_to_show)

        
## Pour la comparaison avec des clients similaires
#
st.header("3. Comparaison avec des clients similaires (Même âge, mêmes revenus,\
          même mondant de crédit)")
# Récupération des features les plus importantes pour notre client
df = pd.DataFrame(shap_values[0].values)
df['feature_name'] = feature_names
df[0] = np.abs(df[0])
important_features = df.sort_values(0, ascending = False)["feature_name"][:9]
list_feats = important_features.values.tolist()

# Objet "paramètres" pour mon API
data = {"id" : option,
        "features" : list_feats}
headers = {"Content-Type": "application/json; charset=utf-8"}

# Récupération de mon dictionnaire fourni par l'API
url = "https://openclassrooms-datascientist-projet7.onrender.com/similar_clients"
neighbors_features = requests.post(url, data = json.dumps(data), headers = headers)
n_f = pd.DataFrame(neighbors_features.json())
feat_compare = st.multiselect("Pour quelles features souhaitez vous une comparaison ?",
                              list_feats)

for f in feat_compare:
    # Tracé
    colors = ["green", "blue", "blue", "blue", "blue", "blue",
              "blue", "blue", "blue", "blue", "blue"]
    fig, ax = plt.subplots()
    ax.bar(n_f.index, n_f[f], color = colors)
    plt.ylabel(f)
    plt.xlabel("Notre client (en vert) + 10 individus proches")
    plt.title(f"Valeurs de {f} pour les clients proches")
    st.pyplot(fig)

## Petites informations "Bonus" sur le client qui s'affichent toutes seules
age = data_client.loc[data_client['feature_name'] == "DAYS_BIRTH",
                      "feature_value"].values[0]/(-365)
age = int(age)
st.sidebar.write(f"Âge : :blue[{age}] ans")

credit_amt = data_client.loc[data_client['feature_name'] == "AMT_CREDIT",
                      "feature_value"].values[0]
credit_amt = int(credit_amt)
st.sidebar.write(f"Montant du crédit : :blue[{credit_amt}] $")

income = data_client.loc[data_client['feature_name'] == "AMT_INCOME_TOTAL",
                      "feature_value"].values[0]
income = int(income)
st.sidebar.write(f"Revenu annuel : :blue[{income}] $")
