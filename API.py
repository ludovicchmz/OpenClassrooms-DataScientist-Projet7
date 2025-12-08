"""
API Code
"""
from flask import Flask, jsonify, request, make_response
import pandas as pd
import pickle
from sklearn.neighbors import NearestNeighbors


app = Flask(__name__)
app.config["DEBUG"] = True

# On récupère notre jeu de test, avec les identifiants des clients
clients_raw = pd.read_csv("smallest_test_8.csv")
clients = clients_raw.sort_values(by = "SK_ID_CURR").reset_index(drop=True)

# On récupère le modèle enregistré
model = pickle.load(open('selected_model.sav', 'rb'))

#Fonction utile pour plus tard
def get_client_feats(clients, id):
    """
    Gives us the list of features for our specific client id

    Parameters
    ----------
    clients : DataFrame
        The dataframe that contains the features for all the clients
    id : int
        the id of the client we are looking for

    Returns
    -------
    array
        An array with only one row : the list of the feature's values for our client.

    """
    if not isinstance (clients, pd.DataFrame):
        raise TypeError("The first argument must be a dataframe")
    elif id not in clients["SK_ID_CURR"].values :
        raise ValueError("this id does not belong to any client we know of")
    else:
        client = clients.loc[clients["SK_ID_CURR"] == id,:].drop(columns = ["SK_ID_CURR"]).values
        return client


@app.route('/predict_proba', methods = ['GET'])
def prob():
    if 'id' in request.args:
        id = int(request.args['id'])
    else:
        return jsonify({"error": "No id field provided. Please specify an id."}), 400

    if id not in clients["SK_ID_CURR"].values:
        return jsonify({"error": "Invalid id provided. Id not in testing set"}), 400

    client = clients.loc[clients["SK_ID_CURR"] == id, :].drop(columns=["SK_ID_CURR"])
    predicted_failure_rate = model.predict_proba(client.values)[0][1]
    return jsonify({"predicted_failure_rate": predicted_failure_rate})

@app.route('/clients_list', methods = ['GET'])
def clilist():
    clients_list = clients["SK_ID_CURR"].to_list()
    return jsonify(clients_list) 

#Pour l'explainer
@app.route('/client_features_prep', methods = ['GET'])
def clifeats():
    if 'id' in request.args:
        id = int(request.args['id'])
    else:
        return jsonify({"error": "No id field provided. Please specify an id."}), 400

    client = get_client_feats(clients, id)
    prep_client = model[:-1].transform(client)
    return jsonify(prep_client.tolist())

#Pour l'affichage des features, non scalées
@app.route('/client_features', methods = ['GET'])
def clientfeats():
    if 'id' in request.args:
        id = int(request.args['id'])
    else:
        return jsonify({"error": "No id field provided. Please specify an id."}), 400

    client = get_client_feats(clients, id)
    return jsonify(client.tolist())

#Pour l'affichage des features des clients "similaires"
@app.route('/similar_clients', methods = ['POST'])
def smilarclients():
    data = request.json
    data_id = data["id"]
    client_index = clients.index[clients["SK_ID_CURR"] == data_id]
    feature_list = data["features"]

    for_neighbors = clients.loc[:, ["DAYS_BIRTH", "AMT_INCOME_TOTAL", "AMT_CREDIT"]].values
    nn = NearestNeighbors(n_neighbors=11)
    nn.fit(for_neighbors)
    distance, indices = nn.kneighbors(for_neighbors[client_index].reshape(1, -1), 11)

    df_neighbors = pd.DataFrame(columns=feature_list)
    j = 0
    for i in indices[0]:
        df_neighbors.loc[j] = clients.iloc[[i]][feature_list].values.tolist()[0]
        j += 1

    return jsonify(df_neighbors.to_dict()) 

# Pour le menu principal
@app.route('/', methods=['GET'])
def home():
    return make_response(
        "Bienvenue sur l'API de prédiction de crédit ! Voici les endpoints disponibles : <br>" \
        "/clients_list : Liste des identifiants des clients <br>" \
        "/predict_proba : Prédiction de la probabilité de défaut pour un client donné",
        200
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
