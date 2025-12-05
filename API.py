"""
API Code
"""
from flask import Flask
from flask import request
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
    # Check if an ID was provided as part of the URL.
    # If ID is provided, assign it to a variable.
    # If no ID is provided, display an error in the browser.
    if 'id' in request.args:
        id = int(request.args['id'])
    else:
        return "Error: No id field provided. Please specify an id."

    # Now we check if the id is valid (is it in test_8 ?)
    if id not in clients["SK_ID_CURR"].values :
        return "Error : Invalid id provided. id not in testing set"
    else:
        # We select the good row and drop the id
        client = clients.loc[clients["SK_ID_CURR"] == id,:].drop(columns = ["SK_ID_CURR"])

        # We get a prediction
        predicted_failure_rate = model.predict_proba(client.values)[0][1]

        response = str(predicted_failure_rate)

    return response

@app.route('/clients_list', methods = ['GET'])
def clilist():
    clients_list = clients["SK_ID_CURR"]
    return clients_list.to_list()

#Pour l'explainer
@app.route('/client_features_prep', methods = ['GET'])
def clifeats():
    # Check if an ID was provided as part of the URL.
    # If ID is provided, assign it to a variable.
    # If no ID is provided, display an error in the browser.
    if 'id' in request.args:
        id = int(request.args['id'])
    else:
        return "Error: No id field provided. Please specify an id."

    client = get_client_feats(clients, id)
    prep_client = model[:-1].transform(client)

    return prep_client.tolist()

#Pour l'affichage des features, non scalées
@app.route('/client_features', methods = ['GET'])
def clientfeats():
    # Check if an ID was provided as part of the URL.
    # If ID is provided, assign it to a variable.
    # If no ID is provided, display an error in the browser.
    if 'id' in request.args:
        id = int(request.args['id'])
    else:
        return "Error: No id field provided. Please specify an id."

    client = get_client_feats(clients, id)

    return client.tolist()

#Pour l'affichage des features des clients "similaires"
@app.route('/similar_clients', methods = ['POST'])
def smilarclients():
    # Getting the body of the request
    data = request.json
    data_id = data["id"]
    client_index = clients.index[clients["SK_ID_CURR"] == data_id]
    feature_list = data["features"]

    # Using a nearest neighbors tool to get the neighbors
    for_neighbors = clients.loc[:,["DAYS_BIRTH","AMT_INCOME_TOTAL","AMT_CREDIT"]].values
    nn = NearestNeighbors(n_neighbors = 11)
    nn.fit(for_neighbors)
    distance, indices = nn.kneighbors(for_neighbors[client_index].reshape(1,-1),11)

    # Building a DataFrame with our wanted features
    df_neighbors = pd.DataFrame(columns = feature_list)
    j = 0
    for i in indices[0]:
        # Une ligne à rallonge pour cause de formatage...
        df_neighbors.loc[j] = clients.iloc[[i]][feature_list].values.tolist()[0]
        j += 1

    return df_neighbors.to_dict()


#app.run(debug=True, use_reloader=False)
