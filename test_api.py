import pandas as pd
import API
import pytest

# Récupération du jeu de test avec les ID clients
clients = pd.read_csv("for_unit_testing.csv")

#Test 1 : Erreur si "clients" n'est pas du bon type (dataframe)
def test_strange_type():
    # Arrange
    id = 100001
    clients = [1,2,3,4]
    
    #Assert
    with pytest.raises(TypeError):
        outcome = API.get_client_feats(clients, id)

#Test 2 : Erreur si ID client inconnu
def test_strange_id():
    # Arrange
    id = 000000
    
    #Assert
    with pytest.raises(ValueError):
        outcome = API.get_client_feats(clients, id)

#Test 3 : Bonne valeur de crédit pour le client 100001 
def test_value_credit_100001():
    # Arrange
    id = 100001
    
    #Act
    outcome = API.get_client_feats(clients, id)
    
    #Assert
    assert outcome[0][5] == 568800
