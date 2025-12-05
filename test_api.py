import pandas as pd
import API
import pytest

# On récupère notre jeu de test, avec les identifiants des clients
clients = pd.read_csv("for_unit_testing.csv")

#Test 1 : On a bien une erreur si "clients" n'est pas un DataFrame
def test_strange_type():
    # Arrange
    id = 100001
    clients = [1,2,3,4]
    
    #Assert
    with pytest.raises(TypeError):
        outcome = API.get_client_feats(clients, id)

#Test 2 : On a bien une erreur si l'identifiant du client n'est pas connu
def test_strange_id():
    # Arrange
    id = 000000
    
    #Assert
    with pytest.raises(ValueError):
        outcome = API.get_client_feats(clients, id)

#Test 3 : La valeur du montant du crédit du client 100001 est bien celle indiquée 
#dans le tableau
def test_value_credit_100001():
    # Arrange
    id = 100001
    
    #Act
    outcome = API.get_client_feats(clients, id)
    
    #Assert
    assert outcome[0][5] == 568800
