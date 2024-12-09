import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from itertools import combinations

# === Définir le modèle ===
(k1, k2, beta) = 0.8743478916919217, 0.8939613180195661, 1.3770147986739538

# Emotion smoothing function
def smooth_emotion(k1, k2, L, x,):
    if x==6 or x==7 or x==13 or x==14:
        return L[x-1]*k2
    else:
         return L[x-1]*k1

# Happiness model ODEs
def happiness_model(Y, t, beta, k1, k2, L):
    C, H = Y
    dCdt = H
    dHdt = -beta * H - np.sin(C) + smooth_emotion(k1, k2, L, x=int(t % 14))
    return [dCdt, dHdt]
Y0 = [1.0, 0.5] 
def average_happiness(L):
    time_points = np.linspace(0, 14, 1000)
    solution = odeint(happiness_model, Y0, time_points, args=(beta, k1, k2, L))
    H_solution = solution[:, 1]  # Extraire H de la solution
    H_mean = np.mean(H_solution)
    return(H_mean)
#maintenant on cherche la liste L qui maximise le bonheur sachant un nombre r de jours confinés
def optimal(r):
    """
    Trouve la liste binaire L qui maximise average_happiness
    tout en ayant exactement r jours confinés (r uns dans L).
    
    :param r: Nombre de jours confinés (uns dans L).
    :return: La liste L optimale et son bonheur moyen.
    """
    n = 14  # Longueur de la liste
    best_L = None
    best_happiness = float('-inf')

    # Générer toutes les combinaisons possibles de `r` jours confinés
    for indices in combinations(range(n), r):
        # Créer une liste binaire basée sur les indices sélectionnés
        L = [0] * n
        for idx in indices:
            L[idx] = 1
        
        # Calculer le bonheur moyen en utilisant votre fonction
        H_mean = average_happiness(L)
        
        # Mettre à jour le meilleur résultat si nécessaire
        if H_mean > best_happiness:
            best_happiness = H_mean
            best_L = L

    return best_L, best_happiness
print(optimal(2))