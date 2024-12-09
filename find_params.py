import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

# === Définir le modèle ===

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
# === Paramètres initiaux ===
Y0 = [1.0, 0.5]  # Conditions initiales pour H et C
time_points = np.linspace(0, 100, 1000)  # Points de temps pour l'intégration
gnh_values = np.sin(0.1 * time_points) + 1  # Exemple de données cibles pour GNH

# === Étape 1 : Optimiser k1, k2, et beta ===
def objective_function(params):
    """
    Fonction objectif pour minimiser l'erreur quadratique moyenne entre le bonheur simulé et les valeurs cibles.
    """
    k1, k2, beta = params
    L_initial = np.ones(14)  # Initialiser L comme une valeur par défaut
    solution = odeint(happiness_model, Y0, time_points, args=(beta, k1, k2, L_initial))
    predicted_happiness = solution[:, 0]
    return np.mean((predicted_happiness - gnh_values)**2)

# Optimisation des paramètres k1, k2, et beta
initial_guess = [1.0, 1.0, 0.5]
bounds = [(0, None), (0, None), (0, None)]  # Contraintes sur les paramètres
result = minimize(objective_function, initial_guess, method='L-BFGS-B', bounds=bounds)

k1_opt, k2_opt, beta_opt = result.x
print("Paramètres optimaux (k1, k2, beta) :", k1_opt, k2_opt, beta_opt)

# === Étape 2 : Trouver L qui maximise le bonheur ===
def happiness_for_L(L):
    """
    Fonction objectif pour maximiser le bonheur moyen pour une configuration donnée de L.
    """
    L = np.round(L)  # Assurer que L est une liste binaire (0 ou 1)
    solution = odeint(happiness_model, Y0, time_points, args=(beta_opt, k1_opt, k2_opt, L))
    predicted_happiness = solution[:, 0]
    return -np.mean(predicted_happiness)  # On minimise le négatif pour maximiser le bonheur

# Optimisation pour L
L_initial = np.ones(14)  # L de départ (toutes les valeurs à 1)
bounds_L = [(0, 1) for _ in range(14)]  # Contraintes sur L (valeurs binaires)
result_L = minimize(happiness_for_L, L_initial, method='L-BFGS-B', bounds=bounds_L)

L_optimal = np.round(result_L.x)
print("L optimal :", L_optimal)

# === Résultats finaux ===
solution_final = odeint(happiness_model, Y0, time_points, args=(beta_opt, k1_opt, k2_opt, L_optimal))
predicted_happiness = solution_final[:, 0]

# === Visualisation ===
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(time_points, predicted_happiness, label="Bonheur prédit (H)", color="green")
plt.plot(time_points, gnh_values, label="Données réelles (GNH)", color="blue", linestyle="dashed")
plt.title("Optimisation de k1, k2, beta et L")
plt.xlabel("Temps")
plt.ylabel("Bonheur")
plt.legend()
plt.grid()
plt.show()
