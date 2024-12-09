Le fichier à éxecuter est main1.py (dans le dossier brouillon), il trace l'évolution du 'Happiness' pour une stratégie de confinement sous la forme d'une liste L.
Important: les paramétres de modèle k1, k2 et beta sont obtenus pour une stratégie de confinement complet, vu que les données datent de la période
du Covid en australie, durant laquelle il'y avait un confinement total.
(k1, k2, beta) : 0.8743478916919217 0.8939613180195661 1.3770147986739538 obtenu à travers l'éxecution de find_params
maintenant que k1, k2 et beta sont trouvés, on peut résoudre numériquement le modèle et trouver pour un nombre minimal de jours confinés, 
la liste L qui maximise le bonheur, pour ceci executez le code optimal_l