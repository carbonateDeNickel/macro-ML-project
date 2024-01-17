# macro-ML-project

Authors: Marie CERVONI, Nicolas PRAT, Gabriel SKLÉNARD


## Points d'attention et corrections à effectuer
- Bien penser que les données sont des time-series : il faut utiliser un certain nombre de lags, et pas seulement les variables contemporaines !

## To-do
- [] IMPORTANT : Ajouter $y_{t-1}$ à $x_{t-1}$ -- de sorte que la recopie des lags de $x$ comprenne celle de $y$, pour aboutir à l'ensemble des variables utilisées pour prédire $y_t$. Ce serait dommage de ne pas utiliser les valeurs passées de $y$ pour prédire $y_t$ !

- [] Implémenter un réseau de neurones linéaire, si possible avec nombre de couches paramétrable pour pouvoir tester plusieurs configurations
- [] Etablir des règles en termes de contiguité temporelle des données. On prévoit avec combien de données passées ? Toutes ? Avec un réseau récurrent ? Mais alors comment on fait des tests ?
- [] Mettre en place une procédure de cross-validation pour choisir le nombre de lags $l$
- [] Comment manipuler et extraire des infos de la rétropropagation du gradient ? Notamment à des fins de recherche des variables les plus influentes
- [] Implémenter le calcul des valeurs de Shapley de tous les enregistrements, si possible d'une manière efficace en temps de calcul