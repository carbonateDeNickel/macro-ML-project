# macro-ML-project

Authors: Marie CERVONI, Nicolas PRAT, Gabriel SKLÉNARD


## Points d'attention et corrections à effectuer
- Bien penser que les données sont des time-series : il faut utiliser un certain nombre de lags, et pas seulement les variables contemporaines !

## To-do
- Implémenter un réseau de neurones linéaire, si possible avec nombre de couches paramétrable pour pouvoir tester plusieurs configurations
- Etablir des règles en termes de contiguité temporelle des données. On prévoit avec combien de données passées ? Toutes ? Avec un réseau récurrent ? Mais alors comment on fait des tests ?
- Comment manipuler et extraire des infos de la rétropropagation du gradient ? Notamment à des fins de recherche des variables les plus influentes
- Implémenter le calcul des valeurs de Shapley de tous les enregistrements, si possible d'une manière efficace en temps de calcul