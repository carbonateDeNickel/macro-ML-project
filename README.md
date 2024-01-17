# macro-ML-project

Authors: Marie CERVONI, Nicolas PRAT, Gabriel SKLÉNARD


## Points d'attention et corrections à effectuer
- Bien penser que les données sont des time-series : il faut utiliser un certain nombre de lags, et pas seulement les variables contemporaines !

## To-do
- [x] IMPORTANT : Ajouter $y_{t-1}$ à $x_{t-1}$ -- de sorte que la recopie des lags de $x$ comprenne celle de $y$, pour aboutir à l'ensemble des variables utilisées pour prédire $y_t$. Ce serait dommage de ne pas utiliser les valeurs passées de $y$ pour prédire $y_t$ !
- [] Mettre en place une procédure de cross-validation pour choisir le nombre de lags $p$
    - Problème : pour définir les train/validation/test sets, il faut des données, or ces données sont construites à la volée, pour chaque $p$ donné. Il faut donc construire ces ensembles différemment.
    - Contentons-nous de définir les dates de test pour $y_t$, et gardons en mémoire que les données de train n'iront que jusqu'à $t_0 - p - 1$. Les datasets pourront être construits le moment venu.
    - [x] Objectif intermédiaire : construire une fonction qui fait une étape de cross-validation


- [] Implémenter un réseau de neurones linéaire, si possible avec nombre de couches paramétrable pour pouvoir tester plusieurs configurations
- [] Etablir des règles en termes de contiguité temporelle des données. On prévoit avec combien de données passées ? Toutes ? Avec un réseau récurrent ? Mais alors comment on fait des tests ?
- [] Comment manipuler et extraire des infos de la rétropropagation du gradient ? Notamment à des fins de recherche des variables les plus influentes
- [] Implémenter le calcul des valeurs de Shapley de tous les enregistrements, si possible d'une manière efficace en temps de calcul