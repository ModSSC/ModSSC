# Self-Training historique sur Wine : protocole de réplication

## Source et résultat visé

La carte `bench/configs/reproductions/self_training/wine_table3.yaml`
reconstruit la ligne **Self-Training** de la Table 3 de Li et Zhou,
*SETRED: Self-Training with Editing* (PAKDD 2005). La cible publiée est une
erreur test moyenne de **7,9 %** sur 50 partitions du jeu UCI Wine.

Le protocole publié utilise les 178 exemples et 13 attributs de Wine, conserve
25 % des exemples pour le test, puis divise les 75 % restants en 10 % étiquetés
et 90 % non étiquetés en préservant approximativement les proportions de
classes. L'apprenant de base est 1-NN. À chaque tour, Self-Training retient
l'exemple prédit le plus sûrement dans chaque classe et s'arrête après au plus
40 tours. La source primaire est disponible sur le
[site des auteurs](https://ai.nju.edu.cn/lim/publications/pakdd05.pdf).

## Reconstruction préenregistrée

Les détails absents de l'article sont figés avant tout calcul test :

- OpenML `data_id=187`, mis en cache et vérifié hors ligne, représente Wine ;
- graines 1 à 50, split test stratifié de 25 %, aucune validation ;
- sélection proportionnelle de 10 % des données d'entraînement comme données
  étiquetées, avec au moins un exemple par classe ;
- standardisation ajustée uniquement sur les exemples étiquetés ;
- 1-NN NumPy, distance euclidienne et vote uniforme ;
- pool de sélection égal à l'ensemble non étiqueté restant ;
- confiance égale à la marge entre la distance au plus proche exemple d'une
  autre classe et la distance au plus proche exemple de la classe prédite ;
- un candidat par classe et par tour, égalités départagées par l'ordre stable
  des indices ;
- 40 tours maximum, sans seuil de confiance ni choix fondé sur le test.

Chaque exécution doit archiver la partition et la trace des tours, notamment le
pool, les candidats proposés, les candidats acceptés et les tailles L/U avant
et après chaque mise à jour.

## Limite de la revendication

L'article ne publie ni les indices des 50 partitions, ni les graines, ni la
normalisation, ni la formule numérique de confiance, ni la taille du pool, ni
les règles d'égalité. Même si la cible numérique est compatible, le statut
scientifique maximal est donc **`paper_approx`**, et non `paper_matched`.

La décision finale compare la moyenne d'erreur des 50 exécutions à 7,9 %, son
intervalle de confiance à 95 %, la dispersion et la dynamique des
pseudo-étiquettes. Aucun de ces résultats ne peut servir à modifier les choix
ci-dessus.
