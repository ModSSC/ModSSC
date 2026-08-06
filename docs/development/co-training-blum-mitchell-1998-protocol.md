# Co-Training original sur WebKB : protocole de réplication

## Source et résultat visé

La carte `bench/configs/reproductions/co_training/webkb_course_table2.yaml`
reconstruit la Table 2 de Blum et Mitchell, *Combining Labeled and Unlabeled
Data with Co-Training* (COLT 1998). La cible principale est l'erreur moyenne de
**5,0 %** du classifieur combiné sur cinq exécutions. Les résultats de contrôle
publiés sont 6,2 % pour la vue page seule et 11,6 % pour la vue liens seule.

Le [papier primaire](https://publications.ri.cmu.edu/combining-labeled-and-unlabeled-data-with-co-training)
et le [jeu officiel](https://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-51/www/co-training/data/)
définissent 1 051 pages WebKB, dont 230 pages de cours. Chaque page fournit deux
vues : le texte intégral de la page et le texte d'ancrage des liens entrants.
L'expérience réserve 263 pages au test, commence avec 3 exemples positifs et 9
négatifs, et utilise les 776 pages restantes sans étiquette.

Deux Naive Bayes sont entraînés séparément sur les deux vues d'un même ensemble
étiqueté partagé $L$. Un pool de 75 exemples est tiré de $U$. Pendant 30 tours,
chaque classifieur sélectionne exactement un exemple positif et trois négatifs.
Les huit propositions sont ajoutées dans l'ordre (vue page, puis vue liens) au
même multiensemble $L$, puis le pool est réapprovisionné par exactement huit
nouveaux exemples tant que le réservoir le permet. Si les deux vues proposent la
même page, ses deux occurrences sont conservées dans $L$, même avec des labels
opposés, mais la page source n'est retirée du pool qu'une fois. Les deux
classifieurs sont réentraînés sur ce même multiensemble $L$, chacun dans sa vue.
La prédiction finale combine les probabilités des deux vues par produit,
conformément au papier.

## Reconstruction préenregistrée

Les choix suivants sont figés sans examiner les métriques test :

- archive CMU officielle authentifiée par SHA-256 dans le fournisseur ModSSC ;
- graines 1 à 5, exactement 263 exemples test stratifiés, aucune validation ;
- 12 exemples étiquetés par allocation proportionnelle, soit 9 négatifs et 3
  positifs, puis 776 exemples non étiquetés ;
- colonne 0 pour `fulltext`, colonne 1 pour `inlinks` ;
- l'expression « bag of words appearing on the page » est reconstruite comme
  le texte visible : en-têtes MIME, balises HTML, scripts et styles sont exclus
  avant une tokenisation déterministe de `CountVectorizer`; le tokenizer
  historique exact reste inconnu ;
- comptes non binaires et matrices denses ;
- vocabulaire de chaque vue ajusté sur L+U, jamais sur le test ;
- `MultinomialNB(alpha=1.0, fit_prior=True)` dans chaque vue ;
- pool `u=75`, quotas `p=1` et `n=3`, `k=30` tours, classes positive/négative
  codées 1/0 et égalités départagées par l'ordre stable du pool ;
- multiensemble $L$ partagé : les quatre propositions de la vue page sont
  suivies des quatre propositions de la vue liens ; les recouvrements sont
  conservés deux fois, y compris avec des pseudo-étiquettes opposées ; les
  indices source sélectionnés sont retirés une seule fois du pool ;
- sélection et produit de probabilités calculés dans l'espace logarithmique,
  transformation monotone et algébriquement équivalente qui évite l'underflow
  des probabilités Naive Bayes sur les documents longs ;
- classifieur combiné par produit de probabilités, sans sélection test.

La trace de chaque tour doit conserver le pool ordonné, les promotions des deux
classifieurs, les recouvrements et conflits, les huit ajouts ordonnés avec
multiplicité, les retraits uniques, la croissance éventuelle du pool, le
réapprovisionnement et la taille du multiensemble $L$ partagé.

## Limite de la revendication

Les archives historiques ne fixent pas les cinq partitions, leur RNG, le
parseur HTML/MIME, la tokenisation, le vocabulaire exact, le lissage Naive
Bayes ou toutes les règles de conflit. La réplication peut donc atteindre au
mieux **`paper_approx`**, même si 5,0 % appartient à l'intervalle de confiance
de la nouvelle expérience.

L'analyse finale rapporte séparément les erreurs `fulltext`, `inlinks` et
combinée, ainsi que la trajectoire des pseudo-étiquettes. Une valeur combinée
proche avec des vues ou une dynamique incompatibles ne sera pas présentée
comme une réplication stricte.
