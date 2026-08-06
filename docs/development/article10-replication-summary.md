# Bilan de réplication des dix méthodes

Ce document est la vue lisible du registre
`provenance/article10/evidence/article10-replication-summary.json`. Il décrit
l’état scientifique au 5 août 2026. Un canari ne compte jamais comme une
répétition papier. Les statuts `paper_matched` et `paper_approx` ci-dessous
proviennent des campagnes complètes et de leurs rapports d’acceptation
empreintés.

L’identité du site d’exécution et ses localisateurs physiques restent privés.
Les noms lisibles utilisés ici sont des alias de publication, jamais des
réécritures des identifiants historiques. Les identités sources sont liées par
SHA-256 et les artefacts externes par des URI `modssc-artifact://`; le bundle de
provenance est décrit dans
`provenance/article10/evidence/execution-history-bundle.json`.

## Vue d’ensemble

| Méthode | Codée | Conformité algorithmique | Campagne papier | Répétitions | Résultat numérique | Statut |
|---|---|---|---|---:|---|---|
| Pseudo-Label | oui | passée | complète | 10/10 | erreur `6,860 %`, IC 95 % `[6,555 ; 7,165]`, article `5,03 %`; marge échouée | `paper_approx` |
| Tri-Training | oui | passée | répliquée selon la marge + extension de robustesse | 3/3 papier; 100/100 extension | erreur étendue `6,716 %`, IC 95 % `[6,308 ; 7,123]`, article `5,5 %`; marge de 2 points respectée, cible hors IC 95 % | `paper_approx` |
| Democratic Co-Learning | oui | passée pour les équations; protocole échoué; quatre constructions de confiance insuffisantes | complète | 20/20 papier; 80/80 diagnostics de confiance | accuracy papier `90,462 %`, IC 95 % `[89,460 ; 91,463]`, article `94,4 %`; marge et dynamique échouées | `paper_approx` |
| FixMatch | oui | passée | complète | 5/5 | erreur historique `5,118 %`, IC 95 % `[4,077 ; 6,159]`, article `5,07 %`; erreur terminale `5,124 %` | `paper_matched` |
| FlexMatch | oui | passée | complète | 3/3 | erreur historique `5,107 %`, IC 95 % `[4,881 ; 5,332]`, article `4,98 %`; erreur terminale `5,267 %` | `paper_matched` |
| FreeMatch | oui | passée | complète | 3/3 | erreur historique `5,377 %`, IC 95 % `[4,437 ; 6,316]`, article `4,90 %`; erreur terminale `5,623 %`, variance élevée | `paper_matched` |
| SoftMatch | oui | passée | complète | 3/3 | erreur historique `4,870 %`, IC 95 % `[4,673 ; 5,067]`, article `4,82 %`; erreur terminale `5,083 %` | `paper_matched` |
| Laplace Learning | oui | passée | complète | 500/500 | accuracy de `16,060 %` à `69,462 %` pour 1–5 labels/classe; dix critères de cellule respectés | `paper_matched` |
| Poisson Learning | oui | passée | complète | 500/500 | accuracy de `90,176 %` à `95,294 %` pour 1–5 labels/classe; dix critères de cellule respectés | `paper_matched` |
| GRAND | oui | passée | complète | 100/100 | accuracy `85,366 %`, IC 95 % `[85,300 ; 85,432]`, article `85,4 %`; compatible | `paper_matched` |

## Consommation des productions

| Méthode | Architecture | Heures accélérateur | Rapport journalier | SHA-256 |
|---|---|---:|---|---|
| GRAND | V100 | `1.5933824413888888` | `daily/article10-grand-paper-production-v1-v5-001/daily-usage.json` | `9ede52ac93490b6790f432954471662654e1b575a7c9996d9bc6bd830de59bcd` |
| Tri-Training | V100 | `0.009924006666666667` | `daily/article10-paper-tri-vote-v1-v6-001/daily-usage.json` | `a22f86b8e5d5be665e8694a3818e72eae3e0d4d7677206d0e2b0554930d4188a` |
| Tri-Training, extension 100 tirages | V100 | `0.26006854` | `daily/article10-paper-tri-vote-extended100-v1-002/daily-usage.json` | `ee4348907e68e133eec473228d4bcdcab52eb1464862d1433a1df66657f58e71` |
| Pseudo-Label | A100 | `0.6074879725` | `daily/article10-paper-pseudo-label-mnist-v1-v6-001/daily-usage.json` | `1424a5595388d5b20e58ea3d0ce32cc30eadd9850391f1961088741b80401eb8` |
| DCL, diagnostics de confiance | V100 | `2.1883333333333335` | descriptor public `provenance/article10/evidence/dcl-confidence-v8-resource-usage.json` | `f4fafb0e5a2fb83a38bea7bb888eca0a8444d4fd6422a4c67441112aa437aa3d` |

La réserve de `15 %` est respectée (`reserve_status: pass`). Ces nombres
proviennent des rapports journaliers des campagnes de production, pas des
canaris. La campagne Calder locale a utilisé au plus deux processus, terminé
ses 1 000 tâches sans échec et duré 11 370 secondes.

Cette table ne prétend pas être exhaustive pour la vague Match : aucun
`daily-usage.json` Match scellé n’est disponible. Les décisions scientifiques
Match reposent sur les rapports d’acceptation et de réconciliation empreintés,
pas sur une estimation reconstruite des heures H100.

La ligne DCL est un audit Slurm séparé des productions papier. Elle inclut les
80 tâches scientifiques ainsi que les allocations de génération, préflight,
réconciliation et évaluation. Son total de `2,1884` heures V100 respecte le
plafond diagnostique préenregistré de 5 heures. Le descriptor public est dérivé
du rapport privé immuable de SHA-256
`22a20d62130fae3c2e2e5fef5615b87f3190c2e4fc5decc424592067a0a0be5e` ;
il conserve les agrégats mais retire le site, le compte et les identifiants de
jobs.

## Décisions définitives

GRAND est la première réplication exacte de cette sélection. Les 100 graines
officielles ont réussi, la moyenne publiée est dans l’IC 95 %, l’écart absolu
est de `0,034` point et les diagnostics requis sont complets. Son statut est
donc `paper_matched`.

Laplace Learning et Poisson Learning répliquent les dix cellules de la Table 1
Calder. Chaque cellule contient 100/100 permutations officielles. Les dix
valeurs publiées appartiennent aux IC 95 % de la réplication, chaque écart
absolu respecte la marge d'un point, les diagnostics sont complets et les
écarts-types publiés sont reconstruits avec `ddof=0`. Les deux méthodes sont
`paper_matched`.

La vague Match a terminé ses 14 répétitions papier. FixMatch atteint `5,118 %`
d’erreur historique contre `5,07 %` dans l’article; FlexMatch `5,107 %` contre
`4,98 %`; FreeMatch `5,377 %` contre `4,90 %`; et SoftMatch `4,870 %` contre
`4,82 %`. Pour les quatre méthodes, la cible publiée appartient à l’IC 95 %,
la marge d’un point et les diagnostics préenregistrés sont respectés. Elles
sont donc toutes `paper_matched`. FreeMatch conserve une réserve explicite :
sa dispersion (`0,378` point) est sensiblement supérieure à celle publiée.
Les métriques terminales sans sélection sur le test sont conservées séparément
et seront les seules utilisables dans le benchmark standardisé.

Tri-Training a réussi ses trois répétitions papier, puis une extension
préenregistrée de 100 tirages distincts. L’extension obtient une erreur finale
moyenne de `6,716 %` contre `5,5 %` dans l’article : l’écart de `1,216` point
respecte la marge de deux points, mais la valeur publiée est hors de l’IC 95 %
resserré `[6,308 ; 7,123]`. L’erreur initiale moyenne est `6,339 %`, IC 95 %
`[5,903 ; 6,776]`, contre `7,6 %` dans l’article; elle respecte également la
marge de deux points mais reste hors de l’IC.

L’analyse appariée montre une différence moyenne
`erreur finale - erreur initiale` de `+0,376` point, IC 95 %
`[-0,062 ; +0,815]` : 28 tirages s’améliorent, 36 se dégradent et 36 restent
inchangés. Les graines 1 à 3 rejouent exactement la campagne initiale, les 100
partitions étiquetées/non étiquetées sont uniques et le test est resté fixe.
L’augmentation du nombre de tirages montre donc que l’écart est systématique
dans le protocole reconstruit, et non un accident des trois premières graines.
Les indices historiques, l’état RNG et certains choix J48/vote restent
indisponibles; le statut final reste `paper_approx` et l’extension a
`result_status: failed_ci95`. Tri-Training est donc enregistré comme
**répliqué selon la marge d’équivalence préenregistrée** de deux points :
`6,716 %` appartient à la bande `[3,5 ; 7,5] %` autour de la cible `5,5 %`.
La cible publiée reste toutefois hors de l’IC 95 % de la réplication; ce
résultat n’est donc pas `paper_matched`.

Pseudo-Label a réussi ses dix répétitions, mais l’erreur moyenne de `6,86 %`
diffère de `1,83` point de la valeur publiée et celle-ci n’appartient pas à
l’IC 95 % de la réplication. Le statut final est
`paper_approx/failed_margin`.

Democratic Co-Learning a réussi ses vingt répétitions techniques. Sa moyenne
est inférieure de `3,938` points à l’article. Les contrôles Naive Bayes et
Combining Only ainsi que la dynamique de la Table 2 sont incompatibles avec la
publication. Les équations sont validées, mais la conformité du protocole est
échouée : `paper_approx/failed_margin`.

Une extension diagnostique préenregistrée, distincte de la réplication papier,
a ensuite comparé sur les mêmes 20 partitions la resubstitution + Wald, la
validation croisée 10-fold + Wald, puis, après déclenchement de la branche
conditionnelle, 10-fold + Wilson et 10-fold + Clopper–Pearson. Les quatre
cellules ont réussi leurs 20/20 tâches sans utiliser de métrique test, mais
aucune ne reproduit la dynamique de la Table 2. Les moyennes
`tours / ajouts [NB, C4.5, 3-NN]` sont respectivement
`5,55 / [4,45, 42,80, 5,60]`, `5,35 / [4,05, 40,85, 4,60]`,
`5,10 / [3,90, 38,90, 3,65]` et
`4,95 / [3,95, 35,20, 3,60]`, contre
`2,2 / [66, 40, 40]` dans l’article. L’inversion apparaît dès les désaccords
bruts du premier tour, avant le filtrage par les intervalles. Le choix de
l’intervalle n’est donc pas une explication suffisante; les sémantiques des
apprenants historiques, les partitions originales indisponibles ou leur
interaction restent les causes plausibles. Cette extension ne crée ni une
nouvelle réplication papier, ni un droit à une revendication supplémentaire.

## Table 1 Calder reconstruite

| Labels/classe | Laplace réplication | IC 95 % Laplace | Laplace article | Poisson réplication | IC 95 % Poisson | Poisson article |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16,06 ± 6,21 | [14,82 ; 17,30] | 16,1 ± 6,2 | 90,18 ± 3,99 | [89,38 ; 90,97] | 90,2 ± 4,0 |
| 2 | 28,20 ± 10,30 | [26,15 ; 30,25] | 28,2 ± 10,3 | 93,59 ± 1,57 | [93,28 ; 93,90] | 93,6 ± 1,6 |
| 3 | 41,97 ± 12,37 | [39,51 ; 44,44] | 42,0 ± 12,4 | 94,53 ± 1,08 | [94,32 ; 94,75] | 94,5 ± 1,1 |
| 4 | 57,81 ± 12,30 | [55,36 ; 60,26] | 57,8 ± 12,3 | 94,95 ± 0,82 | [94,78 ; 95,11] | 94,9 ± 0,8 |
| 5 | 69,46 ± 12,16 | [67,04 ; 71,89] | 69,5 ± 12,2 | 95,29 ± 0,73 | [95,15 ; 95,44] | 95,3 ± 0,7 |

## Vague Match terminée

FixMatch, FlexMatch, FreeMatch et SoftMatch ont franchi la conformité
algorithmique et protocolaire, puis terminé 14/14 répétitions. Les décisions
signées sont `paper_matched` pour les quatre profils papier. La piste
`standardized` reste explicitement `pending` : elle utilisera les checkpoints
terminaux et jamais la convention historique de sélection sur le test. Les
canaris demeurent des diagnostics non reportables et ne sont pas mélangés aux
productions.

## Preuves principales

| Méthode | Rapport d’acceptation ou preuve | SHA-256 |
|---|---|---|
| FixMatch | `paper-acceptance/match-adaptive-v2/fixmatch-source-refresh-v1/paper-acceptance.json` | `f80a675aa3a0463021577614c0334dde9ab3904ab7b9669e923cf0e8f8c4a9d9` |
| FlexMatch / FreeMatch / SoftMatch | `acceptance/match-adaptive-v2/article10-match-adaptive-production-v2-final-v1/paper-acceptance.json` | `b80e95ac13941b8805fcf629bdd833c825ff576bf8b1163496e7e326bfa26475` |
| GRAND | `paper-evaluations/article10-grand-paper-production-v1-v5-001/paper-acceptance.json` | `0c8a48a217ae4da243eac3a8ccbe117742212c9345beab615733986a2da73ddc` |
| Tri-Training, 3 répétitions papier | `paper-evaluations/article10-paper-tri-vote-v1-v6-001/paper-acceptance.json` | `495cf0abafb7575db39c835dc3d162b12e93b5324767a83cca2b0261bd2999a6` |
| Tri-Training, extension 100 tirages | `paper-evaluations/article10-paper-tri-vote-extended100-v1-002/paper-acceptance.json` | `e6d007ab44742c2b24b3a2707fb9f00aa8331867470ea05725d1b95007c179fe` |
| Tri-Training, analyse appariée | `audits/article10-paper-tri-vote-extended100-v1/paired-analysis-v1/robustness-analysis.json` | `4c1cdf1cb24c5a5b1d63b10d7eb9d986d5ff0aefa69f51e7bd6b465bebaaf5c5` |
| Pseudo-Label | `paper-evaluations/article10-paper-pseudo-label-mnist-v1-v6-001/paper-acceptance.json` | `6e656b42d8d24edf10c2b62fe7afbc4c72bd57f3ef12d709a6373f896d83b34e` |
| Democratic Co-Learning | `modssc-artifact://replication/evidence/d66677565b5968daade77bfa252c2178859aaad8390732f6acbae9364dc61dcc` | `d66677565b5968daade77bfa252c2178859aaad8390732f6acbae9364dc61dcc` |
| DCL, diagnostic primaire | `modssc-artifact://replication/evidence/42e6805a991ce4230d6158c7fdf20382884b53c86601d9a4437590faeb362990` | `42e6805a991ce4230d6158c7fdf20382884b53c86601d9a4437590faeb362990` |
| DCL, attribution premier tour | `modssc-artifact://replication/evidence/565bd6f5c453bdfc7cb13bd8f04b4e75c3f8db45d08f3eef816ecf8e1221069e` | `565bd6f5c453bdfc7cb13bd8f04b4e75c3f8db45d08f3eef816ecf8e1221069e` |
| DCL, diagnostic conditionnel | `modssc-artifact://replication/evidence/8d79278dfb3a3b40718d48ad24b81d5d036dd0b16a8a3b54ef844db8924c37dc` | `8d79278dfb3a3b40718d48ad24b81d5d036dd0b16a8a3b54ef844db8924c37dc` |
| Laplace / Poisson | `calder/evaluation/paper-acceptance.json` | `5c5a149db34c531a6a87dbe99f6e9b6152b4c87ee6d1a773cab4381f15b2673a` |
| Table 1 Calder | `calder/evaluation/calder-table1-reconstructed.json` | `a179cd1b505221124835e4feb8768c40cb720a5272012cb4663222a1d6e88ede` |

Toutes les valeurs exactes, les manifestes de campagne, les réconciliations
et les preuves secondaires sont conservés dans le registre JSON. Les 9 000
fichiers de résultat et de trace Calder sont conservés hors du dépôt dans le
bundle logique `evidence://modssc/calder/table1/final-bundle-v4`.
Le manifeste final brut a pour SHA-256
`ed70f03ce70023438466a60d9c14a21e27ffb7e5e711eb2ed04b0bfb374bf143`.
