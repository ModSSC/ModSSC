# Co-Training WebKB : diagnostic et confirmation v2

## Motivation et statut

La v1 reste immuable. Ses deux classifieurs supervisés sont déjà moins bons que
les contrôles de la Table 2, puis la vue `inlinks` produit des pseudo-labels
positifs presque aléatoires. La v2 teste deux mécanismes historiques retrouvés
après le scellement de la v1. Elle constitue une reconstruction plausible, pas
une preuve que l'implémentation COLT 1998 utilisait exactement ces réglages.

Nigam et Ghani (2000), dans leur réanalyse du même corpus WebKB Course,
indiquent explicitement que Blum et Mitchell effectuaient une sélection de
variables à chaque itération. La v2 préenregistre donc une information mutuelle
empirique entre la présence du terme et la classe, recalculée séparément dans
chaque vue à chaque tour, avec les 2 000 variables les mieux classées. Les
égalités suivent l'ordre initial du vocabulaire. Le nombre 2 000 et le critère
MI sont figés avant la confirmation mais restent des détails historiques non
prouvés par l'article COLT.

Pour comparer la confiance entre documents de longueurs très différentes, la
v2 utilise l'équation (1) de Craven et al. sur le corpus WebKB :

\[
S_c(d)=\frac{\log P(c)}{n}+\sum_w P(w\mid d)
\log\frac{P(w\mid c)}{P(w\mid d)},\qquad n=\sum_w count(w,d).
\]

Un document vide reçoit uniquement le log-prior ajusté. Ce score sert seulement
au classement des candidats du pool. La prédiction finale conserve le produit
des probabilités a posteriori prescrit par Blum et Mitchell.

Références de protocole :

- Blum et Mitchell, *Combining Labeled and Unlabeled Data with Co-Training*,
  COLT 1998 ;
- Nigam et Ghani, *Analyzing the Effectiveness and Applicability of
  Co-training*, CIKM 2000 ;
- Craven et al., *Learning to Extract Symbolic Knowledge from the World Wide
  Web*, formulation WebKB, équation (1).

## Deux phases sans sélection sur le test

La carte `bench/configs/diagnostics/co_training/webkb_course_v2.yaml`, dont le
profil porte le suffixe protégé `:diagnostic-dev`, rejoue les graines 1--5 de
la v1, mais ne rapporte que `train_labeled`. La décision de
poursuivre repose sur les empreintes des variables, les scores des propositions,
les conflits, la stabilité des vues et la trajectoire des pseudo-labels. Aucun
score test n'est produit pendant ce diagnostic.

La carte `bench/configs/reproductions/co_training/webkb_course_table2_v2.yaml`
fige ensuite cinq nouvelles graines 6--10, disjointes du diagnostic. Elle tire
les 263 pages test sans stratification, conformément au texte « randomly
selected ». Elle ne doit être lancée qu'après acceptation aveugle des
trajectoires diagnostiques.

Les en-têtes MIME, balises HTML, scripts et styles sont retirés par le pipeline
déjà authentifié. La tokenisation `CountVectorizer`, `min_df=1`, le lissage
`alpha=1`, le prior empirique et la gestion des recouvrements restent explicites.
Aucun stop-list, stemming ou seuil de fréquence supplémentaire n'est ajouté :
leurs valeurs historiques exactes ne sont pas établies par les sources.

Même si la cible numérique de 5,0 % est retrouvée, la revendication reste au
maximum `paper_approx` tant que les partitions, la tokenisation et la sélection
de variables exactes de 1998 ne sont pas récupérées.
