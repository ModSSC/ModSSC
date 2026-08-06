# Exécution locale des remplacements historiques — vague 2

!!! note "Historique"
    Cette page conserve le déroulé exact de la vague 2. Pour une nouvelle
    réplication, la voie canonique est `python -m bench.reproduce`: elle prépare
    les datasets via les providers ModSSC et authentifie les entrées fixes avant
    d'exécuter une carte. Aucun cache préchargé manuellement n'est requis.

Ce flux exécute seulement les trois cartes déjà figées :

1. confirmation Self-Training sur Wine, graines 51 à 100 ;
2. diagnostic Co-Training WebKB sans métrique test, graines 1 à 5 ;
3. confirmation Co-Training WebKB, graines nouvelles 6 à 10.

Il n'utilise pas la campagne générique : ces 60 exécutions CPU sont courtes et
`bench.main` produit déjà un `run.json` et une partition rejouable par graine.
Les validateurs authentifient ensuite le commit, l'environnement, les cartes,
les partitions et la complétude.

## Préconditions bloquantes

- Travailler depuis un commit propre et approuvé. Ne jamais accepter une sortie
  ayant `git_dirty=true`.
- Utiliser un répertoire de sortie neuf. Une tentative interrompue est conservée
  et la vague est rejouée dans un autre répertoire ; aucun artefact n'est écrasé.
- Préparer `wine` et `webkb_course_cotraining` par la commande autonome; ne pas
  copier manuellement un cache provenant d'une autre exécution.
- Ne pas lancer la confirmation Co-Training avant que le sceau du diagnostic
  ait été créé avec un code retour nul.

Exemple de préparation, à adapter à la machine locale :

```bash
test -z "$(git status --porcelain)"
export APPROVED_COMMIT="$(git rev-parse HEAD)"
export MODSSC_DATASET_CACHE_DIR="/chemin/vers/modssc_cache/datasets"
export MODSSC_PREPROCESS_CACHE_DIR="/chemin/vers/modssc_cache/preprocess"
export MODSSC_OUTPUT_DIR="/chemin/vers/resultats/historical-wave2-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$MODSSC_OUTPUT_DIR"
python -m bench.reproduce prepare self_training/wine_table3_confirmation_v2
python -m bench.reproduce prepare co_training/webkb_course_table2_v2
```

La valeur de `MODSSC_OUTPUT_DIR` devient l'identité physique de cette tentative
et ne doit plus être changée pendant la séquence.

## 1. Self-Training — confirmation indépendante

```bash
python -m bench.reproduce run self_training/wine_table3_confirmation_v2

python -m bench.campaign.acceptance.historical \
  --protocol self-training-v2 \
  --sweep-root "$MODSSC_OUTPUT_DIR/reproductions/self_training/wine_table3_confirmation_v2" \
  --expected-git-sha "$APPROVED_COMMIT" \
  --output-json "$MODSSC_OUTPUT_DIR/acceptance/self-training-v2.json" \
  --output-tsv "$MODSSC_OUTPUT_DIR/acceptance/self-training-v2.tsv"
```

Le validateur exige les 50 graines 51 à 100, des partitions distinctes et
rejouables, le même environnement, ainsi que le contrat du pool et de la
confiance reconstruits avant la campagne.

## 2. Co-Training — diagnostic sans métrique test

```bash
python -m bench.main \
  --config bench/configs/diagnostics/co_training/webkb_course_v2.yaml

python -m bench.campaign.acceptance.diagnostics \
  --sweep-root "$MODSSC_OUTPUT_DIR/diagnostics/co_training/webkb_course_v2" \
  --diagnostic-card bench/configs/diagnostics/co_training/webkb_course_v2.yaml \
  --confirmation-card bench/configs/reproductions/co_training/webkb_course_table2_v2.yaml \
  --expected-git-sha "$APPROVED_COMMIT" \
  --output-json "$MODSSC_OUTPUT_DIR/acceptance/co-training-v2-diagnostic-seal.json"
```

Ce garde-fou échoue notamment si :

- une des cinq graines manque ou est dupliquée ;
- un `run.json` contient une métrique `test` ou une partition rapportée autre
  que `train_labeled` et ses deux vues ;
- le commit est sale, différent ou l'environnement varie entre les graines ;
- une partition n'est pas authentifiable ;
- la sélection dynamique, le score de Craven ou leurs traces sur 30 tours ne
  correspondent pas à la carte ;
- la carte de confirmation a changé depuis le diagnostic ;
- le fichier de sceau existe déjà.

Le qualificatif « sans métrique test » est volontairement précis : les graines
1 à 5 rejouent des partitions dont les résultats v1 ont déjà été observés. Ce
diagnostic localise donc les mécanismes après coup ; il n'est pas strictement
aveugle au sens épistémique. La vérification réellement nouvelle est portée par
les graines 6 à 10.

## 3. Co-Training — confirmation sur graines nouvelles

Cette commande n'est autorisée que si le garde-fou précédent retourne zéro et
si son sceau est conservé sans modification.

```bash
python -m bench.reproduce run co_training/webkb_course_table2_v2

python -m bench.campaign.acceptance.historical \
  --protocol co-training-v2 \
  --sweep-root "$MODSSC_OUTPUT_DIR/reproductions/co_training/webkb_course_table2_v2" \
  --expected-git-sha "$APPROVED_COMMIT" \
  --output-json "$MODSSC_OUTPUT_DIR/acceptance/co-training-v2.json" \
  --output-tsv "$MODSSC_OUTPUT_DIR/acceptance/co-training-v2.tsv"
```

L'acceptation finale reste plafonnée à `paper_approx`, même si la cible
numérique est retrouvée : les graines historiques, la tokenisation de 1998 et
le sélecteur exact ne sont pas disponibles. Le résultat ne devient reportable
que si le validateur confirme cinq succès, l'intervalle de confiance, la marge
de deux points et les contrôles secondaires des deux vues.

## Reprise et conservation

`bench.main` n'est pas un ordonnanceur de reprise. En cas d'interruption :

1. conserver intégralement le répertoire incomplet ;
2. choisir un nouveau `MODSSC_OUTPUT_DIR` ;
3. recommencer seulement la carte concernée ;
4. ne valider que le nouveau répertoire complet.

Les trois JSON d'acceptation, les deux TSV, tous les `run.json`, les répertoires
`sampling_split` et le commit Git forment le paquet scientifique minimal à
archiver.
