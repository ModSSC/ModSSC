# Best Sweep Command Runbooks

All command files are in English and contain raw `python -m bench.main --config ...` lines.

## Label Percentages

- `p01`
- `p02`
- `p05`
- `p10`
- `p20`

## Modalities And Datasets

- `audio`: `speechcommands, yesno`
- `graph`: `citeseer, cora, pubmed`
- `tabular`: `adult, breast_cancer, iris, toy`
- `text`: `ag_news, amazon_polarity, amazon_reviews_multi_en, dbpedia_14, imdb, yelp_polarity, yelp_review_full`
- `vision`: `cifar10, cifar100, mnist, stl10, svhn`

## Index Files

- `01_by_dataset.md` -> split files in `by_dataset/`
- `02_by_method.md` -> split files in `by_method/`
- `03_by_modality.md` -> split files in `by_modality/`
- `04_by_percentage.md` -> split files in `by_percentage/`
- `05_lightest_per_modality.md` -> all methods x all percentages using one light dataset per modality

## Command Format

```bash
python -m bench.main --config bench/configs/experiments/best/<pXX>/<kind>/<method>/<modality>/<dataset>.yaml
```
