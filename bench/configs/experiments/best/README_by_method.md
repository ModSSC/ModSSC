# Best configs (commands by method)

## inductive/adamatch

```
mkdir -p log/inductive/adamatch/audio log/inductive/adamatch/graph log/inductive/adamatch/tabular log/inductive/adamatch/text log/inductive/adamatch/vision
python -m bench.main --config bench/configs/experiments/best/inductive/adamatch/audio/speechcommands.yaml > log/inductive/adamatch/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adamatch/audio/yesno.yaml > log/inductive/adamatch/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adamatch/graph/citeseer.yaml > log/inductive/adamatch/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adamatch/graph/cora.yaml > log/inductive/adamatch/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adamatch/graph/pubmed.yaml > log/inductive/adamatch/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adamatch/tabular/adult.yaml > log/inductive/adamatch/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adamatch/tabular/breast_cancer.yaml > log/inductive/adamatch/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adamatch/tabular/iris.yaml > log/inductive/adamatch/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adamatch/tabular/toy.yaml > log/inductive/adamatch/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adamatch/text/ag_news.yaml > log/inductive/adamatch/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adamatch/text/amazon_polarity.yaml > log/inductive/adamatch/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adamatch/text/amazon_reviews_multi_en.yaml > log/inductive/adamatch/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adamatch/text/dbpedia_14.yaml > log/inductive/adamatch/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adamatch/text/imdb.yaml > log/inductive/adamatch/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adamatch/text/yelp_polarity.yaml > log/inductive/adamatch/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adamatch/text/yelp_review_full.yaml > log/inductive/adamatch/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adamatch/vision/cifar10.yaml > log/inductive/adamatch/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adamatch/vision/cifar100.yaml > log/inductive/adamatch/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adamatch/vision/mnist.yaml > log/inductive/adamatch/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adamatch/vision/stl10.yaml > log/inductive/adamatch/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adamatch/vision/svhn.yaml > log/inductive/adamatch/vision/svhn.log 2>&1
```

## inductive/adsh

```
mkdir -p log/inductive/adsh/audio log/inductive/adsh/graph log/inductive/adsh/tabular log/inductive/adsh/text log/inductive/adsh/vision
python -m bench.main --config bench/configs/experiments/best/inductive/adsh/audio/speechcommands.yaml > log/inductive/adsh/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adsh/audio/yesno.yaml > log/inductive/adsh/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adsh/graph/citeseer.yaml > log/inductive/adsh/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adsh/graph/cora.yaml > log/inductive/adsh/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adsh/graph/pubmed.yaml > log/inductive/adsh/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adsh/tabular/adult.yaml > log/inductive/adsh/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adsh/tabular/breast_cancer.yaml > log/inductive/adsh/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adsh/tabular/iris.yaml > log/inductive/adsh/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adsh/tabular/toy.yaml > log/inductive/adsh/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adsh/text/ag_news.yaml > log/inductive/adsh/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adsh/text/amazon_polarity.yaml > log/inductive/adsh/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adsh/text/amazon_reviews_multi_en.yaml > log/inductive/adsh/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adsh/text/dbpedia_14.yaml > log/inductive/adsh/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adsh/text/imdb.yaml > log/inductive/adsh/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adsh/text/yelp_polarity.yaml > log/inductive/adsh/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adsh/text/yelp_review_full.yaml > log/inductive/adsh/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adsh/vision/cifar10.yaml > log/inductive/adsh/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adsh/vision/cifar100.yaml > log/inductive/adsh/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adsh/vision/mnist.yaml > log/inductive/adsh/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adsh/vision/stl10.yaml > log/inductive/adsh/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/adsh/vision/svhn.yaml > log/inductive/adsh/vision/svhn.log 2>&1
```

## inductive/co_training

```
mkdir -p log/inductive/co_training/audio log/inductive/co_training/graph log/inductive/co_training/tabular log/inductive/co_training/text log/inductive/co_training/vision
python -m bench.main --config bench/configs/experiments/best/inductive/co_training/audio/speechcommands.yaml > log/inductive/co_training/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/co_training/audio/yesno.yaml > log/inductive/co_training/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/co_training/graph/citeseer.yaml > log/inductive/co_training/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/co_training/graph/cora.yaml > log/inductive/co_training/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/co_training/graph/pubmed.yaml > log/inductive/co_training/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/co_training/tabular/adult.yaml > log/inductive/co_training/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/co_training/tabular/breast_cancer.yaml > log/inductive/co_training/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/co_training/tabular/iris.yaml > log/inductive/co_training/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/co_training/tabular/toy.yaml > log/inductive/co_training/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/co_training/text/ag_news.yaml > log/inductive/co_training/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/co_training/text/amazon_polarity.yaml > log/inductive/co_training/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/co_training/text/amazon_reviews_multi_en.yaml > log/inductive/co_training/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/co_training/text/dbpedia_14.yaml > log/inductive/co_training/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/co_training/text/imdb.yaml > log/inductive/co_training/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/co_training/text/yelp_polarity.yaml > log/inductive/co_training/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/co_training/text/yelp_review_full.yaml > log/inductive/co_training/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/co_training/vision/cifar10.yaml > log/inductive/co_training/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/co_training/vision/cifar100.yaml > log/inductive/co_training/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/co_training/vision/mnist.yaml > log/inductive/co_training/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/co_training/vision/stl10.yaml > log/inductive/co_training/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/co_training/vision/svhn.yaml > log/inductive/co_training/vision/svhn.log 2>&1
```

## inductive/comatch

```
mkdir -p log/inductive/comatch/audio log/inductive/comatch/graph log/inductive/comatch/tabular log/inductive/comatch/text log/inductive/comatch/vision
python -m bench.main --config bench/configs/experiments/best/inductive/comatch/audio/speechcommands.yaml > log/inductive/comatch/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/comatch/audio/yesno.yaml > log/inductive/comatch/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/comatch/graph/citeseer.yaml > log/inductive/comatch/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/comatch/graph/cora.yaml > log/inductive/comatch/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/comatch/graph/pubmed.yaml > log/inductive/comatch/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/comatch/tabular/adult.yaml > log/inductive/comatch/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/comatch/tabular/breast_cancer.yaml > log/inductive/comatch/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/comatch/tabular/iris.yaml > log/inductive/comatch/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/comatch/tabular/toy.yaml > log/inductive/comatch/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/comatch/text/ag_news.yaml > log/inductive/comatch/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/comatch/text/amazon_polarity.yaml > log/inductive/comatch/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/comatch/text/amazon_reviews_multi_en.yaml > log/inductive/comatch/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/comatch/text/dbpedia_14.yaml > log/inductive/comatch/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/comatch/text/imdb.yaml > log/inductive/comatch/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/comatch/text/yelp_polarity.yaml > log/inductive/comatch/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/comatch/text/yelp_review_full.yaml > log/inductive/comatch/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/comatch/vision/cifar10.yaml > log/inductive/comatch/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/comatch/vision/cifar100.yaml > log/inductive/comatch/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/comatch/vision/mnist.yaml > log/inductive/comatch/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/comatch/vision/stl10.yaml > log/inductive/comatch/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/comatch/vision/svhn.yaml > log/inductive/comatch/vision/svhn.log 2>&1
```

## inductive/daso

```
mkdir -p log/inductive/daso/audio log/inductive/daso/graph log/inductive/daso/tabular log/inductive/daso/text log/inductive/daso/vision
python -m bench.main --config bench/configs/experiments/best/inductive/daso/audio/speechcommands.yaml > log/inductive/daso/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/daso/audio/yesno.yaml > log/inductive/daso/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/daso/graph/citeseer.yaml > log/inductive/daso/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/daso/graph/cora.yaml > log/inductive/daso/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/daso/graph/pubmed.yaml > log/inductive/daso/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/daso/tabular/adult.yaml > log/inductive/daso/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/daso/tabular/breast_cancer.yaml > log/inductive/daso/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/daso/tabular/iris.yaml > log/inductive/daso/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/daso/tabular/toy.yaml > log/inductive/daso/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/daso/text/ag_news.yaml > log/inductive/daso/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/daso/text/amazon_polarity.yaml > log/inductive/daso/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/daso/text/amazon_reviews_multi_en.yaml > log/inductive/daso/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/daso/text/dbpedia_14.yaml > log/inductive/daso/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/daso/text/imdb.yaml > log/inductive/daso/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/daso/text/yelp_polarity.yaml > log/inductive/daso/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/daso/text/yelp_review_full.yaml > log/inductive/daso/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/daso/vision/cifar10.yaml > log/inductive/daso/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/daso/vision/cifar100.yaml > log/inductive/daso/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/daso/vision/mnist.yaml > log/inductive/daso/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/daso/vision/stl10.yaml > log/inductive/daso/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/daso/vision/svhn.yaml > log/inductive/daso/vision/svhn.log 2>&1
```

## inductive/deep_co_training

```
mkdir -p log/inductive/deep_co_training/audio log/inductive/deep_co_training/graph log/inductive/deep_co_training/tabular log/inductive/deep_co_training/text log/inductive/deep_co_training/vision
python -m bench.main --config bench/configs/experiments/best/inductive/deep_co_training/audio/speechcommands.yaml > log/inductive/deep_co_training/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/deep_co_training/audio/yesno.yaml > log/inductive/deep_co_training/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/deep_co_training/graph/citeseer.yaml > log/inductive/deep_co_training/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/deep_co_training/graph/cora.yaml > log/inductive/deep_co_training/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/deep_co_training/graph/pubmed.yaml > log/inductive/deep_co_training/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/deep_co_training/tabular/adult.yaml > log/inductive/deep_co_training/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/deep_co_training/tabular/breast_cancer.yaml > log/inductive/deep_co_training/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/deep_co_training/tabular/iris.yaml > log/inductive/deep_co_training/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/deep_co_training/tabular/toy.yaml > log/inductive/deep_co_training/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/deep_co_training/text/ag_news.yaml > log/inductive/deep_co_training/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/deep_co_training/text/amazon_polarity.yaml > log/inductive/deep_co_training/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/deep_co_training/text/amazon_reviews_multi_en.yaml > log/inductive/deep_co_training/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/deep_co_training/text/dbpedia_14.yaml > log/inductive/deep_co_training/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/deep_co_training/text/imdb.yaml > log/inductive/deep_co_training/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/deep_co_training/text/yelp_polarity.yaml > log/inductive/deep_co_training/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/deep_co_training/text/yelp_review_full.yaml > log/inductive/deep_co_training/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/deep_co_training/vision/cifar10.yaml > log/inductive/deep_co_training/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/deep_co_training/vision/cifar100.yaml > log/inductive/deep_co_training/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/deep_co_training/vision/mnist.yaml > log/inductive/deep_co_training/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/deep_co_training/vision/stl10.yaml > log/inductive/deep_co_training/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/deep_co_training/vision/svhn.yaml > log/inductive/deep_co_training/vision/svhn.log 2>&1
```

## inductive/defixmatch

```
mkdir -p log/inductive/defixmatch/audio log/inductive/defixmatch/graph log/inductive/defixmatch/tabular log/inductive/defixmatch/text log/inductive/defixmatch/vision
python -m bench.main --config bench/configs/experiments/best/inductive/defixmatch/audio/speechcommands.yaml > log/inductive/defixmatch/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/defixmatch/audio/yesno.yaml > log/inductive/defixmatch/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/defixmatch/graph/citeseer.yaml > log/inductive/defixmatch/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/defixmatch/graph/cora.yaml > log/inductive/defixmatch/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/defixmatch/graph/pubmed.yaml > log/inductive/defixmatch/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/defixmatch/tabular/adult.yaml > log/inductive/defixmatch/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/defixmatch/tabular/breast_cancer.yaml > log/inductive/defixmatch/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/defixmatch/tabular/iris.yaml > log/inductive/defixmatch/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/defixmatch/tabular/toy.yaml > log/inductive/defixmatch/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/defixmatch/text/ag_news.yaml > log/inductive/defixmatch/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/defixmatch/text/amazon_polarity.yaml > log/inductive/defixmatch/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/defixmatch/text/amazon_reviews_multi_en.yaml > log/inductive/defixmatch/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/defixmatch/text/dbpedia_14.yaml > log/inductive/defixmatch/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/defixmatch/text/imdb.yaml > log/inductive/defixmatch/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/defixmatch/text/yelp_polarity.yaml > log/inductive/defixmatch/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/defixmatch/text/yelp_review_full.yaml > log/inductive/defixmatch/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/defixmatch/vision/cifar10.yaml > log/inductive/defixmatch/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/defixmatch/vision/cifar100.yaml > log/inductive/defixmatch/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/defixmatch/vision/mnist.yaml > log/inductive/defixmatch/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/defixmatch/vision/stl10.yaml > log/inductive/defixmatch/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/defixmatch/vision/svhn.yaml > log/inductive/defixmatch/vision/svhn.log 2>&1
```

## inductive/democratic_co_learning

```
mkdir -p log/inductive/democratic_co_learning/audio log/inductive/democratic_co_learning/graph log/inductive/democratic_co_learning/tabular log/inductive/democratic_co_learning/text log/inductive/democratic_co_learning/vision
python -m bench.main --config bench/configs/experiments/best/inductive/democratic_co_learning/audio/speechcommands.yaml > log/inductive/democratic_co_learning/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/democratic_co_learning/audio/yesno.yaml > log/inductive/democratic_co_learning/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/democratic_co_learning/graph/citeseer.yaml > log/inductive/democratic_co_learning/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/democratic_co_learning/graph/cora.yaml > log/inductive/democratic_co_learning/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/democratic_co_learning/graph/pubmed.yaml > log/inductive/democratic_co_learning/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/democratic_co_learning/tabular/adult.yaml > log/inductive/democratic_co_learning/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/democratic_co_learning/tabular/breast_cancer.yaml > log/inductive/democratic_co_learning/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/democratic_co_learning/tabular/iris.yaml > log/inductive/democratic_co_learning/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/democratic_co_learning/tabular/toy.yaml > log/inductive/democratic_co_learning/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/democratic_co_learning/text/ag_news.yaml > log/inductive/democratic_co_learning/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/democratic_co_learning/text/amazon_polarity.yaml > log/inductive/democratic_co_learning/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/democratic_co_learning/text/amazon_reviews_multi_en.yaml > log/inductive/democratic_co_learning/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/democratic_co_learning/text/dbpedia_14.yaml > log/inductive/democratic_co_learning/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/democratic_co_learning/text/imdb.yaml > log/inductive/democratic_co_learning/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/democratic_co_learning/text/yelp_polarity.yaml > log/inductive/democratic_co_learning/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/democratic_co_learning/text/yelp_review_full.yaml > log/inductive/democratic_co_learning/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/democratic_co_learning/vision/cifar10.yaml > log/inductive/democratic_co_learning/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/democratic_co_learning/vision/cifar100.yaml > log/inductive/democratic_co_learning/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/democratic_co_learning/vision/mnist.yaml > log/inductive/democratic_co_learning/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/democratic_co_learning/vision/stl10.yaml > log/inductive/democratic_co_learning/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/democratic_co_learning/vision/svhn.yaml > log/inductive/democratic_co_learning/vision/svhn.log 2>&1
```

## inductive/fixmatch

```
mkdir -p log/inductive/fixmatch/audio log/inductive/fixmatch/graph log/inductive/fixmatch/tabular log/inductive/fixmatch/text log/inductive/fixmatch/vision
python -m bench.main --config bench/configs/experiments/best/inductive/fixmatch/audio/speechcommands.yaml > log/inductive/fixmatch/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/fixmatch/audio/yesno.yaml > log/inductive/fixmatch/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/fixmatch/graph/citeseer.yaml > log/inductive/fixmatch/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/fixmatch/graph/cora.yaml > log/inductive/fixmatch/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/fixmatch/graph/pubmed.yaml > log/inductive/fixmatch/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/fixmatch/tabular/adult.yaml > log/inductive/fixmatch/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/fixmatch/tabular/breast_cancer.yaml > log/inductive/fixmatch/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/fixmatch/tabular/iris.yaml > log/inductive/fixmatch/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/fixmatch/tabular/toy.yaml > log/inductive/fixmatch/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/fixmatch/text/ag_news.yaml > log/inductive/fixmatch/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/fixmatch/text/amazon_polarity.yaml > log/inductive/fixmatch/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/fixmatch/text/amazon_reviews_multi_en.yaml > log/inductive/fixmatch/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/fixmatch/text/dbpedia_14.yaml > log/inductive/fixmatch/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/fixmatch/text/imdb.yaml > log/inductive/fixmatch/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/fixmatch/text/yelp_polarity.yaml > log/inductive/fixmatch/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/fixmatch/text/yelp_review_full.yaml > log/inductive/fixmatch/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/fixmatch/vision/cifar10.yaml > log/inductive/fixmatch/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/fixmatch/vision/cifar100.yaml > log/inductive/fixmatch/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/fixmatch/vision/mnist.yaml > log/inductive/fixmatch/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/fixmatch/vision/stl10.yaml > log/inductive/fixmatch/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/fixmatch/vision/svhn.yaml > log/inductive/fixmatch/vision/svhn.log 2>&1
```

## inductive/flexmatch

```
mkdir -p log/inductive/flexmatch/audio log/inductive/flexmatch/graph log/inductive/flexmatch/tabular log/inductive/flexmatch/text log/inductive/flexmatch/vision
python -m bench.main --config bench/configs/experiments/best/inductive/flexmatch/audio/speechcommands.yaml > log/inductive/flexmatch/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/flexmatch/audio/yesno.yaml > log/inductive/flexmatch/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/flexmatch/graph/citeseer.yaml > log/inductive/flexmatch/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/flexmatch/graph/cora.yaml > log/inductive/flexmatch/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/flexmatch/graph/pubmed.yaml > log/inductive/flexmatch/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/flexmatch/tabular/adult.yaml > log/inductive/flexmatch/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/flexmatch/tabular/breast_cancer.yaml > log/inductive/flexmatch/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/flexmatch/tabular/iris.yaml > log/inductive/flexmatch/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/flexmatch/tabular/toy.yaml > log/inductive/flexmatch/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/flexmatch/text/ag_news.yaml > log/inductive/flexmatch/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/flexmatch/text/amazon_polarity.yaml > log/inductive/flexmatch/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/flexmatch/text/amazon_reviews_multi_en.yaml > log/inductive/flexmatch/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/flexmatch/text/dbpedia_14.yaml > log/inductive/flexmatch/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/flexmatch/text/imdb.yaml > log/inductive/flexmatch/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/flexmatch/text/yelp_polarity.yaml > log/inductive/flexmatch/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/flexmatch/text/yelp_review_full.yaml > log/inductive/flexmatch/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/flexmatch/vision/cifar10.yaml > log/inductive/flexmatch/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/flexmatch/vision/cifar100.yaml > log/inductive/flexmatch/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/flexmatch/vision/mnist.yaml > log/inductive/flexmatch/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/flexmatch/vision/stl10.yaml > log/inductive/flexmatch/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/flexmatch/vision/svhn.yaml > log/inductive/flexmatch/vision/svhn.log 2>&1
```

## inductive/free_match

```
mkdir -p log/inductive/free_match/audio log/inductive/free_match/graph log/inductive/free_match/tabular log/inductive/free_match/text log/inductive/free_match/vision
python -m bench.main --config bench/configs/experiments/best/inductive/free_match/audio/speechcommands.yaml > log/inductive/free_match/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/free_match/audio/yesno.yaml > log/inductive/free_match/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/free_match/graph/citeseer.yaml > log/inductive/free_match/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/free_match/graph/cora.yaml > log/inductive/free_match/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/free_match/graph/pubmed.yaml > log/inductive/free_match/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/free_match/tabular/adult.yaml > log/inductive/free_match/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/free_match/tabular/breast_cancer.yaml > log/inductive/free_match/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/free_match/tabular/iris.yaml > log/inductive/free_match/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/free_match/tabular/toy.yaml > log/inductive/free_match/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/free_match/text/ag_news.yaml > log/inductive/free_match/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/free_match/text/amazon_polarity.yaml > log/inductive/free_match/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/free_match/text/amazon_reviews_multi_en.yaml > log/inductive/free_match/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/free_match/text/dbpedia_14.yaml > log/inductive/free_match/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/free_match/text/imdb.yaml > log/inductive/free_match/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/free_match/text/yelp_polarity.yaml > log/inductive/free_match/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/free_match/text/yelp_review_full.yaml > log/inductive/free_match/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/free_match/vision/cifar10.yaml > log/inductive/free_match/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/free_match/vision/cifar100.yaml > log/inductive/free_match/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/free_match/vision/mnist.yaml > log/inductive/free_match/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/free_match/vision/stl10.yaml > log/inductive/free_match/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/free_match/vision/svhn.yaml > log/inductive/free_match/vision/svhn.log 2>&1
```

## inductive/mean_teacher

```
mkdir -p log/inductive/mean_teacher/audio log/inductive/mean_teacher/graph log/inductive/mean_teacher/tabular log/inductive/mean_teacher/text log/inductive/mean_teacher/vision
python -m bench.main --config bench/configs/experiments/best/inductive/mean_teacher/audio/speechcommands.yaml > log/inductive/mean_teacher/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mean_teacher/audio/yesno.yaml > log/inductive/mean_teacher/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mean_teacher/graph/citeseer.yaml > log/inductive/mean_teacher/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mean_teacher/graph/cora.yaml > log/inductive/mean_teacher/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mean_teacher/graph/pubmed.yaml > log/inductive/mean_teacher/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mean_teacher/tabular/adult.yaml > log/inductive/mean_teacher/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mean_teacher/tabular/breast_cancer.yaml > log/inductive/mean_teacher/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mean_teacher/tabular/iris.yaml > log/inductive/mean_teacher/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mean_teacher/tabular/toy.yaml > log/inductive/mean_teacher/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mean_teacher/text/ag_news.yaml > log/inductive/mean_teacher/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mean_teacher/text/amazon_polarity.yaml > log/inductive/mean_teacher/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mean_teacher/text/amazon_reviews_multi_en.yaml > log/inductive/mean_teacher/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mean_teacher/text/dbpedia_14.yaml > log/inductive/mean_teacher/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mean_teacher/text/imdb.yaml > log/inductive/mean_teacher/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mean_teacher/text/yelp_polarity.yaml > log/inductive/mean_teacher/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mean_teacher/text/yelp_review_full.yaml > log/inductive/mean_teacher/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mean_teacher/vision/cifar10.yaml > log/inductive/mean_teacher/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mean_teacher/vision/cifar100.yaml > log/inductive/mean_teacher/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mean_teacher/vision/mnist.yaml > log/inductive/mean_teacher/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mean_teacher/vision/stl10.yaml > log/inductive/mean_teacher/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mean_teacher/vision/svhn.yaml > log/inductive/mean_teacher/vision/svhn.log 2>&1
```

## inductive/meta_pseudo_labels

```
mkdir -p log/inductive/meta_pseudo_labels/audio log/inductive/meta_pseudo_labels/graph log/inductive/meta_pseudo_labels/tabular log/inductive/meta_pseudo_labels/text log/inductive/meta_pseudo_labels/vision
python -m bench.main --config bench/configs/experiments/best/inductive/meta_pseudo_labels/audio/speechcommands.yaml > log/inductive/meta_pseudo_labels/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/meta_pseudo_labels/audio/yesno.yaml > log/inductive/meta_pseudo_labels/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/meta_pseudo_labels/graph/citeseer.yaml > log/inductive/meta_pseudo_labels/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/meta_pseudo_labels/graph/cora.yaml > log/inductive/meta_pseudo_labels/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/meta_pseudo_labels/graph/pubmed.yaml > log/inductive/meta_pseudo_labels/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/meta_pseudo_labels/tabular/adult.yaml > log/inductive/meta_pseudo_labels/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/meta_pseudo_labels/tabular/breast_cancer.yaml > log/inductive/meta_pseudo_labels/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/meta_pseudo_labels/tabular/iris.yaml > log/inductive/meta_pseudo_labels/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/meta_pseudo_labels/tabular/toy.yaml > log/inductive/meta_pseudo_labels/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/meta_pseudo_labels/text/ag_news.yaml > log/inductive/meta_pseudo_labels/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/meta_pseudo_labels/text/amazon_polarity.yaml > log/inductive/meta_pseudo_labels/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/meta_pseudo_labels/text/amazon_reviews_multi_en.yaml > log/inductive/meta_pseudo_labels/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/meta_pseudo_labels/text/dbpedia_14.yaml > log/inductive/meta_pseudo_labels/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/meta_pseudo_labels/text/imdb.yaml > log/inductive/meta_pseudo_labels/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/meta_pseudo_labels/text/yelp_polarity.yaml > log/inductive/meta_pseudo_labels/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/meta_pseudo_labels/text/yelp_review_full.yaml > log/inductive/meta_pseudo_labels/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/meta_pseudo_labels/vision/cifar10.yaml > log/inductive/meta_pseudo_labels/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/meta_pseudo_labels/vision/cifar100.yaml > log/inductive/meta_pseudo_labels/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/meta_pseudo_labels/vision/mnist.yaml > log/inductive/meta_pseudo_labels/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/meta_pseudo_labels/vision/stl10.yaml > log/inductive/meta_pseudo_labels/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/meta_pseudo_labels/vision/svhn.yaml > log/inductive/meta_pseudo_labels/vision/svhn.log 2>&1
```

## inductive/mixmatch

```
mkdir -p log/inductive/mixmatch/audio log/inductive/mixmatch/graph log/inductive/mixmatch/tabular log/inductive/mixmatch/text log/inductive/mixmatch/vision
python -m bench.main --config bench/configs/experiments/best/inductive/mixmatch/audio/speechcommands.yaml > log/inductive/mixmatch/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mixmatch/audio/yesno.yaml > log/inductive/mixmatch/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mixmatch/graph/citeseer.yaml > log/inductive/mixmatch/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mixmatch/graph/cora.yaml > log/inductive/mixmatch/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mixmatch/graph/pubmed.yaml > log/inductive/mixmatch/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mixmatch/tabular/adult.yaml > log/inductive/mixmatch/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mixmatch/tabular/breast_cancer.yaml > log/inductive/mixmatch/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mixmatch/tabular/iris.yaml > log/inductive/mixmatch/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mixmatch/tabular/toy.yaml > log/inductive/mixmatch/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mixmatch/text/ag_news.yaml > log/inductive/mixmatch/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mixmatch/text/amazon_polarity.yaml > log/inductive/mixmatch/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mixmatch/text/amazon_reviews_multi_en.yaml > log/inductive/mixmatch/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mixmatch/text/dbpedia_14.yaml > log/inductive/mixmatch/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mixmatch/text/imdb.yaml > log/inductive/mixmatch/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mixmatch/text/yelp_polarity.yaml > log/inductive/mixmatch/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mixmatch/text/yelp_review_full.yaml > log/inductive/mixmatch/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mixmatch/vision/cifar10.yaml > log/inductive/mixmatch/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mixmatch/vision/cifar100.yaml > log/inductive/mixmatch/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mixmatch/vision/mnist.yaml > log/inductive/mixmatch/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mixmatch/vision/stl10.yaml > log/inductive/mixmatch/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/mixmatch/vision/svhn.yaml > log/inductive/mixmatch/vision/svhn.log 2>&1
```

## inductive/noisy_student

```
mkdir -p log/inductive/noisy_student/audio log/inductive/noisy_student/graph log/inductive/noisy_student/tabular log/inductive/noisy_student/text log/inductive/noisy_student/vision
python -m bench.main --config bench/configs/experiments/best/inductive/noisy_student/audio/speechcommands.yaml > log/inductive/noisy_student/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/noisy_student/audio/yesno.yaml > log/inductive/noisy_student/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/noisy_student/graph/citeseer.yaml > log/inductive/noisy_student/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/noisy_student/graph/cora.yaml > log/inductive/noisy_student/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/noisy_student/graph/pubmed.yaml > log/inductive/noisy_student/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/noisy_student/tabular/adult.yaml > log/inductive/noisy_student/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/noisy_student/tabular/breast_cancer.yaml > log/inductive/noisy_student/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/noisy_student/tabular/iris.yaml > log/inductive/noisy_student/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/noisy_student/tabular/toy.yaml > log/inductive/noisy_student/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/noisy_student/text/ag_news.yaml > log/inductive/noisy_student/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/noisy_student/text/amazon_polarity.yaml > log/inductive/noisy_student/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/noisy_student/text/amazon_reviews_multi_en.yaml > log/inductive/noisy_student/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/noisy_student/text/dbpedia_14.yaml > log/inductive/noisy_student/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/noisy_student/text/imdb.yaml > log/inductive/noisy_student/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/noisy_student/text/yelp_polarity.yaml > log/inductive/noisy_student/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/noisy_student/text/yelp_review_full.yaml > log/inductive/noisy_student/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/noisy_student/vision/cifar10.yaml > log/inductive/noisy_student/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/noisy_student/vision/cifar100.yaml > log/inductive/noisy_student/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/noisy_student/vision/mnist.yaml > log/inductive/noisy_student/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/noisy_student/vision/stl10.yaml > log/inductive/noisy_student/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/noisy_student/vision/svhn.yaml > log/inductive/noisy_student/vision/svhn.log 2>&1
```

## inductive/pi_model

```
mkdir -p log/inductive/pi_model/audio log/inductive/pi_model/graph log/inductive/pi_model/tabular log/inductive/pi_model/text log/inductive/pi_model/vision
python -m bench.main --config bench/configs/experiments/best/inductive/pi_model/audio/speechcommands.yaml > log/inductive/pi_model/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pi_model/audio/yesno.yaml > log/inductive/pi_model/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pi_model/graph/citeseer.yaml > log/inductive/pi_model/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pi_model/graph/cora.yaml > log/inductive/pi_model/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pi_model/graph/pubmed.yaml > log/inductive/pi_model/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pi_model/tabular/adult.yaml > log/inductive/pi_model/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pi_model/tabular/breast_cancer.yaml > log/inductive/pi_model/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pi_model/tabular/iris.yaml > log/inductive/pi_model/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pi_model/tabular/toy.yaml > log/inductive/pi_model/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pi_model/text/ag_news.yaml > log/inductive/pi_model/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pi_model/text/amazon_polarity.yaml > log/inductive/pi_model/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pi_model/text/amazon_reviews_multi_en.yaml > log/inductive/pi_model/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pi_model/text/dbpedia_14.yaml > log/inductive/pi_model/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pi_model/text/imdb.yaml > log/inductive/pi_model/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pi_model/text/yelp_polarity.yaml > log/inductive/pi_model/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pi_model/text/yelp_review_full.yaml > log/inductive/pi_model/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pi_model/vision/cifar10.yaml > log/inductive/pi_model/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pi_model/vision/cifar100.yaml > log/inductive/pi_model/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pi_model/vision/mnist.yaml > log/inductive/pi_model/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pi_model/vision/stl10.yaml > log/inductive/pi_model/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pi_model/vision/svhn.yaml > log/inductive/pi_model/vision/svhn.log 2>&1
```

## inductive/pseudo_label

```
mkdir -p log/inductive/pseudo_label/audio log/inductive/pseudo_label/graph log/inductive/pseudo_label/tabular log/inductive/pseudo_label/text log/inductive/pseudo_label/vision
python -m bench.main --config bench/configs/experiments/best/inductive/pseudo_label/audio/speechcommands.yaml > log/inductive/pseudo_label/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pseudo_label/audio/yesno.yaml > log/inductive/pseudo_label/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pseudo_label/graph/citeseer.yaml > log/inductive/pseudo_label/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pseudo_label/graph/cora.yaml > log/inductive/pseudo_label/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pseudo_label/graph/pubmed.yaml > log/inductive/pseudo_label/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pseudo_label/tabular/adult.yaml > log/inductive/pseudo_label/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pseudo_label/tabular/breast_cancer.yaml > log/inductive/pseudo_label/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pseudo_label/tabular/iris.yaml > log/inductive/pseudo_label/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pseudo_label/tabular/toy.yaml > log/inductive/pseudo_label/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pseudo_label/text/ag_news.yaml > log/inductive/pseudo_label/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pseudo_label/text/amazon_polarity.yaml > log/inductive/pseudo_label/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pseudo_label/text/amazon_reviews_multi_en.yaml > log/inductive/pseudo_label/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pseudo_label/text/dbpedia_14.yaml > log/inductive/pseudo_label/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pseudo_label/text/imdb.yaml > log/inductive/pseudo_label/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pseudo_label/text/yelp_polarity.yaml > log/inductive/pseudo_label/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pseudo_label/text/yelp_review_full.yaml > log/inductive/pseudo_label/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pseudo_label/vision/cifar10.yaml > log/inductive/pseudo_label/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pseudo_label/vision/cifar100.yaml > log/inductive/pseudo_label/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pseudo_label/vision/mnist.yaml > log/inductive/pseudo_label/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pseudo_label/vision/stl10.yaml > log/inductive/pseudo_label/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/pseudo_label/vision/svhn.yaml > log/inductive/pseudo_label/vision/svhn.log 2>&1
```

## inductive/s4vm

```
mkdir -p log/inductive/s4vm/audio log/inductive/s4vm/graph log/inductive/s4vm/tabular log/inductive/s4vm/text log/inductive/s4vm/vision
python -m bench.main --config bench/configs/experiments/best/inductive/s4vm/audio/speechcommands.yaml > log/inductive/s4vm/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/s4vm/audio/yesno.yaml > log/inductive/s4vm/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/s4vm/graph/citeseer.yaml > log/inductive/s4vm/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/s4vm/graph/cora.yaml > log/inductive/s4vm/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/s4vm/graph/pubmed.yaml > log/inductive/s4vm/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/s4vm/tabular/adult.yaml > log/inductive/s4vm/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/s4vm/tabular/breast_cancer.yaml > log/inductive/s4vm/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/s4vm/tabular/iris.yaml > log/inductive/s4vm/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/s4vm/tabular/toy.yaml > log/inductive/s4vm/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/s4vm/text/ag_news.yaml > log/inductive/s4vm/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/s4vm/text/amazon_polarity.yaml > log/inductive/s4vm/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/s4vm/text/amazon_reviews_multi_en.yaml > log/inductive/s4vm/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/s4vm/text/dbpedia_14.yaml > log/inductive/s4vm/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/s4vm/text/imdb.yaml > log/inductive/s4vm/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/s4vm/text/yelp_polarity.yaml > log/inductive/s4vm/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/s4vm/text/yelp_review_full.yaml > log/inductive/s4vm/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/s4vm/vision/cifar10.yaml > log/inductive/s4vm/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/s4vm/vision/cifar100.yaml > log/inductive/s4vm/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/s4vm/vision/mnist.yaml > log/inductive/s4vm/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/s4vm/vision/stl10.yaml > log/inductive/s4vm/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/s4vm/vision/svhn.yaml > log/inductive/s4vm/vision/svhn.log 2>&1
```

## inductive/self_training

```
mkdir -p log/inductive/self_training/audio log/inductive/self_training/graph log/inductive/self_training/tabular log/inductive/self_training/text log/inductive/self_training/vision
python -m bench.main --config bench/configs/experiments/best/inductive/self_training/audio/speechcommands.yaml > log/inductive/self_training/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/self_training/audio/yesno.yaml > log/inductive/self_training/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/self_training/graph/citeseer.yaml > log/inductive/self_training/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/self_training/graph/cora.yaml > log/inductive/self_training/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/self_training/graph/pubmed.yaml > log/inductive/self_training/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/self_training/tabular/adult.yaml > log/inductive/self_training/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/self_training/tabular/breast_cancer.yaml > log/inductive/self_training/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/self_training/tabular/iris.yaml > log/inductive/self_training/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/self_training/tabular/toy.yaml > log/inductive/self_training/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/self_training/text/ag_news.yaml > log/inductive/self_training/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/self_training/text/amazon_polarity.yaml > log/inductive/self_training/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/self_training/text/amazon_reviews_multi_en.yaml > log/inductive/self_training/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/self_training/text/dbpedia_14.yaml > log/inductive/self_training/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/self_training/text/imdb.yaml > log/inductive/self_training/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/self_training/text/yelp_polarity.yaml > log/inductive/self_training/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/self_training/text/yelp_review_full.yaml > log/inductive/self_training/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/self_training/vision/cifar10.yaml > log/inductive/self_training/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/self_training/vision/cifar100.yaml > log/inductive/self_training/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/self_training/vision/mnist.yaml > log/inductive/self_training/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/self_training/vision/stl10.yaml > log/inductive/self_training/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/self_training/vision/svhn.yaml > log/inductive/self_training/vision/svhn.log 2>&1
```

## inductive/setred

```
mkdir -p log/inductive/setred/audio log/inductive/setred/graph log/inductive/setred/tabular log/inductive/setred/text log/inductive/setred/vision
python -m bench.main --config bench/configs/experiments/best/inductive/setred/audio/speechcommands.yaml > log/inductive/setred/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/setred/audio/yesno.yaml > log/inductive/setred/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/setred/graph/citeseer.yaml > log/inductive/setred/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/setred/graph/cora.yaml > log/inductive/setred/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/setred/graph/pubmed.yaml > log/inductive/setred/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/setred/tabular/adult.yaml > log/inductive/setred/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/setred/tabular/breast_cancer.yaml > log/inductive/setred/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/setred/tabular/iris.yaml > log/inductive/setred/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/setred/tabular/toy.yaml > log/inductive/setred/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/setred/text/ag_news.yaml > log/inductive/setred/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/setred/text/amazon_polarity.yaml > log/inductive/setred/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/setred/text/amazon_reviews_multi_en.yaml > log/inductive/setred/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/setred/text/dbpedia_14.yaml > log/inductive/setred/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/setred/text/imdb.yaml > log/inductive/setred/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/setred/text/yelp_polarity.yaml > log/inductive/setred/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/setred/text/yelp_review_full.yaml > log/inductive/setred/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/setred/vision/cifar10.yaml > log/inductive/setred/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/setred/vision/cifar100.yaml > log/inductive/setred/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/setred/vision/mnist.yaml > log/inductive/setred/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/setred/vision/stl10.yaml > log/inductive/setred/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/setred/vision/svhn.yaml > log/inductive/setred/vision/svhn.log 2>&1
```

## inductive/simclr_v2

```
mkdir -p log/inductive/simclr_v2/audio log/inductive/simclr_v2/graph log/inductive/simclr_v2/tabular log/inductive/simclr_v2/text log/inductive/simclr_v2/vision
python -m bench.main --config bench/configs/experiments/best/inductive/simclr_v2/audio/speechcommands.yaml > log/inductive/simclr_v2/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/simclr_v2/audio/yesno.yaml > log/inductive/simclr_v2/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/simclr_v2/graph/citeseer.yaml > log/inductive/simclr_v2/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/simclr_v2/graph/cora.yaml > log/inductive/simclr_v2/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/simclr_v2/graph/pubmed.yaml > log/inductive/simclr_v2/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/simclr_v2/tabular/adult.yaml > log/inductive/simclr_v2/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/simclr_v2/tabular/breast_cancer.yaml > log/inductive/simclr_v2/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/simclr_v2/tabular/iris.yaml > log/inductive/simclr_v2/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/simclr_v2/tabular/toy.yaml > log/inductive/simclr_v2/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/simclr_v2/text/ag_news.yaml > log/inductive/simclr_v2/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/simclr_v2/text/amazon_polarity.yaml > log/inductive/simclr_v2/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/simclr_v2/text/amazon_reviews_multi_en.yaml > log/inductive/simclr_v2/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/simclr_v2/text/dbpedia_14.yaml > log/inductive/simclr_v2/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/simclr_v2/text/imdb.yaml > log/inductive/simclr_v2/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/simclr_v2/text/yelp_polarity.yaml > log/inductive/simclr_v2/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/simclr_v2/text/yelp_review_full.yaml > log/inductive/simclr_v2/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/simclr_v2/vision/cifar10.yaml > log/inductive/simclr_v2/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/simclr_v2/vision/cifar100.yaml > log/inductive/simclr_v2/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/simclr_v2/vision/mnist.yaml > log/inductive/simclr_v2/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/simclr_v2/vision/stl10.yaml > log/inductive/simclr_v2/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/simclr_v2/vision/svhn.yaml > log/inductive/simclr_v2/vision/svhn.log 2>&1
```

## inductive/softmatch

```
mkdir -p log/inductive/softmatch/audio log/inductive/softmatch/graph log/inductive/softmatch/tabular log/inductive/softmatch/text log/inductive/softmatch/vision
python -m bench.main --config bench/configs/experiments/best/inductive/softmatch/audio/speechcommands.yaml > log/inductive/softmatch/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/softmatch/audio/yesno.yaml > log/inductive/softmatch/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/softmatch/graph/citeseer.yaml > log/inductive/softmatch/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/softmatch/graph/cora.yaml > log/inductive/softmatch/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/softmatch/graph/pubmed.yaml > log/inductive/softmatch/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/softmatch/tabular/adult.yaml > log/inductive/softmatch/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/softmatch/tabular/breast_cancer.yaml > log/inductive/softmatch/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/softmatch/tabular/iris.yaml > log/inductive/softmatch/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/softmatch/tabular/toy.yaml > log/inductive/softmatch/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/softmatch/text/ag_news.yaml > log/inductive/softmatch/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/softmatch/text/amazon_polarity.yaml > log/inductive/softmatch/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/softmatch/text/amazon_reviews_multi_en.yaml > log/inductive/softmatch/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/softmatch/text/dbpedia_14.yaml > log/inductive/softmatch/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/softmatch/text/imdb.yaml > log/inductive/softmatch/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/softmatch/text/yelp_polarity.yaml > log/inductive/softmatch/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/softmatch/text/yelp_review_full.yaml > log/inductive/softmatch/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/softmatch/vision/cifar10.yaml > log/inductive/softmatch/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/softmatch/vision/cifar100.yaml > log/inductive/softmatch/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/softmatch/vision/mnist.yaml > log/inductive/softmatch/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/softmatch/vision/stl10.yaml > log/inductive/softmatch/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/softmatch/vision/svhn.yaml > log/inductive/softmatch/vision/svhn.log 2>&1
```

## inductive/temporal_ensembling

```
mkdir -p log/inductive/temporal_ensembling/audio log/inductive/temporal_ensembling/graph log/inductive/temporal_ensembling/tabular log/inductive/temporal_ensembling/text log/inductive/temporal_ensembling/vision
python -m bench.main --config bench/configs/experiments/best/inductive/temporal_ensembling/audio/speechcommands.yaml > log/inductive/temporal_ensembling/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/temporal_ensembling/audio/yesno.yaml > log/inductive/temporal_ensembling/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/temporal_ensembling/graph/citeseer.yaml > log/inductive/temporal_ensembling/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/temporal_ensembling/graph/cora.yaml > log/inductive/temporal_ensembling/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/temporal_ensembling/graph/pubmed.yaml > log/inductive/temporal_ensembling/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/temporal_ensembling/tabular/adult.yaml > log/inductive/temporal_ensembling/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/temporal_ensembling/tabular/breast_cancer.yaml > log/inductive/temporal_ensembling/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/temporal_ensembling/tabular/iris.yaml > log/inductive/temporal_ensembling/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/temporal_ensembling/tabular/toy.yaml > log/inductive/temporal_ensembling/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/temporal_ensembling/text/ag_news.yaml > log/inductive/temporal_ensembling/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/temporal_ensembling/text/amazon_polarity.yaml > log/inductive/temporal_ensembling/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/temporal_ensembling/text/amazon_reviews_multi_en.yaml > log/inductive/temporal_ensembling/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/temporal_ensembling/text/dbpedia_14.yaml > log/inductive/temporal_ensembling/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/temporal_ensembling/text/imdb.yaml > log/inductive/temporal_ensembling/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/temporal_ensembling/text/yelp_polarity.yaml > log/inductive/temporal_ensembling/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/temporal_ensembling/text/yelp_review_full.yaml > log/inductive/temporal_ensembling/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/temporal_ensembling/vision/cifar10.yaml > log/inductive/temporal_ensembling/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/temporal_ensembling/vision/cifar100.yaml > log/inductive/temporal_ensembling/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/temporal_ensembling/vision/mnist.yaml > log/inductive/temporal_ensembling/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/temporal_ensembling/vision/stl10.yaml > log/inductive/temporal_ensembling/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/temporal_ensembling/vision/svhn.yaml > log/inductive/temporal_ensembling/vision/svhn.log 2>&1
```

## inductive/tri_training

```
mkdir -p log/inductive/tri_training/audio log/inductive/tri_training/graph log/inductive/tri_training/tabular log/inductive/tri_training/text log/inductive/tri_training/vision
python -m bench.main --config bench/configs/experiments/best/inductive/tri_training/audio/speechcommands.yaml > log/inductive/tri_training/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tri_training/audio/yesno.yaml > log/inductive/tri_training/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tri_training/graph/citeseer.yaml > log/inductive/tri_training/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tri_training/graph/cora.yaml > log/inductive/tri_training/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tri_training/graph/pubmed.yaml > log/inductive/tri_training/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tri_training/tabular/adult.yaml > log/inductive/tri_training/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tri_training/tabular/breast_cancer.yaml > log/inductive/tri_training/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tri_training/tabular/iris.yaml > log/inductive/tri_training/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tri_training/tabular/toy.yaml > log/inductive/tri_training/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tri_training/text/ag_news.yaml > log/inductive/tri_training/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tri_training/text/amazon_polarity.yaml > log/inductive/tri_training/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tri_training/text/amazon_reviews_multi_en.yaml > log/inductive/tri_training/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tri_training/text/dbpedia_14.yaml > log/inductive/tri_training/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tri_training/text/imdb.yaml > log/inductive/tri_training/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tri_training/text/yelp_polarity.yaml > log/inductive/tri_training/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tri_training/text/yelp_review_full.yaml > log/inductive/tri_training/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tri_training/vision/cifar10.yaml > log/inductive/tri_training/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tri_training/vision/cifar100.yaml > log/inductive/tri_training/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tri_training/vision/mnist.yaml > log/inductive/tri_training/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tri_training/vision/stl10.yaml > log/inductive/tri_training/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tri_training/vision/svhn.yaml > log/inductive/tri_training/vision/svhn.log 2>&1
```

## inductive/trinet

```
mkdir -p log/inductive/trinet/audio log/inductive/trinet/graph log/inductive/trinet/tabular log/inductive/trinet/text log/inductive/trinet/vision
python -m bench.main --config bench/configs/experiments/best/inductive/trinet/audio/speechcommands.yaml > log/inductive/trinet/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/trinet/audio/yesno.yaml > log/inductive/trinet/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/trinet/graph/citeseer.yaml > log/inductive/trinet/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/trinet/graph/cora.yaml > log/inductive/trinet/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/trinet/graph/pubmed.yaml > log/inductive/trinet/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/trinet/tabular/adult.yaml > log/inductive/trinet/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/trinet/tabular/breast_cancer.yaml > log/inductive/trinet/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/trinet/tabular/iris.yaml > log/inductive/trinet/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/trinet/tabular/toy.yaml > log/inductive/trinet/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/trinet/text/ag_news.yaml > log/inductive/trinet/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/trinet/text/amazon_polarity.yaml > log/inductive/trinet/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/trinet/text/amazon_reviews_multi_en.yaml > log/inductive/trinet/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/trinet/text/dbpedia_14.yaml > log/inductive/trinet/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/trinet/text/imdb.yaml > log/inductive/trinet/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/trinet/text/yelp_polarity.yaml > log/inductive/trinet/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/trinet/text/yelp_review_full.yaml > log/inductive/trinet/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/trinet/vision/cifar10.yaml > log/inductive/trinet/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/trinet/vision/cifar100.yaml > log/inductive/trinet/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/trinet/vision/mnist.yaml > log/inductive/trinet/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/trinet/vision/stl10.yaml > log/inductive/trinet/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/trinet/vision/svhn.yaml > log/inductive/trinet/vision/svhn.log 2>&1
```

## inductive/tsvm

```
mkdir -p log/inductive/tsvm/audio log/inductive/tsvm/graph log/inductive/tsvm/tabular log/inductive/tsvm/text log/inductive/tsvm/vision
python -m bench.main --config bench/configs/experiments/best/inductive/tsvm/audio/speechcommands.yaml > log/inductive/tsvm/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tsvm/audio/yesno.yaml > log/inductive/tsvm/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tsvm/graph/citeseer.yaml > log/inductive/tsvm/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tsvm/graph/cora.yaml > log/inductive/tsvm/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tsvm/graph/pubmed.yaml > log/inductive/tsvm/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tsvm/tabular/adult.yaml > log/inductive/tsvm/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tsvm/tabular/breast_cancer.yaml > log/inductive/tsvm/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tsvm/tabular/iris.yaml > log/inductive/tsvm/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tsvm/tabular/toy.yaml > log/inductive/tsvm/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tsvm/text/ag_news.yaml > log/inductive/tsvm/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tsvm/text/amazon_polarity.yaml > log/inductive/tsvm/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tsvm/text/amazon_reviews_multi_en.yaml > log/inductive/tsvm/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tsvm/text/dbpedia_14.yaml > log/inductive/tsvm/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tsvm/text/imdb.yaml > log/inductive/tsvm/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tsvm/text/yelp_polarity.yaml > log/inductive/tsvm/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tsvm/text/yelp_review_full.yaml > log/inductive/tsvm/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tsvm/vision/cifar10.yaml > log/inductive/tsvm/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tsvm/vision/cifar100.yaml > log/inductive/tsvm/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tsvm/vision/mnist.yaml > log/inductive/tsvm/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tsvm/vision/stl10.yaml > log/inductive/tsvm/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/tsvm/vision/svhn.yaml > log/inductive/tsvm/vision/svhn.log 2>&1
```

## inductive/uda

```
mkdir -p log/inductive/uda/audio log/inductive/uda/graph log/inductive/uda/tabular log/inductive/uda/text log/inductive/uda/vision
python -m bench.main --config bench/configs/experiments/best/inductive/uda/audio/speechcommands.yaml > log/inductive/uda/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/uda/audio/yesno.yaml > log/inductive/uda/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/uda/graph/citeseer.yaml > log/inductive/uda/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/uda/graph/cora.yaml > log/inductive/uda/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/uda/graph/pubmed.yaml > log/inductive/uda/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/uda/tabular/adult.yaml > log/inductive/uda/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/uda/tabular/breast_cancer.yaml > log/inductive/uda/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/uda/tabular/iris.yaml > log/inductive/uda/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/uda/tabular/toy.yaml > log/inductive/uda/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/uda/text/ag_news.yaml > log/inductive/uda/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/uda/text/amazon_polarity.yaml > log/inductive/uda/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/uda/text/amazon_reviews_multi_en.yaml > log/inductive/uda/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/uda/text/dbpedia_14.yaml > log/inductive/uda/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/uda/text/imdb.yaml > log/inductive/uda/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/uda/text/yelp_polarity.yaml > log/inductive/uda/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/uda/text/yelp_review_full.yaml > log/inductive/uda/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/uda/vision/cifar10.yaml > log/inductive/uda/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/uda/vision/cifar100.yaml > log/inductive/uda/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/uda/vision/mnist.yaml > log/inductive/uda/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/uda/vision/stl10.yaml > log/inductive/uda/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/uda/vision/svhn.yaml > log/inductive/uda/vision/svhn.log 2>&1
```

## inductive/vat

```
mkdir -p log/inductive/vat/audio log/inductive/vat/graph log/inductive/vat/tabular log/inductive/vat/text log/inductive/vat/vision
python -m bench.main --config bench/configs/experiments/best/inductive/vat/audio/speechcommands.yaml > log/inductive/vat/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/vat/audio/yesno.yaml > log/inductive/vat/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/vat/graph/citeseer.yaml > log/inductive/vat/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/vat/graph/cora.yaml > log/inductive/vat/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/vat/graph/pubmed.yaml > log/inductive/vat/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/vat/tabular/adult.yaml > log/inductive/vat/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/vat/tabular/breast_cancer.yaml > log/inductive/vat/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/vat/tabular/iris.yaml > log/inductive/vat/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/vat/tabular/toy.yaml > log/inductive/vat/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/vat/text/ag_news.yaml > log/inductive/vat/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/vat/text/amazon_polarity.yaml > log/inductive/vat/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/vat/text/amazon_reviews_multi_en.yaml > log/inductive/vat/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/vat/text/dbpedia_14.yaml > log/inductive/vat/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/vat/text/imdb.yaml > log/inductive/vat/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/vat/text/yelp_polarity.yaml > log/inductive/vat/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/vat/text/yelp_review_full.yaml > log/inductive/vat/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/vat/vision/cifar10.yaml > log/inductive/vat/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/vat/vision/cifar100.yaml > log/inductive/vat/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/vat/vision/mnist.yaml > log/inductive/vat/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/vat/vision/stl10.yaml > log/inductive/vat/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/inductive/vat/vision/svhn.yaml > log/inductive/vat/vision/svhn.log 2>&1
```

## transductive/appnp

```
mkdir -p log/transductive/appnp/audio log/transductive/appnp/graph log/transductive/appnp/tabular log/transductive/appnp/text log/transductive/appnp/vision
python -m bench.main --config bench/configs/experiments/best/transductive/appnp/audio/speechcommands.yaml > log/transductive/appnp/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/appnp/audio/yesno.yaml > log/transductive/appnp/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/appnp/graph/citeseer.yaml > log/transductive/appnp/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/appnp/graph/cora.yaml > log/transductive/appnp/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/appnp/graph/pubmed.yaml > log/transductive/appnp/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/appnp/tabular/adult.yaml > log/transductive/appnp/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/appnp/tabular/breast_cancer.yaml > log/transductive/appnp/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/appnp/tabular/iris.yaml > log/transductive/appnp/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/appnp/tabular/toy.yaml > log/transductive/appnp/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/appnp/text/ag_news.yaml > log/transductive/appnp/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/appnp/text/amazon_polarity.yaml > log/transductive/appnp/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/appnp/text/amazon_reviews_multi_en.yaml > log/transductive/appnp/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/appnp/text/dbpedia_14.yaml > log/transductive/appnp/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/appnp/text/imdb.yaml > log/transductive/appnp/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/appnp/text/yelp_polarity.yaml > log/transductive/appnp/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/appnp/text/yelp_review_full.yaml > log/transductive/appnp/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/appnp/vision/cifar10.yaml > log/transductive/appnp/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/appnp/vision/cifar100.yaml > log/transductive/appnp/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/appnp/vision/mnist.yaml > log/transductive/appnp/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/appnp/vision/stl10.yaml > log/transductive/appnp/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/appnp/vision/svhn.yaml > log/transductive/appnp/vision/svhn.log 2>&1
```

## transductive/chebnet

```
mkdir -p log/transductive/chebnet/audio log/transductive/chebnet/graph log/transductive/chebnet/tabular log/transductive/chebnet/text log/transductive/chebnet/vision
python -m bench.main --config bench/configs/experiments/best/transductive/chebnet/audio/speechcommands.yaml > log/transductive/chebnet/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/chebnet/audio/yesno.yaml > log/transductive/chebnet/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/chebnet/graph/citeseer.yaml > log/transductive/chebnet/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/chebnet/graph/cora.yaml > log/transductive/chebnet/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/chebnet/graph/pubmed.yaml > log/transductive/chebnet/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/chebnet/tabular/adult.yaml > log/transductive/chebnet/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/chebnet/tabular/breast_cancer.yaml > log/transductive/chebnet/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/chebnet/tabular/iris.yaml > log/transductive/chebnet/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/chebnet/tabular/toy.yaml > log/transductive/chebnet/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/chebnet/text/ag_news.yaml > log/transductive/chebnet/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/chebnet/text/amazon_polarity.yaml > log/transductive/chebnet/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/chebnet/text/amazon_reviews_multi_en.yaml > log/transductive/chebnet/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/chebnet/text/dbpedia_14.yaml > log/transductive/chebnet/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/chebnet/text/imdb.yaml > log/transductive/chebnet/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/chebnet/text/yelp_polarity.yaml > log/transductive/chebnet/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/chebnet/text/yelp_review_full.yaml > log/transductive/chebnet/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/chebnet/vision/cifar10.yaml > log/transductive/chebnet/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/chebnet/vision/cifar100.yaml > log/transductive/chebnet/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/chebnet/vision/mnist.yaml > log/transductive/chebnet/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/chebnet/vision/stl10.yaml > log/transductive/chebnet/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/chebnet/vision/svhn.yaml > log/transductive/chebnet/vision/svhn.log 2>&1
```

## transductive/dgi

```
mkdir -p log/transductive/dgi/audio log/transductive/dgi/graph log/transductive/dgi/tabular log/transductive/dgi/text log/transductive/dgi/vision
python -m bench.main --config bench/configs/experiments/best/transductive/dgi/audio/speechcommands.yaml > log/transductive/dgi/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dgi/audio/yesno.yaml > log/transductive/dgi/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dgi/graph/citeseer.yaml > log/transductive/dgi/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dgi/graph/cora.yaml > log/transductive/dgi/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dgi/graph/pubmed.yaml > log/transductive/dgi/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dgi/tabular/adult.yaml > log/transductive/dgi/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dgi/tabular/breast_cancer.yaml > log/transductive/dgi/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dgi/tabular/iris.yaml > log/transductive/dgi/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dgi/tabular/toy.yaml > log/transductive/dgi/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dgi/text/ag_news.yaml > log/transductive/dgi/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dgi/text/amazon_polarity.yaml > log/transductive/dgi/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dgi/text/amazon_reviews_multi_en.yaml > log/transductive/dgi/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dgi/text/dbpedia_14.yaml > log/transductive/dgi/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dgi/text/imdb.yaml > log/transductive/dgi/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dgi/text/yelp_polarity.yaml > log/transductive/dgi/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dgi/text/yelp_review_full.yaml > log/transductive/dgi/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dgi/vision/cifar10.yaml > log/transductive/dgi/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dgi/vision/cifar100.yaml > log/transductive/dgi/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dgi/vision/mnist.yaml > log/transductive/dgi/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dgi/vision/stl10.yaml > log/transductive/dgi/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dgi/vision/svhn.yaml > log/transductive/dgi/vision/svhn.log 2>&1
```

## transductive/dynamic_label_propagation

```
mkdir -p log/transductive/dynamic_label_propagation/audio log/transductive/dynamic_label_propagation/graph log/transductive/dynamic_label_propagation/tabular log/transductive/dynamic_label_propagation/text log/transductive/dynamic_label_propagation/vision
python -m bench.main --config bench/configs/experiments/best/transductive/dynamic_label_propagation/audio/speechcommands.yaml > log/transductive/dynamic_label_propagation/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dynamic_label_propagation/audio/yesno.yaml > log/transductive/dynamic_label_propagation/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dynamic_label_propagation/graph/citeseer.yaml > log/transductive/dynamic_label_propagation/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dynamic_label_propagation/graph/cora.yaml > log/transductive/dynamic_label_propagation/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dynamic_label_propagation/graph/pubmed.yaml > log/transductive/dynamic_label_propagation/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dynamic_label_propagation/tabular/adult.yaml > log/transductive/dynamic_label_propagation/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dynamic_label_propagation/tabular/breast_cancer.yaml > log/transductive/dynamic_label_propagation/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dynamic_label_propagation/tabular/iris.yaml > log/transductive/dynamic_label_propagation/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dynamic_label_propagation/tabular/toy.yaml > log/transductive/dynamic_label_propagation/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dynamic_label_propagation/text/ag_news.yaml > log/transductive/dynamic_label_propagation/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dynamic_label_propagation/text/amazon_polarity.yaml > log/transductive/dynamic_label_propagation/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dynamic_label_propagation/text/amazon_reviews_multi_en.yaml > log/transductive/dynamic_label_propagation/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dynamic_label_propagation/text/dbpedia_14.yaml > log/transductive/dynamic_label_propagation/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dynamic_label_propagation/text/imdb.yaml > log/transductive/dynamic_label_propagation/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dynamic_label_propagation/text/yelp_polarity.yaml > log/transductive/dynamic_label_propagation/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dynamic_label_propagation/text/yelp_review_full.yaml > log/transductive/dynamic_label_propagation/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dynamic_label_propagation/vision/cifar10.yaml > log/transductive/dynamic_label_propagation/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dynamic_label_propagation/vision/cifar100.yaml > log/transductive/dynamic_label_propagation/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dynamic_label_propagation/vision/mnist.yaml > log/transductive/dynamic_label_propagation/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dynamic_label_propagation/vision/stl10.yaml > log/transductive/dynamic_label_propagation/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/dynamic_label_propagation/vision/svhn.yaml > log/transductive/dynamic_label_propagation/vision/svhn.log 2>&1
```

## transductive/gat

```
mkdir -p log/transductive/gat/audio log/transductive/gat/graph log/transductive/gat/tabular log/transductive/gat/text log/transductive/gat/vision
python -m bench.main --config bench/configs/experiments/best/transductive/gat/audio/speechcommands.yaml > log/transductive/gat/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gat/audio/yesno.yaml > log/transductive/gat/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gat/graph/citeseer.yaml > log/transductive/gat/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gat/graph/cora.yaml > log/transductive/gat/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gat/graph/pubmed.yaml > log/transductive/gat/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gat/tabular/adult.yaml > log/transductive/gat/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gat/tabular/breast_cancer.yaml > log/transductive/gat/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gat/tabular/iris.yaml > log/transductive/gat/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gat/tabular/toy.yaml > log/transductive/gat/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gat/text/ag_news.yaml > log/transductive/gat/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gat/text/amazon_polarity.yaml > log/transductive/gat/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gat/text/amazon_reviews_multi_en.yaml > log/transductive/gat/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gat/text/dbpedia_14.yaml > log/transductive/gat/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gat/text/imdb.yaml > log/transductive/gat/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gat/text/yelp_polarity.yaml > log/transductive/gat/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gat/text/yelp_review_full.yaml > log/transductive/gat/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gat/vision/cifar10.yaml > log/transductive/gat/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gat/vision/cifar100.yaml > log/transductive/gat/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gat/vision/mnist.yaml > log/transductive/gat/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gat/vision/stl10.yaml > log/transductive/gat/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gat/vision/svhn.yaml > log/transductive/gat/vision/svhn.log 2>&1
```

## transductive/gcn

```
mkdir -p log/transductive/gcn/audio log/transductive/gcn/graph log/transductive/gcn/tabular log/transductive/gcn/text log/transductive/gcn/vision
python -m bench.main --config bench/configs/experiments/best/transductive/gcn/audio/speechcommands.yaml > log/transductive/gcn/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcn/audio/yesno.yaml > log/transductive/gcn/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcn/graph/citeseer.yaml > log/transductive/gcn/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcn/graph/cora.yaml > log/transductive/gcn/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcn/graph/pubmed.yaml > log/transductive/gcn/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcn/tabular/adult.yaml > log/transductive/gcn/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcn/tabular/breast_cancer.yaml > log/transductive/gcn/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcn/tabular/iris.yaml > log/transductive/gcn/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcn/tabular/toy.yaml > log/transductive/gcn/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcn/text/ag_news.yaml > log/transductive/gcn/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcn/text/amazon_polarity.yaml > log/transductive/gcn/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcn/text/amazon_reviews_multi_en.yaml > log/transductive/gcn/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcn/text/dbpedia_14.yaml > log/transductive/gcn/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcn/text/imdb.yaml > log/transductive/gcn/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcn/text/yelp_polarity.yaml > log/transductive/gcn/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcn/text/yelp_review_full.yaml > log/transductive/gcn/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcn/vision/cifar10.yaml > log/transductive/gcn/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcn/vision/cifar100.yaml > log/transductive/gcn/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcn/vision/mnist.yaml > log/transductive/gcn/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcn/vision/stl10.yaml > log/transductive/gcn/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcn/vision/svhn.yaml > log/transductive/gcn/vision/svhn.log 2>&1
```

## transductive/gcnii

```
mkdir -p log/transductive/gcnii/audio log/transductive/gcnii/graph log/transductive/gcnii/tabular log/transductive/gcnii/text log/transductive/gcnii/vision
python -m bench.main --config bench/configs/experiments/best/transductive/gcnii/audio/speechcommands.yaml > log/transductive/gcnii/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcnii/audio/yesno.yaml > log/transductive/gcnii/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcnii/graph/citeseer.yaml > log/transductive/gcnii/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcnii/graph/cora.yaml > log/transductive/gcnii/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcnii/graph/pubmed.yaml > log/transductive/gcnii/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcnii/tabular/adult.yaml > log/transductive/gcnii/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcnii/tabular/breast_cancer.yaml > log/transductive/gcnii/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcnii/tabular/iris.yaml > log/transductive/gcnii/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcnii/tabular/toy.yaml > log/transductive/gcnii/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcnii/text/ag_news.yaml > log/transductive/gcnii/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcnii/text/amazon_polarity.yaml > log/transductive/gcnii/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcnii/text/amazon_reviews_multi_en.yaml > log/transductive/gcnii/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcnii/text/dbpedia_14.yaml > log/transductive/gcnii/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcnii/text/imdb.yaml > log/transductive/gcnii/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcnii/text/yelp_polarity.yaml > log/transductive/gcnii/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcnii/text/yelp_review_full.yaml > log/transductive/gcnii/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcnii/vision/cifar10.yaml > log/transductive/gcnii/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcnii/vision/cifar100.yaml > log/transductive/gcnii/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcnii/vision/mnist.yaml > log/transductive/gcnii/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcnii/vision/stl10.yaml > log/transductive/gcnii/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/gcnii/vision/svhn.yaml > log/transductive/gcnii/vision/svhn.log 2>&1
```

## transductive/grafn

```
mkdir -p log/transductive/grafn/audio log/transductive/grafn/graph log/transductive/grafn/tabular log/transductive/grafn/text log/transductive/grafn/vision
python -m bench.main --config bench/configs/experiments/best/transductive/grafn/audio/speechcommands.yaml > log/transductive/grafn/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grafn/audio/yesno.yaml > log/transductive/grafn/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grafn/graph/citeseer.yaml > log/transductive/grafn/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grafn/graph/cora.yaml > log/transductive/grafn/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grafn/graph/pubmed.yaml > log/transductive/grafn/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grafn/tabular/adult.yaml > log/transductive/grafn/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grafn/tabular/breast_cancer.yaml > log/transductive/grafn/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grafn/tabular/iris.yaml > log/transductive/grafn/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grafn/tabular/toy.yaml > log/transductive/grafn/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grafn/text/ag_news.yaml > log/transductive/grafn/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grafn/text/amazon_polarity.yaml > log/transductive/grafn/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grafn/text/amazon_reviews_multi_en.yaml > log/transductive/grafn/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grafn/text/dbpedia_14.yaml > log/transductive/grafn/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grafn/text/imdb.yaml > log/transductive/grafn/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grafn/text/yelp_polarity.yaml > log/transductive/grafn/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grafn/text/yelp_review_full.yaml > log/transductive/grafn/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grafn/vision/cifar10.yaml > log/transductive/grafn/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grafn/vision/cifar100.yaml > log/transductive/grafn/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grafn/vision/mnist.yaml > log/transductive/grafn/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grafn/vision/stl10.yaml > log/transductive/grafn/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grafn/vision/svhn.yaml > log/transductive/grafn/vision/svhn.log 2>&1
```

## transductive/grand

```
mkdir -p log/transductive/grand/audio log/transductive/grand/graph log/transductive/grand/tabular log/transductive/grand/text log/transductive/grand/vision
python -m bench.main --config bench/configs/experiments/best/transductive/grand/audio/speechcommands.yaml > log/transductive/grand/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grand/audio/yesno.yaml > log/transductive/grand/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grand/graph/citeseer.yaml > log/transductive/grand/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grand/graph/cora.yaml > log/transductive/grand/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grand/graph/pubmed.yaml > log/transductive/grand/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grand/tabular/adult.yaml > log/transductive/grand/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grand/tabular/breast_cancer.yaml > log/transductive/grand/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grand/tabular/iris.yaml > log/transductive/grand/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grand/tabular/toy.yaml > log/transductive/grand/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grand/text/ag_news.yaml > log/transductive/grand/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grand/text/amazon_polarity.yaml > log/transductive/grand/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grand/text/amazon_reviews_multi_en.yaml > log/transductive/grand/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grand/text/dbpedia_14.yaml > log/transductive/grand/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grand/text/imdb.yaml > log/transductive/grand/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grand/text/yelp_polarity.yaml > log/transductive/grand/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grand/text/yelp_review_full.yaml > log/transductive/grand/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grand/vision/cifar10.yaml > log/transductive/grand/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grand/vision/cifar100.yaml > log/transductive/grand/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grand/vision/mnist.yaml > log/transductive/grand/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grand/vision/stl10.yaml > log/transductive/grand/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/grand/vision/svhn.yaml > log/transductive/grand/vision/svhn.log 2>&1
```

## transductive/graph_mincuts

```
mkdir -p log/transductive/graph_mincuts/audio log/transductive/graph_mincuts/graph log/transductive/graph_mincuts/tabular log/transductive/graph_mincuts/text log/transductive/graph_mincuts/vision
python -m bench.main --config bench/configs/experiments/best/transductive/graph_mincuts/audio/speechcommands.yaml > log/transductive/graph_mincuts/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graph_mincuts/audio/yesno.yaml > log/transductive/graph_mincuts/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graph_mincuts/graph/citeseer.yaml > log/transductive/graph_mincuts/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graph_mincuts/graph/cora.yaml > log/transductive/graph_mincuts/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graph_mincuts/graph/pubmed.yaml > log/transductive/graph_mincuts/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graph_mincuts/tabular/adult.yaml > log/transductive/graph_mincuts/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graph_mincuts/tabular/breast_cancer.yaml > log/transductive/graph_mincuts/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graph_mincuts/tabular/iris.yaml > log/transductive/graph_mincuts/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graph_mincuts/tabular/toy.yaml > log/transductive/graph_mincuts/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graph_mincuts/text/ag_news.yaml > log/transductive/graph_mincuts/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graph_mincuts/text/amazon_polarity.yaml > log/transductive/graph_mincuts/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graph_mincuts/text/amazon_reviews_multi_en.yaml > log/transductive/graph_mincuts/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graph_mincuts/text/dbpedia_14.yaml > log/transductive/graph_mincuts/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graph_mincuts/text/imdb.yaml > log/transductive/graph_mincuts/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graph_mincuts/text/yelp_polarity.yaml > log/transductive/graph_mincuts/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graph_mincuts/text/yelp_review_full.yaml > log/transductive/graph_mincuts/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graph_mincuts/vision/cifar10.yaml > log/transductive/graph_mincuts/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graph_mincuts/vision/cifar100.yaml > log/transductive/graph_mincuts/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graph_mincuts/vision/mnist.yaml > log/transductive/graph_mincuts/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graph_mincuts/vision/stl10.yaml > log/transductive/graph_mincuts/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graph_mincuts/vision/svhn.yaml > log/transductive/graph_mincuts/vision/svhn.log 2>&1
```

## transductive/graphhop

```
mkdir -p log/transductive/graphhop/audio log/transductive/graphhop/graph log/transductive/graphhop/tabular log/transductive/graphhop/text log/transductive/graphhop/vision
python -m bench.main --config bench/configs/experiments/best/transductive/graphhop/audio/speechcommands.yaml > log/transductive/graphhop/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphhop/audio/yesno.yaml > log/transductive/graphhop/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphhop/graph/citeseer.yaml > log/transductive/graphhop/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphhop/graph/cora.yaml > log/transductive/graphhop/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphhop/graph/pubmed.yaml > log/transductive/graphhop/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphhop/tabular/adult.yaml > log/transductive/graphhop/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphhop/tabular/breast_cancer.yaml > log/transductive/graphhop/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphhop/tabular/iris.yaml > log/transductive/graphhop/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphhop/tabular/toy.yaml > log/transductive/graphhop/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphhop/text/ag_news.yaml > log/transductive/graphhop/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphhop/text/amazon_polarity.yaml > log/transductive/graphhop/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphhop/text/amazon_reviews_multi_en.yaml > log/transductive/graphhop/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphhop/text/dbpedia_14.yaml > log/transductive/graphhop/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphhop/text/imdb.yaml > log/transductive/graphhop/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphhop/text/yelp_polarity.yaml > log/transductive/graphhop/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphhop/text/yelp_review_full.yaml > log/transductive/graphhop/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphhop/vision/cifar10.yaml > log/transductive/graphhop/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphhop/vision/cifar100.yaml > log/transductive/graphhop/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphhop/vision/mnist.yaml > log/transductive/graphhop/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphhop/vision/stl10.yaml > log/transductive/graphhop/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphhop/vision/svhn.yaml > log/transductive/graphhop/vision/svhn.log 2>&1
```

## transductive/graphsage

```
mkdir -p log/transductive/graphsage/audio log/transductive/graphsage/graph log/transductive/graphsage/tabular log/transductive/graphsage/text log/transductive/graphsage/vision
python -m bench.main --config bench/configs/experiments/best/transductive/graphsage/audio/speechcommands.yaml > log/transductive/graphsage/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphsage/audio/yesno.yaml > log/transductive/graphsage/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphsage/graph/citeseer.yaml > log/transductive/graphsage/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphsage/graph/cora.yaml > log/transductive/graphsage/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphsage/graph/pubmed.yaml > log/transductive/graphsage/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphsage/tabular/adult.yaml > log/transductive/graphsage/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphsage/tabular/breast_cancer.yaml > log/transductive/graphsage/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphsage/tabular/iris.yaml > log/transductive/graphsage/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphsage/tabular/toy.yaml > log/transductive/graphsage/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphsage/text/ag_news.yaml > log/transductive/graphsage/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphsage/text/amazon_polarity.yaml > log/transductive/graphsage/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphsage/text/amazon_reviews_multi_en.yaml > log/transductive/graphsage/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphsage/text/dbpedia_14.yaml > log/transductive/graphsage/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphsage/text/imdb.yaml > log/transductive/graphsage/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphsage/text/yelp_polarity.yaml > log/transductive/graphsage/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphsage/text/yelp_review_full.yaml > log/transductive/graphsage/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphsage/vision/cifar10.yaml > log/transductive/graphsage/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphsage/vision/cifar100.yaml > log/transductive/graphsage/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphsage/vision/mnist.yaml > log/transductive/graphsage/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphsage/vision/stl10.yaml > log/transductive/graphsage/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/graphsage/vision/svhn.yaml > log/transductive/graphsage/vision/svhn.log 2>&1
```

## transductive/h_gcn

```
mkdir -p log/transductive/h_gcn/audio log/transductive/h_gcn/graph log/transductive/h_gcn/tabular log/transductive/h_gcn/text log/transductive/h_gcn/vision
python -m bench.main --config bench/configs/experiments/best/transductive/h_gcn/audio/speechcommands.yaml > log/transductive/h_gcn/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/h_gcn/audio/yesno.yaml > log/transductive/h_gcn/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/h_gcn/graph/citeseer.yaml > log/transductive/h_gcn/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/h_gcn/graph/cora.yaml > log/transductive/h_gcn/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/h_gcn/graph/pubmed.yaml > log/transductive/h_gcn/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/h_gcn/tabular/adult.yaml > log/transductive/h_gcn/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/h_gcn/tabular/breast_cancer.yaml > log/transductive/h_gcn/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/h_gcn/tabular/iris.yaml > log/transductive/h_gcn/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/h_gcn/tabular/toy.yaml > log/transductive/h_gcn/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/h_gcn/text/ag_news.yaml > log/transductive/h_gcn/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/h_gcn/text/amazon_polarity.yaml > log/transductive/h_gcn/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/h_gcn/text/amazon_reviews_multi_en.yaml > log/transductive/h_gcn/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/h_gcn/text/dbpedia_14.yaml > log/transductive/h_gcn/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/h_gcn/text/imdb.yaml > log/transductive/h_gcn/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/h_gcn/text/yelp_polarity.yaml > log/transductive/h_gcn/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/h_gcn/text/yelp_review_full.yaml > log/transductive/h_gcn/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/h_gcn/vision/cifar10.yaml > log/transductive/h_gcn/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/h_gcn/vision/cifar100.yaml > log/transductive/h_gcn/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/h_gcn/vision/mnist.yaml > log/transductive/h_gcn/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/h_gcn/vision/stl10.yaml > log/transductive/h_gcn/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/h_gcn/vision/svhn.yaml > log/transductive/h_gcn/vision/svhn.log 2>&1
```

## transductive/label_propagation

```
mkdir -p log/transductive/label_propagation/audio log/transductive/label_propagation/graph log/transductive/label_propagation/tabular log/transductive/label_propagation/text log/transductive/label_propagation/vision
python -m bench.main --config bench/configs/experiments/best/transductive/label_propagation/audio/speechcommands.yaml > log/transductive/label_propagation/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_propagation/audio/yesno.yaml > log/transductive/label_propagation/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_propagation/graph/citeseer.yaml > log/transductive/label_propagation/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_propagation/graph/cora.yaml > log/transductive/label_propagation/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_propagation/graph/pubmed.yaml > log/transductive/label_propagation/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_propagation/tabular/adult.yaml > log/transductive/label_propagation/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_propagation/tabular/breast_cancer.yaml > log/transductive/label_propagation/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_propagation/tabular/iris.yaml > log/transductive/label_propagation/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_propagation/tabular/toy.yaml > log/transductive/label_propagation/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_propagation/text/ag_news.yaml > log/transductive/label_propagation/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_propagation/text/amazon_polarity.yaml > log/transductive/label_propagation/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_propagation/text/amazon_reviews_multi_en.yaml > log/transductive/label_propagation/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_propagation/text/dbpedia_14.yaml > log/transductive/label_propagation/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_propagation/text/imdb.yaml > log/transductive/label_propagation/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_propagation/text/yelp_polarity.yaml > log/transductive/label_propagation/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_propagation/text/yelp_review_full.yaml > log/transductive/label_propagation/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_propagation/vision/cifar10.yaml > log/transductive/label_propagation/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_propagation/vision/cifar100.yaml > log/transductive/label_propagation/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_propagation/vision/mnist.yaml > log/transductive/label_propagation/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_propagation/vision/stl10.yaml > log/transductive/label_propagation/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_propagation/vision/svhn.yaml > log/transductive/label_propagation/vision/svhn.log 2>&1
```

## transductive/label_spreading

```
mkdir -p log/transductive/label_spreading/audio log/transductive/label_spreading/graph log/transductive/label_spreading/tabular log/transductive/label_spreading/text log/transductive/label_spreading/vision
python -m bench.main --config bench/configs/experiments/best/transductive/label_spreading/audio/speechcommands.yaml > log/transductive/label_spreading/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_spreading/audio/yesno.yaml > log/transductive/label_spreading/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_spreading/graph/citeseer.yaml > log/transductive/label_spreading/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_spreading/graph/cora.yaml > log/transductive/label_spreading/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_spreading/graph/pubmed.yaml > log/transductive/label_spreading/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_spreading/tabular/adult.yaml > log/transductive/label_spreading/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_spreading/tabular/breast_cancer.yaml > log/transductive/label_spreading/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_spreading/tabular/iris.yaml > log/transductive/label_spreading/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_spreading/tabular/toy.yaml > log/transductive/label_spreading/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_spreading/text/ag_news.yaml > log/transductive/label_spreading/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_spreading/text/amazon_polarity.yaml > log/transductive/label_spreading/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_spreading/text/amazon_reviews_multi_en.yaml > log/transductive/label_spreading/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_spreading/text/dbpedia_14.yaml > log/transductive/label_spreading/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_spreading/text/imdb.yaml > log/transductive/label_spreading/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_spreading/text/yelp_polarity.yaml > log/transductive/label_spreading/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_spreading/text/yelp_review_full.yaml > log/transductive/label_spreading/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_spreading/vision/cifar10.yaml > log/transductive/label_spreading/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_spreading/vision/cifar100.yaml > log/transductive/label_spreading/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_spreading/vision/mnist.yaml > log/transductive/label_spreading/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_spreading/vision/stl10.yaml > log/transductive/label_spreading/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/label_spreading/vision/svhn.yaml > log/transductive/label_spreading/vision/svhn.log 2>&1
```

## transductive/laplace_learning

```
mkdir -p log/transductive/laplace_learning/audio log/transductive/laplace_learning/graph log/transductive/laplace_learning/tabular log/transductive/laplace_learning/text log/transductive/laplace_learning/vision
python -m bench.main --config bench/configs/experiments/best/transductive/laplace_learning/audio/speechcommands.yaml > log/transductive/laplace_learning/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/laplace_learning/audio/yesno.yaml > log/transductive/laplace_learning/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/laplace_learning/graph/citeseer.yaml > log/transductive/laplace_learning/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/laplace_learning/graph/cora.yaml > log/transductive/laplace_learning/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/laplace_learning/graph/pubmed.yaml > log/transductive/laplace_learning/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/laplace_learning/tabular/adult.yaml > log/transductive/laplace_learning/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/laplace_learning/tabular/breast_cancer.yaml > log/transductive/laplace_learning/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/laplace_learning/tabular/iris.yaml > log/transductive/laplace_learning/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/laplace_learning/tabular/toy.yaml > log/transductive/laplace_learning/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/laplace_learning/text/ag_news.yaml > log/transductive/laplace_learning/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/laplace_learning/text/amazon_polarity.yaml > log/transductive/laplace_learning/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/laplace_learning/text/amazon_reviews_multi_en.yaml > log/transductive/laplace_learning/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/laplace_learning/text/dbpedia_14.yaml > log/transductive/laplace_learning/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/laplace_learning/text/imdb.yaml > log/transductive/laplace_learning/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/laplace_learning/text/yelp_polarity.yaml > log/transductive/laplace_learning/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/laplace_learning/text/yelp_review_full.yaml > log/transductive/laplace_learning/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/laplace_learning/vision/cifar10.yaml > log/transductive/laplace_learning/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/laplace_learning/vision/cifar100.yaml > log/transductive/laplace_learning/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/laplace_learning/vision/mnist.yaml > log/transductive/laplace_learning/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/laplace_learning/vision/stl10.yaml > log/transductive/laplace_learning/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/laplace_learning/vision/svhn.yaml > log/transductive/laplace_learning/vision/svhn.log 2>&1
```

## transductive/lazy_random_walk

```
mkdir -p log/transductive/lazy_random_walk/audio log/transductive/lazy_random_walk/graph log/transductive/lazy_random_walk/tabular log/transductive/lazy_random_walk/text log/transductive/lazy_random_walk/vision
python -m bench.main --config bench/configs/experiments/best/transductive/lazy_random_walk/audio/speechcommands.yaml > log/transductive/lazy_random_walk/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/lazy_random_walk/audio/yesno.yaml > log/transductive/lazy_random_walk/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/lazy_random_walk/graph/citeseer.yaml > log/transductive/lazy_random_walk/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/lazy_random_walk/graph/cora.yaml > log/transductive/lazy_random_walk/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/lazy_random_walk/graph/pubmed.yaml > log/transductive/lazy_random_walk/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/lazy_random_walk/tabular/adult.yaml > log/transductive/lazy_random_walk/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/lazy_random_walk/tabular/breast_cancer.yaml > log/transductive/lazy_random_walk/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/lazy_random_walk/tabular/iris.yaml > log/transductive/lazy_random_walk/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/lazy_random_walk/tabular/toy.yaml > log/transductive/lazy_random_walk/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/lazy_random_walk/text/ag_news.yaml > log/transductive/lazy_random_walk/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/lazy_random_walk/text/amazon_polarity.yaml > log/transductive/lazy_random_walk/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/lazy_random_walk/text/amazon_reviews_multi_en.yaml > log/transductive/lazy_random_walk/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/lazy_random_walk/text/dbpedia_14.yaml > log/transductive/lazy_random_walk/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/lazy_random_walk/text/imdb.yaml > log/transductive/lazy_random_walk/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/lazy_random_walk/text/yelp_polarity.yaml > log/transductive/lazy_random_walk/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/lazy_random_walk/text/yelp_review_full.yaml > log/transductive/lazy_random_walk/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/lazy_random_walk/vision/cifar10.yaml > log/transductive/lazy_random_walk/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/lazy_random_walk/vision/cifar100.yaml > log/transductive/lazy_random_walk/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/lazy_random_walk/vision/mnist.yaml > log/transductive/lazy_random_walk/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/lazy_random_walk/vision/stl10.yaml > log/transductive/lazy_random_walk/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/lazy_random_walk/vision/svhn.yaml > log/transductive/lazy_random_walk/vision/svhn.log 2>&1
```

## transductive/n_gcn

```
mkdir -p log/transductive/n_gcn/audio log/transductive/n_gcn/graph log/transductive/n_gcn/tabular log/transductive/n_gcn/text log/transductive/n_gcn/vision
python -m bench.main --config bench/configs/experiments/best/transductive/n_gcn/audio/speechcommands.yaml > log/transductive/n_gcn/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/n_gcn/audio/yesno.yaml > log/transductive/n_gcn/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/n_gcn/graph/citeseer.yaml > log/transductive/n_gcn/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/n_gcn/graph/cora.yaml > log/transductive/n_gcn/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/n_gcn/graph/pubmed.yaml > log/transductive/n_gcn/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/n_gcn/tabular/adult.yaml > log/transductive/n_gcn/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/n_gcn/tabular/breast_cancer.yaml > log/transductive/n_gcn/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/n_gcn/tabular/iris.yaml > log/transductive/n_gcn/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/n_gcn/tabular/toy.yaml > log/transductive/n_gcn/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/n_gcn/text/ag_news.yaml > log/transductive/n_gcn/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/n_gcn/text/amazon_polarity.yaml > log/transductive/n_gcn/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/n_gcn/text/amazon_reviews_multi_en.yaml > log/transductive/n_gcn/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/n_gcn/text/dbpedia_14.yaml > log/transductive/n_gcn/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/n_gcn/text/imdb.yaml > log/transductive/n_gcn/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/n_gcn/text/yelp_polarity.yaml > log/transductive/n_gcn/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/n_gcn/text/yelp_review_full.yaml > log/transductive/n_gcn/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/n_gcn/vision/cifar10.yaml > log/transductive/n_gcn/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/n_gcn/vision/cifar100.yaml > log/transductive/n_gcn/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/n_gcn/vision/mnist.yaml > log/transductive/n_gcn/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/n_gcn/vision/stl10.yaml > log/transductive/n_gcn/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/n_gcn/vision/svhn.yaml > log/transductive/n_gcn/vision/svhn.log 2>&1
```

## transductive/node2vec

```
mkdir -p log/transductive/node2vec/audio log/transductive/node2vec/graph log/transductive/node2vec/tabular log/transductive/node2vec/text log/transductive/node2vec/vision
python -m bench.main --config bench/configs/experiments/best/transductive/node2vec/audio/speechcommands.yaml > log/transductive/node2vec/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/node2vec/audio/yesno.yaml > log/transductive/node2vec/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/node2vec/graph/citeseer.yaml > log/transductive/node2vec/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/node2vec/graph/cora.yaml > log/transductive/node2vec/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/node2vec/graph/pubmed.yaml > log/transductive/node2vec/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/node2vec/tabular/adult.yaml > log/transductive/node2vec/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/node2vec/tabular/breast_cancer.yaml > log/transductive/node2vec/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/node2vec/tabular/iris.yaml > log/transductive/node2vec/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/node2vec/tabular/toy.yaml > log/transductive/node2vec/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/node2vec/text/ag_news.yaml > log/transductive/node2vec/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/node2vec/text/amazon_polarity.yaml > log/transductive/node2vec/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/node2vec/text/amazon_reviews_multi_en.yaml > log/transductive/node2vec/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/node2vec/text/dbpedia_14.yaml > log/transductive/node2vec/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/node2vec/text/imdb.yaml > log/transductive/node2vec/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/node2vec/text/yelp_polarity.yaml > log/transductive/node2vec/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/node2vec/text/yelp_review_full.yaml > log/transductive/node2vec/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/node2vec/vision/cifar10.yaml > log/transductive/node2vec/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/node2vec/vision/cifar100.yaml > log/transductive/node2vec/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/node2vec/vision/mnist.yaml > log/transductive/node2vec/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/node2vec/vision/stl10.yaml > log/transductive/node2vec/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/node2vec/vision/svhn.yaml > log/transductive/node2vec/vision/svhn.log 2>&1
```

## transductive/p_laplace_learning

```
mkdir -p log/transductive/p_laplace_learning/audio log/transductive/p_laplace_learning/graph log/transductive/p_laplace_learning/tabular log/transductive/p_laplace_learning/text log/transductive/p_laplace_learning/vision
python -m bench.main --config bench/configs/experiments/best/transductive/p_laplace_learning/audio/speechcommands.yaml > log/transductive/p_laplace_learning/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/p_laplace_learning/audio/yesno.yaml > log/transductive/p_laplace_learning/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/p_laplace_learning/graph/citeseer.yaml > log/transductive/p_laplace_learning/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/p_laplace_learning/graph/cora.yaml > log/transductive/p_laplace_learning/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/p_laplace_learning/graph/pubmed.yaml > log/transductive/p_laplace_learning/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/p_laplace_learning/tabular/adult.yaml > log/transductive/p_laplace_learning/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/p_laplace_learning/tabular/breast_cancer.yaml > log/transductive/p_laplace_learning/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/p_laplace_learning/tabular/iris.yaml > log/transductive/p_laplace_learning/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/p_laplace_learning/tabular/toy.yaml > log/transductive/p_laplace_learning/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/p_laplace_learning/text/ag_news.yaml > log/transductive/p_laplace_learning/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/p_laplace_learning/text/amazon_polarity.yaml > log/transductive/p_laplace_learning/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/p_laplace_learning/text/amazon_reviews_multi_en.yaml > log/transductive/p_laplace_learning/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/p_laplace_learning/text/dbpedia_14.yaml > log/transductive/p_laplace_learning/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/p_laplace_learning/text/imdb.yaml > log/transductive/p_laplace_learning/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/p_laplace_learning/text/yelp_polarity.yaml > log/transductive/p_laplace_learning/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/p_laplace_learning/text/yelp_review_full.yaml > log/transductive/p_laplace_learning/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/p_laplace_learning/vision/cifar10.yaml > log/transductive/p_laplace_learning/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/p_laplace_learning/vision/cifar100.yaml > log/transductive/p_laplace_learning/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/p_laplace_learning/vision/mnist.yaml > log/transductive/p_laplace_learning/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/p_laplace_learning/vision/stl10.yaml > log/transductive/p_laplace_learning/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/p_laplace_learning/vision/svhn.yaml > log/transductive/p_laplace_learning/vision/svhn.log 2>&1
```

## transductive/planetoid

```
mkdir -p log/transductive/planetoid/audio log/transductive/planetoid/graph log/transductive/planetoid/tabular log/transductive/planetoid/text log/transductive/planetoid/vision
python -m bench.main --config bench/configs/experiments/best/transductive/planetoid/audio/speechcommands.yaml > log/transductive/planetoid/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/planetoid/audio/yesno.yaml > log/transductive/planetoid/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/planetoid/graph/citeseer.yaml > log/transductive/planetoid/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/planetoid/graph/cora.yaml > log/transductive/planetoid/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/planetoid/graph/pubmed.yaml > log/transductive/planetoid/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/planetoid/tabular/adult.yaml > log/transductive/planetoid/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/planetoid/tabular/breast_cancer.yaml > log/transductive/planetoid/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/planetoid/tabular/iris.yaml > log/transductive/planetoid/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/planetoid/tabular/toy.yaml > log/transductive/planetoid/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/planetoid/text/ag_news.yaml > log/transductive/planetoid/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/planetoid/text/amazon_polarity.yaml > log/transductive/planetoid/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/planetoid/text/amazon_reviews_multi_en.yaml > log/transductive/planetoid/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/planetoid/text/dbpedia_14.yaml > log/transductive/planetoid/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/planetoid/text/imdb.yaml > log/transductive/planetoid/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/planetoid/text/yelp_polarity.yaml > log/transductive/planetoid/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/planetoid/text/yelp_review_full.yaml > log/transductive/planetoid/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/planetoid/vision/cifar10.yaml > log/transductive/planetoid/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/planetoid/vision/cifar100.yaml > log/transductive/planetoid/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/planetoid/vision/mnist.yaml > log/transductive/planetoid/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/planetoid/vision/stl10.yaml > log/transductive/planetoid/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/planetoid/vision/svhn.yaml > log/transductive/planetoid/vision/svhn.log 2>&1
```

## transductive/poisson_learning

```
mkdir -p log/transductive/poisson_learning/audio log/transductive/poisson_learning/graph log/transductive/poisson_learning/tabular log/transductive/poisson_learning/text log/transductive/poisson_learning/vision
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_learning/audio/speechcommands.yaml > log/transductive/poisson_learning/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_learning/audio/yesno.yaml > log/transductive/poisson_learning/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_learning/graph/citeseer.yaml > log/transductive/poisson_learning/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_learning/graph/cora.yaml > log/transductive/poisson_learning/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_learning/graph/pubmed.yaml > log/transductive/poisson_learning/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_learning/tabular/adult.yaml > log/transductive/poisson_learning/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_learning/tabular/breast_cancer.yaml > log/transductive/poisson_learning/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_learning/tabular/iris.yaml > log/transductive/poisson_learning/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_learning/tabular/toy.yaml > log/transductive/poisson_learning/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_learning/text/ag_news.yaml > log/transductive/poisson_learning/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_learning/text/amazon_polarity.yaml > log/transductive/poisson_learning/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_learning/text/amazon_reviews_multi_en.yaml > log/transductive/poisson_learning/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_learning/text/dbpedia_14.yaml > log/transductive/poisson_learning/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_learning/text/imdb.yaml > log/transductive/poisson_learning/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_learning/text/yelp_polarity.yaml > log/transductive/poisson_learning/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_learning/text/yelp_review_full.yaml > log/transductive/poisson_learning/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_learning/vision/cifar10.yaml > log/transductive/poisson_learning/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_learning/vision/cifar100.yaml > log/transductive/poisson_learning/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_learning/vision/mnist.yaml > log/transductive/poisson_learning/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_learning/vision/stl10.yaml > log/transductive/poisson_learning/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_learning/vision/svhn.yaml > log/transductive/poisson_learning/vision/svhn.log 2>&1
```

## transductive/poisson_mbo

```
mkdir -p log/transductive/poisson_mbo/audio log/transductive/poisson_mbo/graph log/transductive/poisson_mbo/tabular log/transductive/poisson_mbo/text log/transductive/poisson_mbo/vision
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_mbo/audio/speechcommands.yaml > log/transductive/poisson_mbo/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_mbo/audio/yesno.yaml > log/transductive/poisson_mbo/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_mbo/graph/citeseer.yaml > log/transductive/poisson_mbo/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_mbo/graph/cora.yaml > log/transductive/poisson_mbo/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_mbo/graph/pubmed.yaml > log/transductive/poisson_mbo/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_mbo/tabular/adult.yaml > log/transductive/poisson_mbo/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_mbo/tabular/breast_cancer.yaml > log/transductive/poisson_mbo/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_mbo/tabular/iris.yaml > log/transductive/poisson_mbo/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_mbo/tabular/toy.yaml > log/transductive/poisson_mbo/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_mbo/text/ag_news.yaml > log/transductive/poisson_mbo/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_mbo/text/amazon_polarity.yaml > log/transductive/poisson_mbo/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_mbo/text/amazon_reviews_multi_en.yaml > log/transductive/poisson_mbo/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_mbo/text/dbpedia_14.yaml > log/transductive/poisson_mbo/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_mbo/text/imdb.yaml > log/transductive/poisson_mbo/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_mbo/text/yelp_polarity.yaml > log/transductive/poisson_mbo/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_mbo/text/yelp_review_full.yaml > log/transductive/poisson_mbo/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_mbo/vision/cifar10.yaml > log/transductive/poisson_mbo/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_mbo/vision/cifar100.yaml > log/transductive/poisson_mbo/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_mbo/vision/mnist.yaml > log/transductive/poisson_mbo/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_mbo/vision/stl10.yaml > log/transductive/poisson_mbo/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/poisson_mbo/vision/svhn.yaml > log/transductive/poisson_mbo/vision/svhn.log 2>&1
```

## transductive/sgc

```
mkdir -p log/transductive/sgc/audio log/transductive/sgc/graph log/transductive/sgc/tabular log/transductive/sgc/text log/transductive/sgc/vision
python -m bench.main --config bench/configs/experiments/best/transductive/sgc/audio/speechcommands.yaml > log/transductive/sgc/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/sgc/audio/yesno.yaml > log/transductive/sgc/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/sgc/graph/citeseer.yaml > log/transductive/sgc/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/sgc/graph/cora.yaml > log/transductive/sgc/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/sgc/graph/pubmed.yaml > log/transductive/sgc/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/sgc/tabular/adult.yaml > log/transductive/sgc/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/sgc/tabular/breast_cancer.yaml > log/transductive/sgc/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/sgc/tabular/iris.yaml > log/transductive/sgc/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/sgc/tabular/toy.yaml > log/transductive/sgc/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/sgc/text/ag_news.yaml > log/transductive/sgc/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/sgc/text/amazon_polarity.yaml > log/transductive/sgc/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/sgc/text/amazon_reviews_multi_en.yaml > log/transductive/sgc/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/sgc/text/dbpedia_14.yaml > log/transductive/sgc/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/sgc/text/imdb.yaml > log/transductive/sgc/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/sgc/text/yelp_polarity.yaml > log/transductive/sgc/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/sgc/text/yelp_review_full.yaml > log/transductive/sgc/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/sgc/vision/cifar10.yaml > log/transductive/sgc/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/sgc/vision/cifar100.yaml > log/transductive/sgc/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/sgc/vision/mnist.yaml > log/transductive/sgc/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/sgc/vision/stl10.yaml > log/transductive/sgc/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/sgc/vision/svhn.yaml > log/transductive/sgc/vision/svhn.log 2>&1
```

## transductive/tsvm

```
mkdir -p log/transductive/tsvm/audio log/transductive/tsvm/graph log/transductive/tsvm/tabular log/transductive/tsvm/text log/transductive/tsvm/vision
python -m bench.main --config bench/configs/experiments/best/transductive/tsvm/audio/speechcommands.yaml > log/transductive/tsvm/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/tsvm/audio/yesno.yaml > log/transductive/tsvm/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/tsvm/graph/citeseer.yaml > log/transductive/tsvm/graph/citeseer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/tsvm/graph/cora.yaml > log/transductive/tsvm/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/tsvm/graph/pubmed.yaml > log/transductive/tsvm/graph/pubmed.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/tsvm/tabular/adult.yaml > log/transductive/tsvm/tabular/adult.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/tsvm/tabular/breast_cancer.yaml > log/transductive/tsvm/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/tsvm/tabular/iris.yaml > log/transductive/tsvm/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/tsvm/tabular/toy.yaml > log/transductive/tsvm/tabular/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/tsvm/text/ag_news.yaml > log/transductive/tsvm/text/ag_news.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/tsvm/text/amazon_polarity.yaml > log/transductive/tsvm/text/amazon_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/tsvm/text/amazon_reviews_multi_en.yaml > log/transductive/tsvm/text/amazon_reviews_multi_en.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/tsvm/text/dbpedia_14.yaml > log/transductive/tsvm/text/dbpedia_14.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/tsvm/text/imdb.yaml > log/transductive/tsvm/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/tsvm/text/yelp_polarity.yaml > log/transductive/tsvm/text/yelp_polarity.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/tsvm/text/yelp_review_full.yaml > log/transductive/tsvm/text/yelp_review_full.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/tsvm/vision/cifar10.yaml > log/transductive/tsvm/vision/cifar10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/tsvm/vision/cifar100.yaml > log/transductive/tsvm/vision/cifar100.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/tsvm/vision/mnist.yaml > log/transductive/tsvm/vision/mnist.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/tsvm/vision/stl10.yaml > log/transductive/tsvm/vision/stl10.log 2>&1
python -m bench.main --config bench/configs/experiments/best/transductive/tsvm/vision/svhn.yaml > log/transductive/tsvm/vision/svhn.log 2>&1
```
