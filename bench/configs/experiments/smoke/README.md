# Smoke configs (commands by method)

## inductive/adamatch

```
mkdir -p log/inductive/adamatch/audio log/inductive/adamatch/graph log/inductive/adamatch/tabular log/inductive/adamatch/text log/inductive/adamatch/vision
python -m bench.main --config bench/configs/experiments/smoke/inductive/adamatch/audio/speechcommands.yaml > log/inductive/adamatch/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/adamatch/graph/cora.yaml > log/inductive/adamatch/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/adamatch/tabular/iris.yaml > log/inductive/adamatch/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/adamatch/text/imdb.yaml > log/inductive/adamatch/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/adamatch/vision/mnist.yaml > log/inductive/adamatch/vision/mnist.log 2>&1
```

## inductive/adsh

```
mkdir -p log/inductive/adsh/audio log/inductive/adsh/graph log/inductive/adsh/tabular log/inductive/adsh/text log/inductive/adsh/vision
python -m bench.main --config bench/configs/experiments/smoke/inductive/adsh/audio/speechcommands.yaml > log/inductive/adsh/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/adsh/graph/cora.yaml > log/inductive/adsh/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/adsh/tabular/iris.yaml > log/inductive/adsh/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/adsh/text/imdb.yaml > log/inductive/adsh/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/adsh/vision/mnist.yaml > log/inductive/adsh/vision/mnist.log 2>&1
```

## inductive/co_training

```
mkdir -p log/inductive/co_training/audio log/inductive/co_training/graph log/inductive/co_training/tabular log/inductive/co_training/text log/inductive/co_training/vision
python -m bench.main --config bench/configs/experiments/smoke/inductive/co_training/audio/speechcommands.yaml > log/inductive/co_training/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/co_training/graph/cora.yaml > log/inductive/co_training/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/co_training/tabular/iris.yaml > log/inductive/co_training/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/co_training/text/imdb.yaml > log/inductive/co_training/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/co_training/vision/mnist.yaml > log/inductive/co_training/vision/mnist.log 2>&1
```

## inductive/comatch

```
mkdir -p log/inductive/comatch/audio log/inductive/comatch/graph log/inductive/comatch/tabular log/inductive/comatch/text log/inductive/comatch/vision
python -m bench.main --config bench/configs/experiments/smoke/inductive/comatch/audio/speechcommands.yaml > log/inductive/comatch/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/comatch/graph/cora.yaml > log/inductive/comatch/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/comatch/tabular/iris.yaml > log/inductive/comatch/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/comatch/text/imdb.yaml > log/inductive/comatch/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/comatch/vision/mnist.yaml > log/inductive/comatch/vision/mnist.log 2>&1
```

## inductive/daso

```
mkdir -p log/inductive/daso/audio log/inductive/daso/graph log/inductive/daso/tabular log/inductive/daso/text log/inductive/daso/vision
python -m bench.main --config bench/configs/experiments/smoke/inductive/daso/audio/speechcommands.yaml > log/inductive/daso/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/daso/graph/cora.yaml > log/inductive/daso/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/daso/tabular/iris.yaml > log/inductive/daso/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/daso/text/imdb.yaml > log/inductive/daso/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/daso/vision/mnist.yaml > log/inductive/daso/vision/mnist.log 2>&1
```

## inductive/deep_co_training

```
mkdir -p log/inductive/deep_co_training/audio log/inductive/deep_co_training/graph log/inductive/deep_co_training/tabular log/inductive/deep_co_training/text log/inductive/deep_co_training/vision
python -m bench.main --config bench/configs/experiments/smoke/inductive/deep_co_training/audio/speechcommands.yaml > log/inductive/deep_co_training/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/deep_co_training/graph/cora.yaml > log/inductive/deep_co_training/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/deep_co_training/tabular/iris.yaml > log/inductive/deep_co_training/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/deep_co_training/text/imdb.yaml > log/inductive/deep_co_training/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/deep_co_training/vision/mnist.yaml > log/inductive/deep_co_training/vision/mnist.log 2>&1
```

## inductive/defixmatch

```
mkdir -p log/inductive/defixmatch/audio log/inductive/defixmatch/graph log/inductive/defixmatch/tabular log/inductive/defixmatch/text log/inductive/defixmatch/vision
python -m bench.main --config bench/configs/experiments/smoke/inductive/defixmatch/audio/speechcommands.yaml > log/inductive/defixmatch/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/defixmatch/graph/cora.yaml > log/inductive/defixmatch/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/defixmatch/tabular/iris.yaml > log/inductive/defixmatch/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/defixmatch/text/imdb.yaml > log/inductive/defixmatch/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/defixmatch/vision/mnist.yaml > log/inductive/defixmatch/vision/mnist.log 2>&1
```

## inductive/democratic_co_learning

```
mkdir -p log/inductive/democratic_co_learning/audio log/inductive/democratic_co_learning/graph log/inductive/democratic_co_learning/tabular log/inductive/democratic_co_learning/text log/inductive/democratic_co_learning/vision
python -m bench.main --config bench/configs/experiments/smoke/inductive/democratic_co_learning/audio/speechcommands.yaml > log/inductive/democratic_co_learning/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/democratic_co_learning/graph/cora.yaml > log/inductive/democratic_co_learning/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/democratic_co_learning/tabular/iris.yaml > log/inductive/democratic_co_learning/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/democratic_co_learning/text/imdb.yaml > log/inductive/democratic_co_learning/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/democratic_co_learning/vision/mnist.yaml > log/inductive/democratic_co_learning/vision/mnist.log 2>&1
```

## inductive/fixmatch

```
mkdir -p log/inductive/fixmatch/audio log/inductive/fixmatch/graph log/inductive/fixmatch/tabular log/inductive/fixmatch/text log/inductive/fixmatch/vision
python -m bench.main --config bench/configs/experiments/smoke/inductive/fixmatch/audio/speechcommands.yaml > log/inductive/fixmatch/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/fixmatch/graph/cora.yaml > log/inductive/fixmatch/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/fixmatch/tabular/iris.yaml > log/inductive/fixmatch/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/fixmatch/text/imdb.yaml > log/inductive/fixmatch/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/fixmatch/vision/mnist.yaml > log/inductive/fixmatch/vision/mnist.log 2>&1
```

## inductive/flexmatch

```
mkdir -p log/inductive/flexmatch/audio log/inductive/flexmatch/graph log/inductive/flexmatch/tabular log/inductive/flexmatch/text log/inductive/flexmatch/vision
python -m bench.main --config bench/configs/experiments/smoke/inductive/flexmatch/audio/speechcommands.yaml > log/inductive/flexmatch/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/flexmatch/graph/cora.yaml > log/inductive/flexmatch/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/flexmatch/tabular/iris.yaml > log/inductive/flexmatch/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/flexmatch/text/imdb.yaml > log/inductive/flexmatch/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/flexmatch/vision/mnist.yaml > log/inductive/flexmatch/vision/mnist.log 2>&1
```

## inductive/free_match

```
mkdir -p log/inductive/free_match/audio log/inductive/free_match/graph log/inductive/free_match/tabular log/inductive/free_match/text log/inductive/free_match/vision
python -m bench.main --config bench/configs/experiments/smoke/inductive/free_match/audio/speechcommands.yaml > log/inductive/free_match/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/free_match/graph/cora.yaml > log/inductive/free_match/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/free_match/tabular/iris.yaml > log/inductive/free_match/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/free_match/text/imdb.yaml > log/inductive/free_match/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/free_match/vision/mnist.yaml > log/inductive/free_match/vision/mnist.log 2>&1
```

## inductive/mean_teacher

```
mkdir -p log/inductive/mean_teacher/audio log/inductive/mean_teacher/graph log/inductive/mean_teacher/tabular log/inductive/mean_teacher/text log/inductive/mean_teacher/vision
python -m bench.main --config bench/configs/experiments/smoke/inductive/mean_teacher/audio/speechcommands.yaml > log/inductive/mean_teacher/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/mean_teacher/audio/yesno.yaml > log/inductive/mean_teacher/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/mean_teacher/graph/cora.yaml > log/inductive/mean_teacher/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/mean_teacher/tabular/iris.yaml > log/inductive/mean_teacher/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/mean_teacher/text/imdb.yaml > log/inductive/mean_teacher/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/mean_teacher/vision/mnist.yaml > log/inductive/mean_teacher/vision/mnist.log 2>&1
```

## inductive/meta_pseudo_labels

```
mkdir -p log/inductive/meta_pseudo_labels/audio log/inductive/meta_pseudo_labels/graph log/inductive/meta_pseudo_labels/tabular log/inductive/meta_pseudo_labels/text log/inductive/meta_pseudo_labels/vision
python -m bench.main --config bench/configs/experiments/smoke/inductive/meta_pseudo_labels/audio/speechcommands.yaml > log/inductive/meta_pseudo_labels/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/meta_pseudo_labels/audio/yesno.yaml > log/inductive/meta_pseudo_labels/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/meta_pseudo_labels/graph/cora.yaml > log/inductive/meta_pseudo_labels/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/meta_pseudo_labels/tabular/iris.yaml > log/inductive/meta_pseudo_labels/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/meta_pseudo_labels/text/imdb.yaml > log/inductive/meta_pseudo_labels/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/meta_pseudo_labels/vision/mnist.yaml > log/inductive/meta_pseudo_labels/vision/mnist.log 2>&1
```

## inductive/mixmatch

```
mkdir -p log/inductive/mixmatch/audio log/inductive/mixmatch/graph log/inductive/mixmatch/tabular log/inductive/mixmatch/text log/inductive/mixmatch/vision
python -m bench.main --config bench/configs/experiments/smoke/inductive/mixmatch/audio/speechcommands.yaml > log/inductive/mixmatch/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/mixmatch/graph/cora.yaml > log/inductive/mixmatch/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/mixmatch/tabular/iris.yaml > log/inductive/mixmatch/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/mixmatch/text/imdb.yaml > log/inductive/mixmatch/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/mixmatch/vision/mnist.yaml > log/inductive/mixmatch/vision/mnist.log 2>&1
```

## inductive/noisy_student

```
mkdir -p log/inductive/noisy_student/audio log/inductive/noisy_student/graph log/inductive/noisy_student/tabular log/inductive/noisy_student/text log/inductive/noisy_student/vision
python -m bench.main --config bench/configs/experiments/smoke/inductive/noisy_student/audio/speechcommands.yaml > log/inductive/noisy_student/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/noisy_student/graph/cora.yaml > log/inductive/noisy_student/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/noisy_student/tabular/iris.yaml > log/inductive/noisy_student/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/noisy_student/text/imdb.yaml > log/inductive/noisy_student/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/noisy_student/vision/mnist.yaml > log/inductive/noisy_student/vision/mnist.log 2>&1
```

## inductive/pi_model

```
mkdir -p log/inductive/pi_model/audio log/inductive/pi_model/graph log/inductive/pi_model/tabular log/inductive/pi_model/text log/inductive/pi_model/vision
python -m bench.main --config bench/configs/experiments/smoke/inductive/pi_model/audio/speechcommands.yaml > log/inductive/pi_model/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/pi_model/graph/cora.yaml > log/inductive/pi_model/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/pi_model/tabular/iris.yaml > log/inductive/pi_model/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/pi_model/text/imdb.yaml > log/inductive/pi_model/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/pi_model/vision/mnist.yaml > log/inductive/pi_model/vision/mnist.log 2>&1
```

## inductive/pseudo_label

```
mkdir -p log/inductive/pseudo_label/audio log/inductive/pseudo_label/graph log/inductive/pseudo_label/tabular log/inductive/pseudo_label/text log/inductive/pseudo_label/vision
python -m bench.main --config bench/configs/experiments/smoke/inductive/pseudo_label/audio/speechcommands.yaml > log/inductive/pseudo_label/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/pseudo_label/graph/cora.yaml > log/inductive/pseudo_label/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/pseudo_label/tabular/iris.yaml > log/inductive/pseudo_label/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/pseudo_label/text/imdb.yaml > log/inductive/pseudo_label/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/pseudo_label/vision/mnist.yaml > log/inductive/pseudo_label/vision/mnist.log 2>&1
```

## inductive/s4vm

```
mkdir -p log/inductive/s4vm/audio log/inductive/s4vm/graph log/inductive/s4vm/tabular log/inductive/s4vm/text log/inductive/s4vm/vision
python -m bench.main --config bench/configs/experiments/smoke/inductive/s4vm/audio/speechcommands.yaml > log/inductive/s4vm/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/s4vm/graph/cora.yaml > log/inductive/s4vm/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/s4vm/tabular/breast_cancer.yaml > log/inductive/s4vm/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/s4vm/text/imdb.yaml > log/inductive/s4vm/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/s4vm/vision/mnist.yaml > log/inductive/s4vm/vision/mnist.log 2>&1
```

## inductive/self_training

```
mkdir -p log/inductive/self_training/audio log/inductive/self_training/graph log/inductive/self_training/tabular log/inductive/self_training/text log/inductive/self_training/vision
python -m bench.main --config bench/configs/experiments/smoke/inductive/self_training/audio/speechcommands.yaml > log/inductive/self_training/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/self_training/graph/cora.yaml > log/inductive/self_training/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/self_training/tabular/iris.yaml > log/inductive/self_training/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/self_training/text/imdb.yaml > log/inductive/self_training/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/self_training/vision/mnist.yaml > log/inductive/self_training/vision/mnist.log 2>&1
```

## inductive/setred

```
mkdir -p log/inductive/setred/audio log/inductive/setred/graph log/inductive/setred/tabular log/inductive/setred/text log/inductive/setred/vision
python -m bench.main --config bench/configs/experiments/smoke/inductive/setred/audio/speechcommands.yaml > log/inductive/setred/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/setred/graph/cora.yaml > log/inductive/setred/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/setred/tabular/iris.yaml > log/inductive/setred/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/setred/text/imdb.yaml > log/inductive/setred/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/setred/vision/mnist.yaml > log/inductive/setred/vision/mnist.log 2>&1
```

## inductive/simclr_v2

```
mkdir -p log/inductive/simclr_v2/audio log/inductive/simclr_v2/graph log/inductive/simclr_v2/tabular log/inductive/simclr_v2/text log/inductive/simclr_v2/vision
python -m bench.main --config bench/configs/experiments/smoke/inductive/simclr_v2/audio/speechcommands.yaml > log/inductive/simclr_v2/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/simclr_v2/graph/cora.yaml > log/inductive/simclr_v2/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/simclr_v2/tabular/iris.yaml > log/inductive/simclr_v2/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/simclr_v2/text/imdb.yaml > log/inductive/simclr_v2/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/simclr_v2/vision/mnist.yaml > log/inductive/simclr_v2/vision/mnist.log 2>&1
```

## inductive/softmatch

```
mkdir -p log/inductive/softmatch/audio log/inductive/softmatch/graph log/inductive/softmatch/tabular log/inductive/softmatch/text log/inductive/softmatch/vision
python -m bench.main --config bench/configs/experiments/smoke/inductive/softmatch/audio/speechcommands.yaml > log/inductive/softmatch/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/softmatch/graph/cora.yaml > log/inductive/softmatch/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/softmatch/tabular/iris.yaml > log/inductive/softmatch/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/softmatch/text/imdb.yaml > log/inductive/softmatch/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/softmatch/vision/mnist.yaml > log/inductive/softmatch/vision/mnist.log 2>&1
```

## inductive/temporal_ensembling

```
mkdir -p log/inductive/temporal_ensembling/audio log/inductive/temporal_ensembling/graph log/inductive/temporal_ensembling/tabular log/inductive/temporal_ensembling/text log/inductive/temporal_ensembling/vision
python -m bench.main --config bench/configs/experiments/smoke/inductive/temporal_ensembling/audio/speechcommands.yaml > log/inductive/temporal_ensembling/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/temporal_ensembling/graph/cora.yaml > log/inductive/temporal_ensembling/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/temporal_ensembling/tabular/iris.yaml > log/inductive/temporal_ensembling/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/temporal_ensembling/text/imdb.yaml > log/inductive/temporal_ensembling/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/temporal_ensembling/vision/mnist.yaml > log/inductive/temporal_ensembling/vision/mnist.log 2>&1
```

## inductive/tri_training

```
mkdir -p log/inductive/tri_training/audio log/inductive/tri_training/graph log/inductive/tri_training/tabular log/inductive/tri_training/text log/inductive/tri_training/vision
python -m bench.main --config bench/configs/experiments/smoke/inductive/tri_training/audio/speechcommands.yaml > log/inductive/tri_training/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/tri_training/graph/cora.yaml > log/inductive/tri_training/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/tri_training/tabular/iris.yaml > log/inductive/tri_training/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/tri_training/text/imdb.yaml > log/inductive/tri_training/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/tri_training/vision/mnist.yaml > log/inductive/tri_training/vision/mnist.log 2>&1
```

## inductive/trinet

```
mkdir -p log/inductive/trinet/audio log/inductive/trinet/graph log/inductive/trinet/tabular log/inductive/trinet/text log/inductive/trinet/vision
python -m bench.main --config bench/configs/experiments/smoke/inductive/trinet/audio/speechcommands.yaml > log/inductive/trinet/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/trinet/graph/cora.yaml > log/inductive/trinet/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/trinet/tabular/iris.yaml > log/inductive/trinet/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/trinet/text/imdb.yaml > log/inductive/trinet/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/trinet/vision/mnist.yaml > log/inductive/trinet/vision/mnist.log 2>&1
```

## inductive/tsvm

```
mkdir -p log/inductive/tsvm/audio log/inductive/tsvm/graph log/inductive/tsvm/tabular log/inductive/tsvm/text log/inductive/tsvm/vision
python -m bench.main --config bench/configs/experiments/smoke/inductive/tsvm/audio/speechcommands.yaml > log/inductive/tsvm/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/tsvm/graph/cora.yaml > log/inductive/tsvm/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/tsvm/tabular/breast_cancer.yaml > log/inductive/tsvm/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/tsvm/text/imdb.yaml > log/inductive/tsvm/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/tsvm/vision/mnist.yaml > log/inductive/tsvm/vision/mnist.log 2>&1
```

## inductive/uda

```
mkdir -p log/inductive/uda/audio log/inductive/uda/graph log/inductive/uda/tabular log/inductive/uda/text log/inductive/uda/vision
python -m bench.main --config bench/configs/experiments/smoke/inductive/uda/audio/speechcommands.yaml > log/inductive/uda/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/uda/audio/yesno.yaml > log/inductive/uda/audio/yesno.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/uda/graph/cora.yaml > log/inductive/uda/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/uda/tabular/iris.yaml > log/inductive/uda/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/uda/text/imdb.yaml > log/inductive/uda/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/uda/vision/mnist.yaml > log/inductive/uda/vision/mnist.log 2>&1
```

## inductive/vat

```
mkdir -p log/inductive/vat/audio log/inductive/vat/graph log/inductive/vat/tabular log/inductive/vat/text log/inductive/vat/vision
python -m bench.main --config bench/configs/experiments/smoke/inductive/vat/audio/speechcommands.yaml > log/inductive/vat/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/vat/graph/cora.yaml > log/inductive/vat/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/vat/tabular/iris.yaml > log/inductive/vat/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/vat/text/imdb.yaml > log/inductive/vat/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/inductive/vat/vision/mnist.yaml > log/inductive/vat/vision/mnist.log 2>&1
```

## transductive/appnp

```
mkdir -p log/transductive/appnp/audio log/transductive/appnp/graph log/transductive/appnp/tabular log/transductive/appnp/text log/transductive/appnp/vision
python -m bench.main --config bench/configs/experiments/smoke/transductive/appnp/audio/speechcommands.yaml > log/transductive/appnp/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/appnp/graph/cora.yaml > log/transductive/appnp/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/appnp/tabular/iris.yaml > log/transductive/appnp/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/appnp/text/imdb.yaml > log/transductive/appnp/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/appnp/vision/mnist.yaml > log/transductive/appnp/vision/mnist.log 2>&1
```

## transductive/chebnet

```
mkdir -p log/transductive/chebnet/audio log/transductive/chebnet/graph log/transductive/chebnet/tabular log/transductive/chebnet/text log/transductive/chebnet/vision
python -m bench.main --config bench/configs/experiments/smoke/transductive/chebnet/audio/speechcommands.yaml > log/transductive/chebnet/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/chebnet/graph/cora.yaml > log/transductive/chebnet/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/chebnet/tabular/iris.yaml > log/transductive/chebnet/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/chebnet/text/imdb.yaml > log/transductive/chebnet/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/chebnet/vision/mnist.yaml > log/transductive/chebnet/vision/mnist.log 2>&1
```

## transductive/dynamic_label_propagation

```
mkdir -p log/transductive/dynamic_label_propagation/audio log/transductive/dynamic_label_propagation/graph log/transductive/dynamic_label_propagation/tabular log/transductive/dynamic_label_propagation/text log/transductive/dynamic_label_propagation/vision
python -m bench.main --config bench/configs/experiments/smoke/transductive/dynamic_label_propagation/audio/speechcommands.yaml > log/transductive/dynamic_label_propagation/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/dynamic_label_propagation/graph/cora.yaml > log/transductive/dynamic_label_propagation/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/dynamic_label_propagation/tabular/iris.yaml > log/transductive/dynamic_label_propagation/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/dynamic_label_propagation/text/imdb.yaml > log/transductive/dynamic_label_propagation/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/dynamic_label_propagation/vision/mnist.yaml > log/transductive/dynamic_label_propagation/vision/mnist.log 2>&1
```

## transductive/gat

```
mkdir -p log/transductive/gat/audio log/transductive/gat/graph log/transductive/gat/tabular log/transductive/gat/text log/transductive/gat/vision
python -m bench.main --config bench/configs/experiments/smoke/transductive/gat/audio/speechcommands.yaml > log/transductive/gat/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/gat/graph/cora.yaml > log/transductive/gat/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/gat/tabular/iris.yaml > log/transductive/gat/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/gat/text/imdb.yaml > log/transductive/gat/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/gat/vision/mnist.yaml > log/transductive/gat/vision/mnist.log 2>&1
```

## transductive/gcn

```
mkdir -p log/transductive/gcn/audio log/transductive/gcn/graph log/transductive/gcn/tabular log/transductive/gcn/text log/transductive/gcn/vision
python -m bench.main --config bench/configs/experiments/smoke/transductive/gcn/audio/speechcommands.yaml > log/transductive/gcn/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/gcn/graph/cora.yaml > log/transductive/gcn/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/gcn/tabular/iris.yaml > log/transductive/gcn/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/gcn/text/imdb.yaml > log/transductive/gcn/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/gcn/vision/mnist.yaml > log/transductive/gcn/vision/mnist.log 2>&1
```

## transductive/gcnii

```
mkdir -p log/transductive/gcnii/audio log/transductive/gcnii/graph log/transductive/gcnii/tabular log/transductive/gcnii/text log/transductive/gcnii/vision
python -m bench.main --config bench/configs/experiments/smoke/transductive/gcnii/audio/speechcommands.yaml > log/transductive/gcnii/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/gcnii/graph/cora.yaml > log/transductive/gcnii/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/gcnii/tabular/iris.yaml > log/transductive/gcnii/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/gcnii/text/imdb.yaml > log/transductive/gcnii/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/gcnii/vision/mnist.yaml > log/transductive/gcnii/vision/mnist.log 2>&1
```

## transductive/grafn

```
mkdir -p log/transductive/grafn/audio log/transductive/grafn/graph log/transductive/grafn/tabular log/transductive/grafn/text log/transductive/grafn/vision
python -m bench.main --config bench/configs/experiments/smoke/transductive/grafn/audio/speechcommands.yaml > log/transductive/grafn/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/grafn/graph/cora.yaml > log/transductive/grafn/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/grafn/tabular/iris.yaml > log/transductive/grafn/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/grafn/text/imdb.yaml > log/transductive/grafn/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/grafn/vision/mnist.yaml > log/transductive/grafn/vision/mnist.log 2>&1
```

## transductive/grand

```
mkdir -p log/transductive/grand/audio log/transductive/grand/graph log/transductive/grand/tabular log/transductive/grand/text log/transductive/grand/vision
python -m bench.main --config bench/configs/experiments/smoke/transductive/grand/audio/speechcommands.yaml > log/transductive/grand/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/grand/graph/cora.yaml > log/transductive/grand/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/grand/tabular/iris.yaml > log/transductive/grand/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/grand/text/imdb.yaml > log/transductive/grand/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/grand/vision/mnist.yaml > log/transductive/grand/vision/mnist.log 2>&1
```

## transductive/graph_mincuts

```
mkdir -p log/transductive/graph_mincuts/audio log/transductive/graph_mincuts/graph log/transductive/graph_mincuts/tabular log/transductive/graph_mincuts/text log/transductive/graph_mincuts/vision
python -m bench.main --config bench/configs/experiments/smoke/transductive/graph_mincuts/audio/speechcommands.yaml > log/transductive/graph_mincuts/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/graph_mincuts/graph/cora.yaml > log/transductive/graph_mincuts/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/graph_mincuts/graph/toy.yaml > log/transductive/graph_mincuts/graph/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/graph_mincuts/tabular/breast_cancer.yaml > log/transductive/graph_mincuts/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/graph_mincuts/text/imdb.yaml > log/transductive/graph_mincuts/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/graph_mincuts/vision/mnist.yaml > log/transductive/graph_mincuts/vision/mnist.log 2>&1
```

## transductive/graphhop

```
mkdir -p log/transductive/graphhop/audio log/transductive/graphhop/graph log/transductive/graphhop/tabular log/transductive/graphhop/text log/transductive/graphhop/vision
python -m bench.main --config bench/configs/experiments/smoke/transductive/graphhop/audio/speechcommands.yaml > log/transductive/graphhop/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/graphhop/graph/cora.yaml > log/transductive/graphhop/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/graphhop/tabular/iris.yaml > log/transductive/graphhop/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/graphhop/text/imdb.yaml > log/transductive/graphhop/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/graphhop/vision/mnist.yaml > log/transductive/graphhop/vision/mnist.log 2>&1
```

## transductive/graphsage

```
mkdir -p log/transductive/graphsage/audio log/transductive/graphsage/graph log/transductive/graphsage/tabular log/transductive/graphsage/text log/transductive/graphsage/vision
python -m bench.main --config bench/configs/experiments/smoke/transductive/graphsage/audio/speechcommands.yaml > log/transductive/graphsage/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/graphsage/graph/cora.yaml > log/transductive/graphsage/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/graphsage/tabular/iris.yaml > log/transductive/graphsage/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/graphsage/text/imdb.yaml > log/transductive/graphsage/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/graphsage/vision/mnist.yaml > log/transductive/graphsage/vision/mnist.log 2>&1
```

## transductive/h_gcn

```
mkdir -p log/transductive/h_gcn/audio log/transductive/h_gcn/graph log/transductive/h_gcn/tabular log/transductive/h_gcn/text log/transductive/h_gcn/vision
python -m bench.main --config bench/configs/experiments/smoke/transductive/h_gcn/audio/speechcommands.yaml > log/transductive/h_gcn/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/h_gcn/graph/cora.yaml > log/transductive/h_gcn/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/h_gcn/tabular/iris.yaml > log/transductive/h_gcn/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/h_gcn/text/imdb.yaml > log/transductive/h_gcn/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/h_gcn/vision/mnist.yaml > log/transductive/h_gcn/vision/mnist.log 2>&1
```

## transductive/label_propagation

```
mkdir -p log/transductive/label_propagation/audio log/transductive/label_propagation/graph log/transductive/label_propagation/tabular log/transductive/label_propagation/text log/transductive/label_propagation/vision
python -m bench.main --config bench/configs/experiments/smoke/transductive/label_propagation/audio/speechcommands.yaml > log/transductive/label_propagation/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/label_propagation/graph/cora.yaml > log/transductive/label_propagation/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/label_propagation/tabular/iris.yaml > log/transductive/label_propagation/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/label_propagation/text/imdb.yaml > log/transductive/label_propagation/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/label_propagation/vision/mnist.yaml > log/transductive/label_propagation/vision/mnist.log 2>&1
```

## transductive/label_spreading

```
mkdir -p log/transductive/label_spreading/audio log/transductive/label_spreading/graph log/transductive/label_spreading/tabular log/transductive/label_spreading/text log/transductive/label_spreading/vision
python -m bench.main --config bench/configs/experiments/smoke/transductive/label_spreading/audio/speechcommands.yaml > log/transductive/label_spreading/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/label_spreading/graph/cora.yaml > log/transductive/label_spreading/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/label_spreading/tabular/iris.yaml > log/transductive/label_spreading/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/label_spreading/text/imdb.yaml > log/transductive/label_spreading/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/label_spreading/vision/mnist.yaml > log/transductive/label_spreading/vision/mnist.log 2>&1
```

## transductive/laplace_learning

```
mkdir -p log/transductive/laplace_learning/audio log/transductive/laplace_learning/graph log/transductive/laplace_learning/tabular log/transductive/laplace_learning/text log/transductive/laplace_learning/vision
python -m bench.main --config bench/configs/experiments/smoke/transductive/laplace_learning/audio/speechcommands.yaml > log/transductive/laplace_learning/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/laplace_learning/graph/cora.yaml > log/transductive/laplace_learning/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/laplace_learning/tabular/iris.yaml > log/transductive/laplace_learning/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/laplace_learning/text/imdb.yaml > log/transductive/laplace_learning/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/laplace_learning/vision/mnist.yaml > log/transductive/laplace_learning/vision/mnist.log 2>&1
```

## transductive/lazy_random_walk

```
mkdir -p log/transductive/lazy_random_walk/audio log/transductive/lazy_random_walk/graph log/transductive/lazy_random_walk/tabular log/transductive/lazy_random_walk/text log/transductive/lazy_random_walk/vision
python -m bench.main --config bench/configs/experiments/smoke/transductive/lazy_random_walk/audio/speechcommands.yaml > log/transductive/lazy_random_walk/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/lazy_random_walk/graph/cora.yaml > log/transductive/lazy_random_walk/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/lazy_random_walk/graph/toy.yaml > log/transductive/lazy_random_walk/graph/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/lazy_random_walk/tabular/breast_cancer.yaml > log/transductive/lazy_random_walk/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/lazy_random_walk/text/imdb.yaml > log/transductive/lazy_random_walk/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/lazy_random_walk/vision/mnist.yaml > log/transductive/lazy_random_walk/vision/mnist.log 2>&1
```

## transductive/n_gcn

```
mkdir -p log/transductive/n_gcn/audio log/transductive/n_gcn/graph log/transductive/n_gcn/tabular log/transductive/n_gcn/text log/transductive/n_gcn/vision
python -m bench.main --config bench/configs/experiments/smoke/transductive/n_gcn/audio/speechcommands.yaml > log/transductive/n_gcn/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/n_gcn/graph/cora.yaml > log/transductive/n_gcn/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/n_gcn/tabular/iris.yaml > log/transductive/n_gcn/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/n_gcn/text/imdb.yaml > log/transductive/n_gcn/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/n_gcn/vision/mnist.yaml > log/transductive/n_gcn/vision/mnist.log 2>&1
```

## transductive/p_laplace_learning

```
mkdir -p log/transductive/p_laplace_learning/audio log/transductive/p_laplace_learning/graph log/transductive/p_laplace_learning/tabular log/transductive/p_laplace_learning/text log/transductive/p_laplace_learning/vision
python -m bench.main --config bench/configs/experiments/smoke/transductive/p_laplace_learning/audio/speechcommands.yaml > log/transductive/p_laplace_learning/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/p_laplace_learning/graph/cora.yaml > log/transductive/p_laplace_learning/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/p_laplace_learning/tabular/iris.yaml > log/transductive/p_laplace_learning/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/p_laplace_learning/text/imdb.yaml > log/transductive/p_laplace_learning/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/p_laplace_learning/vision/mnist.yaml > log/transductive/p_laplace_learning/vision/mnist.log 2>&1
```

## transductive/planetoid

```
mkdir -p log/transductive/planetoid/audio log/transductive/planetoid/graph log/transductive/planetoid/tabular log/transductive/planetoid/text log/transductive/planetoid/vision
python -m bench.main --config bench/configs/experiments/smoke/transductive/planetoid/audio/speechcommands.yaml > log/transductive/planetoid/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/planetoid/graph/cora.yaml > log/transductive/planetoid/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/planetoid/tabular/iris.yaml > log/transductive/planetoid/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/planetoid/text/imdb.yaml > log/transductive/planetoid/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/planetoid/vision/mnist.yaml > log/transductive/planetoid/vision/mnist.log 2>&1
```

## transductive/poisson_learning

```
mkdir -p log/transductive/poisson_learning/audio log/transductive/poisson_learning/graph log/transductive/poisson_learning/tabular log/transductive/poisson_learning/text log/transductive/poisson_learning/vision
python -m bench.main --config bench/configs/experiments/smoke/transductive/poisson_learning/audio/speechcommands.yaml > log/transductive/poisson_learning/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/poisson_learning/graph/cora.yaml > log/transductive/poisson_learning/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/poisson_learning/tabular/iris.yaml > log/transductive/poisson_learning/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/poisson_learning/text/imdb.yaml > log/transductive/poisson_learning/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/poisson_learning/vision/mnist.yaml > log/transductive/poisson_learning/vision/mnist.log 2>&1
```

## transductive/poisson_mbo

```
mkdir -p log/transductive/poisson_mbo/audio log/transductive/poisson_mbo/graph log/transductive/poisson_mbo/tabular log/transductive/poisson_mbo/text log/transductive/poisson_mbo/vision
python -m bench.main --config bench/configs/experiments/smoke/transductive/poisson_mbo/audio/speechcommands.yaml > log/transductive/poisson_mbo/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/poisson_mbo/graph/cora.yaml > log/transductive/poisson_mbo/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/poisson_mbo/tabular/iris.yaml > log/transductive/poisson_mbo/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/poisson_mbo/text/imdb.yaml > log/transductive/poisson_mbo/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/poisson_mbo/vision/mnist.yaml > log/transductive/poisson_mbo/vision/mnist.log 2>&1
```

## transductive/sgc

```
mkdir -p log/transductive/sgc/audio log/transductive/sgc/graph log/transductive/sgc/tabular log/transductive/sgc/text log/transductive/sgc/vision
python -m bench.main --config bench/configs/experiments/smoke/transductive/sgc/audio/speechcommands.yaml > log/transductive/sgc/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/sgc/graph/cora.yaml > log/transductive/sgc/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/sgc/tabular/iris.yaml > log/transductive/sgc/tabular/iris.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/sgc/text/imdb.yaml > log/transductive/sgc/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/sgc/vision/mnist.yaml > log/transductive/sgc/vision/mnist.log 2>&1
```

## transductive/tsvm

```
mkdir -p log/transductive/tsvm/audio log/transductive/tsvm/graph log/transductive/tsvm/tabular log/transductive/tsvm/text log/transductive/tsvm/vision
python -m bench.main --config bench/configs/experiments/smoke/transductive/tsvm/audio/speechcommands.yaml > log/transductive/tsvm/audio/speechcommands.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/tsvm/graph/cora.yaml > log/transductive/tsvm/graph/cora.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/tsvm/graph/toy.yaml > log/transductive/tsvm/graph/toy.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/tsvm/tabular/breast_cancer.yaml > log/transductive/tsvm/tabular/breast_cancer.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/tsvm/text/imdb.yaml > log/transductive/tsvm/text/imdb.log 2>&1
python -m bench.main --config bench/configs/experiments/smoke/transductive/tsvm/vision/mnist.yaml > log/transductive/tsvm/vision/mnist.log 2>&1
```
