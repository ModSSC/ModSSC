# Paper methods, source links, datasets, and protocol notes

This page covers the methods registered in ModSSC. It combines:

- `src/modssc/inductive/registry.py` and `src/modssc/transductive/registry.py`
- paper/code notes in `docs/article_code/**`
- extracted dataset/protocol evidence from the local dashboard data
- current source/code pages or author repositories, when a reliable link could be found

Status conventions:

- "source link" means a paper, author, lab, or repository page associated with the method.
- "code page" means an author/lab page distributes code, but not necessarily as a GitHub repository.
- "not found/cited" means no reliable source/code link was found in the local archive or current web check.
- Protocol notes are compact. Rows marked "needs deeper extraction" have a paper archive entry but no complete `experiment_params.md` extraction yet.

## Benchmark comparison policy

The benchmark source is `bench/configs/best/` only. `bench/configs/experiments/` remains for smoke runs, documentation examples, and development templates; it is not a benchmark source unless a campaign explicitly promotes a config into `best`.

Benchmark ranking uses the mean `test.accuracy` over the five configured seeds. `macro_f1`, runtime, memory, and failures are reported as diagnostics, but they do not decide the winner. Pipeline selection inside a method uses `val.accuracy`; only the selected pipeline's `test.accuracy` is used for final reporting.

Comparisons are valid only on paired cells: same dataset, modality, regime, split contract, and seed set. Inductive vs transductive compares the best selected pipeline from each family on paired cells. Classical vs neural is classified by the effective pipeline and backend, not only by the method name. Poisson comparisons use `poisson_learning` and `poisson_mbo` against GNN-family methods on paired transductive cells. Cross-modal or intermodal transfer is a success only when the transferred method beats native methods on the target modality in `test.accuracy`; otherwise the result should report the regime where it is competitive or fails.

## Paper-fidelity status

Only `paper_matched` may be described as "implemented as in the paper." Current statuses are intentionally conservative:

- `paper_matched`: exact article protocol, backbone/model, splits, tuning, selection rule, and source validation are matched.
- `paper_approx`: the implemented algorithm follows the paper family and key controls, but at least one exact protocol element is standardized or not fully proven identical.
- `standardized_only`: valid ModSSC benchmark method or control, but not an article-reproduction claim.
- `not_claimable`: source/protocol extraction is incomplete or evidence is insufficient for a paper-level claim.

No current method is marked `paper_matched`. `paper_approx` and `standardized_only` methods may be benchmarked, but results must not be presented as exact paper reproductions.

| Method | Family | Status | Comparison class |
|---|---|---|---|
| `supervised` | control | `standardized_only` | baseline |
| `pseudo_label` | pseudo_label | `paper_approx` | inductive_classic_or_neural_by_backend |
| `self_training` | self_training | `not_claimable` | inductive_classic |
| `setred` | self_training_editing | `not_claimable` | inductive_classic_or_neural_by_backend |
| `pi_model` | consistency_regularization | `paper_approx` | inductive_neural |
| `fixmatch` | fixmatch_thresholding | `paper_approx` | inductive_neural |
| `comatch` | contrastive_graph_regularization | `paper_approx` | inductive_neural |
| `defixmatch` | debiasing_fixmatch | `paper_approx` | inductive_neural |
| `daso` | imbalanced_ssl | `paper_approx` | inductive_neural |
| `adsh` | adaptive_thresholding | `paper_approx` | inductive_neural |
| `flexmatch` | adaptive_thresholding | `paper_approx` | inductive_neural |
| `adamatch` | domain_adaptation_ssl | `paper_approx` | inductive_neural |
| `free_match` | adaptive_thresholding | `paper_approx` | inductive_neural |
| `softmatch` | adaptive_thresholding | `paper_approx` | inductive_neural |
| `mixmatch` | mixup_consistency | `paper_approx` | inductive_neural |
| `simclr_v2` | self_supervised_transfer | `standardized_only` | inductive_neural |
| `mean_teacher` | teacher_student_consistency | `paper_approx` | inductive_neural |
| `meta_pseudo_labels` | teacher_student_pseudo_labeling | `paper_approx` | inductive_neural |
| `temporal_ensembling` | consistency_regularization | `paper_approx` | inductive_neural |
| `uda` | augmentation_consistency | `paper_approx` | inductive_neural |
| `vat` | adversarial_consistency | `paper_approx` | inductive_neural |
| `noisy_student` | teacher_student_pseudo_labeling | `standardized_only` | inductive_neural |
| `co_training` | multi_view_co_training | `not_claimable` | inductive_classic_or_neural_by_backend |
| `democratic_co_learning` | ensemble_co_learning | `not_claimable` | inductive_classic_or_neural_by_backend |
| `deep_co_training` | deep_co_training | `not_claimable` | inductive_neural |
| `tri_training` | ensemble_self_training | `paper_approx` | inductive_classic_or_neural_by_backend |
| `trinet` | deep_tri_training | `not_claimable` | inductive_neural |
| `s4vm` | safe_margin_ssl | `paper_approx` | inductive_classic_or_neural_by_backend |
| `label_propagation` | graph_diffusion | `paper_approx` | transductive_classic |
| `label_spreading` | graph_diffusion | `paper_approx` | transductive_classic |
| `laplace_learning` | graph_pde | `not_claimable` | transductive_classic |
| `lazy_random_walk` | random_walk | `not_claimable` | transductive_classic |
| `dynamic_label_propagation` | graph_diffusion | `paper_approx` | transductive_classic |
| `graph_mincuts` | graph_cut | `not_claimable` | transductive_classic |
| `tsvm` | margin_ssl | `paper_approx` | transductive_classic |
| `poisson_learning` | poisson_graph_pde | `paper_approx` | transductive_classic |
| `poisson_mbo` | poisson_graph_pde | `paper_approx` | transductive_classic |
| `p_laplace_learning` | graph_pde | `paper_approx` | transductive_classic |
| `chebnet` | gnn_spectral | `paper_approx` | transductive_neural |
| `planetoid` | gnn_embedding | `paper_approx` | transductive_neural |
| `gcn` | gnn_message_passing | `paper_approx` | transductive_neural |
| `graphsage` | gnn_message_passing | `paper_approx` | transductive_neural |
| `gat` | gnn_attention | `paper_approx` | transductive_neural |
| `sgc` | gnn_linearized | `paper_approx` | transductive_neural |
| `appnp` | gnn_propagation | `paper_approx` | transductive_neural |
| `h_gcn` | gnn_hierarchical | `paper_approx` | transductive_neural |
| `n_gcn` | gnn_multiscale | `paper_approx` | transductive_neural |
| `graphhop` | label_aggregation_graph | `paper_approx` | transductive_neural |
| `grafn` | few_label_graph | `paper_approx` | transductive_neural |
| `gcnii` | deep_gnn | `paper_approx` | transductive_neural |
| `grand` | random_propagation_gnn | `paper_approx` | transductive_neural |

## PDF index

Every paper-backed method now has a `paper_pdf` value in its `MethodInfo`. Public direct PDF URLs are used when available; otherwise the link points to the archived PDF under `docs/article_code/**`. The `supervised` entry is a ModSSC control baseline, not a paper-backed method.

### Inductive PDFs

| Method | PDF |
|---|---|
| `adamatch` | [PDF](https://arxiv.org/pdf/2106.04732) |
| `adsh` | [PDF](https://proceedings.mlr.press/v162/guo22e/guo22e.pdf) |
| `co_training` | [PDF](https://www.cs.cmu.edu/~avrim/Papers/co-training.pdf) |
| `comatch` | [PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_CoMatch_Semi-Supervised_Learning_With_Contrastive_Graph_Regularization_ICCV_2021_paper.pdf) |
| `daso` | [PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Oh_DASO_Distribution-Aware_Semantics-Oriented_Pseudo-Label_for_Imbalanced_Semi-Supervised_Learning_CVPR_2022_paper.pdf) |
| `deep_co_training` | [PDF](https://arxiv.org/pdf/1803.05984) |
| `defixmatch` | [PDF](https://openreview.net/pdf?id=TN9gQ4x0Ep3) |
| `democratic_co_learning` | local archive: `docs/article_code/inductive/2004-Democratic colearning/21-2004-Democratic colearning.pdf` |
| `fixmatch` | [PDF](https://arxiv.org/pdf/2001.07685) |
| `flexmatch` | [PDF](https://arxiv.org/pdf/2110.08263) |
| `free_match` | [PDF](https://arxiv.org/pdf/2205.07246) |
| `mean_teacher` | [PDF](https://arxiv.org/pdf/1703.01780) |
| `meta_pseudo_labels` | [PDF](https://arxiv.org/pdf/2003.10580) |
| `mixmatch` | [PDF](https://arxiv.org/pdf/1905.02249) |
| `noisy_student` | [PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xie_Self-Training_With_Noisy_Student_Improves_ImageNet_Classification_CVPR_2020_paper.pdf) |
| `pi_model` | [PDF](https://arxiv.org/pdf/1610.02242) |
| `pseudo_label` | local archive: `docs/article_code/inductive/2013-Pseudo Label/8_pseudo_label.pdf` |
| `s4vm` | [PDF](https://icml.cc/Conferences/2011/papers/548_icmlpaper.pdf) |
| `self_training` | local archive: `docs/article_code/inductive/1995-Self Training/4-1995-Unsupervised word sense disambiguation rivaling supervised methods.pdf` |
| `setred` | local archive: `docs/article_code/inductive/2005-SETRED selftraining with editing/9-2005-SETRED selftraining with editing.pdf` |
| `simclr_v2` | [PDF](https://arxiv.org/pdf/2006.10029) |
| `softmatch` | [PDF](https://openreview.net/pdf?id=ymt1zQXBDiF) |
| `supervised` | n/a: ModSSC baseline, not a paper-backed method |
| `temporal_ensembling` | [PDF](https://arxiv.org/pdf/1610.02242) |
| `tri_training` | [PDF](https://www.lamda.nju.edu.cn/publication/tkde05.pdf) |
| `trinet` | [PDF](https://www.lamda.nju.edu.cn/publication/ijcai18trinet.pdf) |
| `uda` | [PDF](https://arxiv.org/pdf/1904.12848) |
| `vat` | [PDF](https://arxiv.org/pdf/1704.03976) |

### Transductive PDFs

| Method | PDF |
|---|---|
| `appnp` | [PDF](https://arxiv.org/pdf/1810.05997) |
| `chebnet` | [PDF](https://arxiv.org/pdf/1606.09375) |
| `dynamic_label_propagation` | [PDF](https://openaccess.thecvf.com/content_iccv_2013/papers/Wang_Dynamic_Label_Propagation_2013_ICCV_paper.pdf) |
| `gat` | [PDF](https://arxiv.org/pdf/1710.10903) |
| `gcn` | [PDF](https://arxiv.org/pdf/1609.02907) |
| `gcnii` | [PDF](https://arxiv.org/pdf/2007.02133) |
| `grafn` | [PDF](https://arxiv.org/pdf/2204.01303) |
| `grand` | [PDF](https://arxiv.org/pdf/2005.11079) |
| `graph_mincuts` | local archive: `docs/article_code/transductive/2001-Graph Mincuts/Learning from Labeled and Unlabeled Data using Graph Mincuts.pdf` |
| `graphhop` | [PDF](https://arxiv.org/pdf/2101.02326) |
| `graphsage` | [PDF](https://arxiv.org/pdf/1706.02216) |
| `h_gcn` | [PDF](https://arxiv.org/pdf/1902.06667) |
| `label_propagation` | local archive: `docs/article_code/transductive/2002-Label Propagation/Learning from Labeled and Unlabeled Data with Label Propagation.pdf` |
| `label_spreading` | local archive: `docs/article_code/transductive/2004-Label Spreading/Learning with Local and Global Consistency.pdf` |
| `laplace_learning` | local archive: `docs/article_code/transductive/2003-Laplace Learning/Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions.pdf` |
| `lazy_random_walk` | local archive: `docs/article_code/transductive/2004-Lazy Random Walk/Learning from labeled and labeled data using Random Walks.pdf` |
| `n_gcn` | [PDF](https://proceedings.mlr.press/v115/abu-el-haija20a/abu-el-haija20a.pdf) |
| `p_laplace_learning` | [PDF](https://arxiv.org/pdf/1901.05031) |
| `planetoid` | [PDF](https://arxiv.org/pdf/1603.08861) |
| `poisson_learning` | [PDF](https://proceedings.mlr.press/v119/calder20a/calder20a.pdf) |
| `poisson_mbo` | [PDF](https://proceedings.mlr.press/v119/calder20a/calder20a.pdf) |
| `sgc` | [PDF](https://arxiv.org/pdf/1902.07153) |
| `tsvm` | [PDF](https://www.cs.cornell.edu/people/tj/publications/joachims_99a.pdf) |

## Inductive methods

| Method | Source link | Paper datasets | Paper protocol and params |
|---|---|---|---|
| `supervised` | none; ModSSC baseline | task-dependent | Control baseline, not tied to a single SSL paper. Use the same split, backbone, augmentation, optimizer, and budget as the SSL method being compared. |
| `pseudo_label` | not found/cited | MNIST; local data also maps CIFAR-10, CIFAR-100, STL-10, SVHN through later baselines | ICML workshop paper from 2013. MNIST uses 100, 600, 1000, 3000 labels; 1000 validation samples; remaining train samples as unlabeled; 10 random splits, 30 splits for 100 labels. One hidden layer MLP, 5000 hidden units, ReLU hidden, sigmoid output; SGD plus dropout; initial LR 1.5; minibatches of 32 labeled and 256 unlabeled. Alpha schedule uses `alpha_f=3`, `T1=100`, `T2=600` without DAE pretraining and `T1=200`, `T2=800` with DAE pretraining. |
| `self_training` | not found/cited | breast_cancer; citation benchmarks in local data for later variants | Classic self-training baseline. Paper archive exists but exact hyperparameter extraction is incomplete; compare only with explicit split/classifier settings from the selected study. |
| `setred` | not found/cited | breast_cancer | Self-training with editing. Archive indicates repeated evaluation/fold protocols, but the exact parameter table still needs deeper extraction before claiming matched paper settings. |
| `pi_model` | [s-laine/tempens](https://github.com/s-laine/tempens) | CIFAR-10, CIFAR-100, MNIST, STL-10, SVHN | Shares the linked source with Temporal Ensembling. SVHN uses 500 labels; CIFAR-10 uses 4000 labels; CIFAR-100 can use Tiny Images as extra unlabeled data. Local extraction: `wmax=100` for Pi-model, `wmax=300` for CIFAR-100 plus Tiny Images; linked source notes Theano/Lasagne and dataset-specific augmentation/ZCA settings. |
| `fixmatch` | [google-research/fixmatch](https://github.com/google-research/fixmatch) | AG News, CIFAR-10, CIFAR-100, DBpedia, IMDb, MNIST, STL-10, SVHN; ImageNet in the paper | Main image protocol: WRN/ResNet backbones; labeled batch `B=64`, unlabeled ratio `mu=7`, confidence threshold `tau=0.95`, unsupervised weight `lambda_u=1`, SGD momentum 0.9 with Nesterov, LR 0.03, cosine schedule. Weight decay is 0.0005 for most image datasets and 0.001 for CIFAR-100. ImageNet uses lower threshold (`tau=0.7`) and larger unsupervised weight (`lambda_u=10`). |
| `comatch` | [salesforce/CoMatch](https://github.com/salesforce/CoMatch) | CIFAR-10, STL-10; ImageNet in the paper | CIFAR-10 uses WRN-28-2; STL-10 uses ResNet-18; ImageNet uses ResNet-50. SGD momentum 0.9; CIFAR/STL weight decay 0.0005 and LR 0.03 cosine. Common SSL settings include `lambda_cls=1`, `tau=0.95`, `mu=7`, `B=64`, graph smoothing `alpha=0.9`, memory queue `K=2560`, contrastive temperature around 0.2, pseudo-label temperature around 0.8, and dataset-specific `lambda_ctr`. |
| `defixmatch` | [HugoSchmutz/DeFixmatch](https://github.com/HugoSchmutz/DeFixmatch) | CIFAR-10, CIFAR-100, MNIST, STL-10, SVHN | ICLR 2023 paper: Don't fear the unlabelled: safe semi-supervised learning via debiasing. The code implements the debiased FixMatch variant reported as DeFixmatch; use paper-specific MCAR/debiasing settings before comparing to standard FixMatch. |
| `daso` | [ytaek-oh/daso](https://github.com/ytaek-oh/daso) | CIFAR-10-LT, CIFAR-100-LT, STL-10-LT; Semi-Aves in the paper | Imbalanced SSL protocol built on FixMatch/ReMixMatch-style backbones. CIFAR/STL experiments use 250k iterations; Semi-Aves uses 90 epochs. Label/unlabeled imbalance ratios are swept (`gamma_l`, `gamma_u`); distribution-aware alignment uses a memory queue/prototype temperature and alignment weight selected by paper ablations. |
| `adsh` | [LAMDA ADSH code page](https://www.lamda.nju.edu.cn/code_ADSH.ashx) | CIFAR-10, STL-10, SVHN | Adaptive per-class thresholding for distribution-aware SSL. Protocol sensitivity is high: paper protocol uses CNN/ResNet/WRN-style settings and random augmentation, while naive scratch ResNet baselines are not directly comparable. |
| `flexmatch` | [TorchSSL/TorchSSL](https://github.com/TorchSSL/TorchSSL) | AG News, CIFAR-10, CIFAR-100, DBpedia, IMDb, STL-10, SVHN; ImageNet in the paper | FixMatch-style setup with class-adaptive thresholds. Common paper settings: `mu=7`, `tau=0.95` for CIFAR/SVHN/STL, `tau=0.7` for ImageNet, labeled batch 64 except ImageNet 128, LR 0.03, SGD momentum 0.9, EMA 0.999, `lambda_u=1`, WRN-28-2/28-8/37-2 or ResNet-50, weight decay 5e-4 except CIFAR-100 1e-3 and ImageNet 3e-4. |
| `adamatch` | [google-research/adamatch](https://github.com/google-research/adamatch) | CIFAR-10, MNIST, SVHN; Digit-Five and DomainNet in the paper | ICLR 2022 paper: AdaMatch: A Unified Approach to Semi-Supervised Learning and Domain Adaptation. Digit/Five and DomainNet protocols use ResNetV2-101 for 224px images, WRN-34-2 for 64px, WRN-28-2 for 32px, LR 0.03 with cosine decay, and confidence threshold around 0.9. Pretrained-domain runs often set weight decay to 0. |
| `free_match` | [microsoft/Semi-supervised-learning](https://github.com/microsoft/Semi-supervised-learning) | CIFAR-10, CIFAR-100, STL-10, SVHN; ImageNet in the paper | Self-adaptive confidence thresholding plus fairness regularization. Uses WRN-28-2/28-8/37-2 or ResNet-50, SGD momentum 0.9, LR 0.03 cosine, EMA 0.999, labeled batch 64 except ImageNet 128, and `lambda_u=1`. Fairness weight is 0.01 for very-low-label settings and 0.05 otherwise; SVHN uses warmup and a bounded adaptive threshold range. |
| `softmatch` | [Hhhhhhao/SoftMatch](https://github.com/Hhhhhhao/SoftMatch) | AG News, CIFAR-10, CIFAR-100, DBpedia, IMDb, STL-10, SVHN; ImageNet/text settings in the paper | ICLR 2023 paper: SoftMatch: Addressing the Quantity-Quality Tradeoff in Semi-supervised Learning. The repository is a redirect/explanation repo and points to USB/TorchSSL code entries. Uses WRN/ResNet backbones, SGD momentum 0.9, LR 0.03 cosine, EMA, labeled batch 64 and unlabeled ratio 7 for image benchmarks. |
| `mixmatch` | [google-research/mixmatch](https://github.com/google-research/mixmatch) | CIFAR-10, CIFAR-100, STL-10, SVHN | WRN-28 backbones with MixUp, label guessing, sharpening, and consistency loss. Local extraction records weight decay 0.0004, STL-10 with 5000 labeled samples plus unlabeled folds, long training, and dataset-specific augmentation. |
| `simclr_v2` | [google-research/simclr](https://github.com/google-research/simclr) | ImageNet in the paper; CIFAR-10 in local data | Semi-supervised transfer from large self-supervised ResNet models. Paper protocol pretrains large ResNet variants, fine-tunes on 1%/10% ImageNet labels, and distills to smaller students; compare using the exact pretrained checkpoint/backbone because results are not comparable to scratch supervised training. |
| `mean_teacher` | [CuriousAI/mean-teacher](https://github.com/CuriousAI/mean-teacher) | CIFAR-10, CIFAR-100, STL-10, SVHN; ImageNet in the paper | Student/teacher EMA consistency. Paper extraction covers SVHN/CIFAR/ImageNet settings, LeakyReLU conv activations, dataset-specific preprocessing/augmentation, and EMA teacher updates. Local data marks long training and VAE/embedding evidence in some baselines. |
| `meta_pseudo_labels` | [google-research/google-research/meta_pseudo_labels](https://github.com/google-research/google-research/tree/master/meta_pseudo_labels) | CIFAR-10, CIFAR-100, STL-10, SVHN; ImageNet and large unlabeled sources in the paper | Teacher optimized by student feedback. Common settings include WRN-28-2/ResNet-50, Nesterov momentum 0.9, cosine LR, up to 1M steps for CIFAR/SVHN and 0.5M for ImageNet, tuning over short 50k-step trials, LARS fine-tuning with LR 0.001 and batch 4096, label smoothing 0.1, and dataset-specific weight decay. |
| `temporal_ensembling` | [s-laine/tempens](https://github.com/s-laine/tempens) | CIFAR-10, CIFAR-100, SVHN | Shares the linked source with Pi Model. SVHN uses 500 labels; CIFAR-10 uses 4000 labels; CIFAR-100 can use Tiny Images. Local extraction: temporal ensemble EMA `alpha=0.6`, `wmax=30`, with dataset-specific augmentation/ZCA and Theano/Lasagne versions pinned in the repo README. |
| `uda` | [google-research/uda](https://github.com/google-research/uda) | AG News, CIFAR-10, CIFAR-100, CiteSeer, DBpedia, IMDb, STL-10, SVHN; ImageNet/text datasets in the paper | Consistency between original and strongly augmented samples. Image settings use Nesterov SGD 0.9, cosine LR, labeled/unlabeled batches such as 64/448 on CIFAR/SVHN, and confidence filtering. ImageNet uses much larger supervised/unsupervised batches and a larger unsupervised weight in 10% label settings. Text settings use BERT-base with LR in {1e-5, 2e-5, 5e-5}, batch sizes around 32/128, dropout 0.1, and back-translation. |
| `vat` | [takerum/vat](https://github.com/takerum/vat) | CIFAR-10, CIFAR-100, CiteSeer, MNIST, STL-10, SVHN | Virtual adversarial regularization. Paper extraction records MNIST/CIFAR/SVHN protocols, `alpha=1`, supervised minibatch 64, VAT minibatch 256, Adam LR around 0.003 for MNIST and 0.001 for CNN experiments, batch norm, and LeakyReLU slope 0.1. |
| `noisy_student` | [google-research/noisystudent](https://github.com/google-research/noisystudent) | ImageNet and JFT-style unlabeled data in the paper; local data maps CIFAR-10, MNIST, SVHN for local baselines | Teacher-student self-training with noise. Paper protocol uses EfficientNet teachers/students, RandAugment, dropout around 0.5, stochastic depth survival around 0.8, large unlabeled-to-labeled batch ratios, iterative pseudo-labeling, and short final fine-tuning. |
| `co_training` | not found/cited | CIFAR-10, CIFAR-100, CiteSeer, Cora, PubMed, SVHN in local data; original paper uses two-view classification | Original setting assumes two sufficient and redundant views. Local extraction exists but mixes original and later benchmark mappings; do not compare unless the two-view feature split and base learner are specified. |
| `democratic_co_learning` | not found/cited | Adult | Multiple learners vote with confidence estimates. Archive exists, but exact experimental parameters are not fully extracted. |
| `deep_co_training` | not found/cited | CIFAR-10, CIFAR-100, SVHN | Two deep classifiers are regularized with adversarial examples and view disagreement. Local data marks ResNet/VAE/embedding evidence and long training; no reliable source repository was found. |
| `tri_training` | not found/cited | UCI-style tabular datasets in paper archive | Local extraction: 12 UCI datasets, 3 classifiers, 4 unlabeled rates, repeated over 144 plotted settings. Uses three classifiers and pseudo-label agreement between two models to train the third. |
| `trinet` | [LAMDA Tri-net code page](https://www.lamda.nju.edu.cn/code_Tri-net.ashx) | CIFAR-10, MNIST, SVHN | IJCAI 2018 paper and LAMDA PyTorch code page. Deep tri-training variant with output smearing, diversity augmentation, and pseudo-label editing; exact table-level hyperparameters still need deeper extraction from the paper archive before matched-setting benchmarking. |
| `s4vm` | [LAMDA S4VM code page](https://www.lamda.nju.edu.cn/code_S4VM.ashx) | Adult, MNIST | Safe semi-supervised SVM. LAMDA provides a code link for S4VM. Local extraction for the later multivariate-measure study records a linear SVM with `C=1`; compare with the selected safety-constraint and split protocol because paper variants differ. |

## Transductive methods

| Method | Source link | Paper datasets | Paper protocol and params |
|---|---|---|---|
| `label_propagation` | not found/cited | CIFAR-10, CiteSeer, Cora, MNIST, PubMed, SVHN in local data | Classic graph propagation baseline. Paper protocols depend on the graph construction; match the same affinity/kernel, label rate, and class-mass normalization setting. |
| `label_spreading` | not found/cited | MNIST/USPS-style digit data | Local extraction: USPS digits 1-4 with 3874 samples, `alpha=0.99`, RBF width 1.25 for harmonic/affinity experiments, kNN with `k=1`, and 100 trials. Also includes two-moons and text/web classification demonstrations. |
| `laplace_learning` | not found/cited | CIFAR-10 in local data; synthetic/digit/text tasks in archive | Harmonic/Laplace graph SSL. Matched settings hinge on graph weights, class priors, and class-mass normalization; exact paper parameter extraction is incomplete. |
| `lazy_random_walk` | not found/cited | CIFAR-10 in local data | Random-walk SSL baseline. Archive exists, but exact graph and walk parameters still need deeper extraction before claiming matched paper settings. |
| `dynamic_label_propagation` | not found/cited | MNIST | Local extraction: MNIST uses 60k train and 10k test with 1% (600) and 5% (3000) labeled train samples plus 10k test samples. Object recognition settings use SIFT descriptors, 16x16 patches, stride 8, k-means codebook size 2048, chi-square distance, and 5%/10% labels. Sensitivity sweeps use `alpha` in [0.01, 0.1] with `lambda=0.1`, and `lambda` in [0.01, 1] with `alpha=0.05`. |
| `graph_mincuts` | not found/cited | archive-only; no reliable dataset extraction | Binary graph mincut SSL. Paper archive is present, but dataset/protocol OCR is incomplete. Do not claim matched paper settings until graph construction, terminal weights, and split details are extracted. |
| `tsvm` | [SVMlight](https://www.cs.cornell.edu/people/tj/svm_light/) | CiteSeer, Cora, MNIST, PubMed in local data; Reuters-21578 in local extraction | The SVMlight link includes approximate training for large transductive SVMs. Local extraction records Reuters-21578 ModApte split with 9603 train and 3299 test samples. Use paper-specific SVM kernel/C settings and transductive unlabeled pool definition. |
| `poisson_learning` | [jwcalder/GraphLearning](https://github.com/jwcalder/GraphLearning) | CIFAR-10, FashionMNIST, MNIST, WebKB | ICML 2020 paper. Use the GraphLearning link to inspect Poisson Learning settings. Matched runs require the selected kNN graph, embedding/metric, weighting kernel, label rate, and dataset preprocessing. |
| `poisson_mbo` | [jwcalder/GraphLearning](https://github.com/jwcalder/GraphLearning) | CIFAR-10, FashionMNIST, MNIST, WebKB | Poisson MBO is Algorithm 2 in the Poisson Learning ICML 2020 paper, not a separate arXiv paper. Use the same paper/PDF and GraphLearning source link. |
| `p_laplace_learning` | [mauriciofloresML/Laplacian_Lp_Graph_SSL](https://github.com/mauriciofloresML/Laplacian_Lp_Graph_SSL) | EMNIST, FashionMNIST, MNIST | Use the linked source to inspect experiment settings. Match the paper's selected `p`, graph kernel, label rate, and convergence tolerance. |
| `chebnet` | [mdeff/cnn_graph](https://github.com/mdeff/cnn_graph) | CiteSeer, Cora, MNIST, PubMed | Spectral graph CNN with Chebyshev filters. Compare with the original graph construction and polynomial order/filter settings; local data marks full-graph transductive settings. |
| `planetoid` | [kimiyoung/planetoid](https://github.com/kimiyoung/planetoid) | CiteSeer, Cora, PubMed; DIEL/NELL in repo data notes | Linked source provides transductive and inductive demos. Standard citation protocol uses sparse features, graph adjacency, fixed train/validation/test masks, and graph/context embedding objectives. |
| `gcn` | [tkipf/gcn](https://github.com/tkipf/gcn) | CiteSeer, Cora, PubMed | Standard citation graph protocol: 20 labels per class, 500 validation nodes, 1000 test nodes; two-layer GCN, hidden size 16, dropout 0.5, L2 5e-4, Adam LR 0.01, early stopping. |
| `graphsage` | [williamleif/GraphSAGE](https://github.com/williamleif/GraphSAGE) | Web of Science citation graph, Reddit, PPI in paper | Inductive graph representation protocol. Paper uses sampled multi-hop neighborhoods, citation train/test by year, Reddit/PPI inductive splits, and TensorFlow/Adam training. Exact sampler fanouts and aggregator settings must match the selected table. |
| `gat` | [PetarV-/GAT](https://github.com/PetarV-/GAT) | CiteSeer, Cora, PubMed; PPI in paper | Citation protocol follows GCN-style splits; typical paper settings use 8 attention heads, 8 hidden units per head on Cora/CiteSeer, dropout 0.6, L2 5e-4, Adam LR 0.005, ELU activations, and inductive PPI evaluation. |
| `sgc` | [Tiiiger/SGC](https://github.com/Tiiiger/SGC) | CiteSeer, Cora, PubMed | Simplifies GCN by removing nonlinearities and collapsing propagation before a linear classifier. Match fixed citation splits, tuned propagation order `K`, and the paper's regularization/LR settings. |
| `appnp` | [gasteigerjo/ppnp](https://github.com/gasteigerjo/ppnp) | CiteSeer, Cora, PubMed | Personalized PageRank propagation after prediction. Paper protocol uses repeated random splits/initializations, same architecture budget across datasets, dropout/L2/LR tuning, bootstrap confidence intervals, and paired significance testing. |
| `h_gcn` | [CRIPAC-DIG/H-GCN](https://github.com/CRIPAC-DIG/H-GCN) | CiteSeer, Cora, PubMed | Hierarchical GCN for semi-supervised node classification. Use the linked datasets/scripts to inspect settings; local data marks full-graph transductive, very-low-label, CNN/GCN-style settings. |
| `n_gcn` | [samihaija/mixhop](https://github.com/samihaija/mixhop) | CiteSeer, Cora, PubMed; PPI in paper | Multi-scale GCN. Local extraction: citation splits use 20 labels/class, 500 validation, 1000 test; PPI uses 20 train graphs, 2 validation, 2 test. TensorFlow experiments use two-layer GCN/SAGE modules, hidden size 16, dropout 0.5, L2 1e-5, Adam LR 0.01, 600 steps, 20 random initializations, and sweeps over scale/radius and classifier variant. |
| `graphhop` | [TianXieUSC/GraphHop](https://github.com/TianXieUSC/GraphHop) | CiteSeer, Cora, PubMed; protein multilabel tasks in paper | Iterative label aggregation/update with extremely low label rates, including one-label-per-class settings. Match hop count, label update schedule, and graph preprocessing from the linked source. |
| `grafn` | [Junseok0207/GraFN](https://github.com/Junseok0207/GraFN) | CiteSeer, Cora, PubMed; Amazon Computers/Photos in paper | Few-label node classification with non-parametric distribution assignment. Paper settings sample support nodes per class from labeled nodes, use cosine similarity with temperature, graph augmentations, and repeated few-label splits. |
| `gcnii` | [chennnM/GCNII](https://github.com/chennnM/GCNII) | Cora, CiteSeer, PubMed, Chameleon, Cornell, Texas, Wisconsin, PPI, ogbn-arxiv | Deep GCN with initial residual and identity mapping. Repo provides `semi.sh`, `full.sh`, and `ppi.sh`; match the script for the target table because depth and dataset-specific hyperparameters vary. |
| `grand` | [THUDM/GRAND](https://github.com/THUDM/GRAND) | Cora, CiteSeer, PubMed-style citation graphs in repo/paper | NeurIPS 2020 paper. Graph Random Neural Network with random propagation and consistency regularization. Use linked scripts for propagation step count, dropout/noise, consistency loss weight, and split protocol. |
