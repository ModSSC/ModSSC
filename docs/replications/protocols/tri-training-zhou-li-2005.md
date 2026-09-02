# Tri-Training — Zhou and Li (2005)

## Primary sources

- Zhou and Li, *Tri-Training: Exploiting Unlabeled Data Using Three
  Classifiers*, [paper](https://www.lamda.nju.edu.cn/publication/tkde05.pdf).
- Authors' [code page](https://www.lamda.nju.edu.cn/code_TriTrain.ashx) and
  released `TriTrain.java` archive. The reviewed Java file has SHA-256
  `0f3497f93190138e9ea93061dd72502f4134b977c4d7670b0898b94613433286`.

## Registered protocol

The two active cards register the Vote and WDBC J4.8 rows of Table III at the
80-percent unlabeled rate, each with three article repetitions. They preserve
the 25-percent test holdout, fresh labeled/unlabeled partitions, three
bootstrapped classifiers, the published error-update inequalities, and the
released executable's unpruned J48 and probability-average prediction rule.

The article does not publish its exact test indices, random stream, or complete
classifier version/options. ModSSC uses recorded deterministic partitions and
a native NumPy reconstruction; it never executes the Weka archive.

## Claim boundary

Both cards are capped at `paper_approx`. Fresh complete assessments are
required, and this page contains no run result or verdict.
