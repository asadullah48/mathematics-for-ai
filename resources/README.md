# 📚 Resources

A curated set of external books, papers, courses, and video lectures for
learning the mathematics behind AI/ML - organized by topic, matched to
this repo's own modules so you can go straight from theory to a working
implementation.

**A note on this folder:** these are *links*, not hosted copies. Most of
the classics below (Murphy's *Probabilistic Machine Learning*, Gallier &
Quaintance, the `d2l.ai` appendix, ...) are free to *read* on their
authors' own sites, but redistributing copies of someone else's book is a
different thing from linking to it - so this index points to the
original, canonical source for each resource instead of mirroring PDFs
into this repo. (Inspired by, and several entries drawn from,
[dair-ai/Mathematics-for-ML](https://github.com/dair-ai/Mathematics-for-ML)
- credit to that list for the curation.)

## Table of Contents
- [Linear Algebra](#linear-algebra)
- [Calculus & Optimization](#calculus--optimization)
- [Probability & Statistics](#probability--statistics)
- [Full-Spectrum Books](#full-spectrum-books)
- [Papers](#papers)
- [Video Lectures & Courses](#video-lectures--courses)

---

## Linear Algebra

Pairs with [`src/math_utils/linear_algebra.py`](../src/math_utils/linear_algebra.py)
and [`notebooks/linear-algebra/`](../notebooks/linear-algebra/).

| Resource | Author(s) | Link |
|---|---|---|
| Linear Algebra Done Right (lectures + slides) | Sheldon Axler | https://linear.axler.net/LADRvideos.html |
| Linear Algebra | Khan Academy | https://www.khanacademy.org/math/linear-algebra |
| Multivariate Calculus (backprop-relevant: chain rule, Jacobian) | Imperial College London | https://www.youtube.com/playlist?list=PLiiljHvN6z193BBzS0Ln8NnqQmzimTW23 |
| Mathematics for Machine Learning - Linear Algebra | Imperial College London | https://www.youtube.com/playlist?list=PLiiljHvN6z1_o1ztXTKWPrShrMrBLo5P3 |

## Calculus & Optimization

Pairs with [`src/math_utils/calculus.py`](../src/math_utils/calculus.py),
[`notebooks/calculus/`](../notebooks/calculus/), and
[`scripts/optimization_routines.py`](../scripts/optimization_routines.py).

| Resource | Author(s) | Link |
|---|---|---|
| The Matrix Calculus You Need For Deep Learning | Terence Parr & Jeremy Howard | https://arxiv.org/abs/1802.01528 |
| Calculus (Precalculus -> Multivariate) | Khan Academy | https://www.khanacademy.org/math/calculus-home |

## Probability & Statistics

Pairs with [`src/math_utils/probability.py`](../src/math_utils/probability.py)
and [`src/math_utils/statistics.py`](../src/math_utils/statistics.py).

| Resource | Author(s) | Link |
|---|---|---|
| Probability Theory: The Logic of Science | E. T. Jaynes | https://bayes.wustl.edu/etj/prob/book.pdf |
| Statistics and Probability | Khan Academy | https://www.khanacademy.org/math/statistics-probability |
| Bayes Rules! An Introduction to Applied Bayesian Modeling | Johnson, Ott, Dogucu | https://www.bayesrulesbook.com/index.html |
| The Elements of Statistical Learning | Hastie, Tibshirani, Friedman | https://hastie.su.domains/ElemStatLearn/ |
| An Introduction to Statistical Learning | James, Witten, Hastie, Tibshirani | https://www.statlearning.com/ |

## Full-Spectrum Books

Cover several of the topics above in one place - good next step after the
`notebooks/basics/` introduction.

| Resource | Author(s) | Link |
|---|---|---|
| Mathematics for Machine Learning | Deisenroth, Faisal, Ong | https://mml-book.github.io |
| Probabilistic Machine Learning: An Introduction | Kevin Patrick Murphy | https://probml.github.io/pml-book/book1.html |
| Applied Math and Machine Learning Basics (Deep Learning book, Part I) | Goodfellow, Bengio, Courville | https://www.deeplearningbook.org/contents/part_basics.html |
| Mathematics for Deep Learning (D2L appendix) | Werness, Hu, et al. | https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/index.html |
| The Mathematical Engineering of Deep Learning | Liquet, Moka, Nazarathy | https://deeplearningmath.org |
| Algebra, Topology, Differential Calculus, and Optimization Theory For CS and ML | Gallier & Quaintance | https://www.cis.upenn.edu/~jean/math-deep.pdf |
| Information Theory, Inference and Learning Algorithms | David J. C. MacKay | https://www.inference.org.uk/itprnn/book.html |

## Papers

| Resource | Author(s) | Link |
|---|---|---|
| The Mathematics of AI | Gitta Kutyniok | https://arxiv.org/pdf/2203.08890.pdf |

## Video Lectures & Courses

| Resource | Author(s) | Link |
|---|---|---|
| CS229: Machine Learning | Anand Avati (Stanford) | https://www.youtube.com/playlist?list=PLoROMvodv4rNH7qL6-efu_q2_bPuy0adh |

---

**Want to go deeper on a specific method?** Every implementation in
[`src/math_utils/`](../src/math_utils/) has a docstring naming the concept
it implements - search this table for that concept's name to find the
matching theory.
