# On the use of associative memory in Hopfield networks designed to solve propositional satisfiability problems
This repository contains the code used for the paper <b>"On the use of associative memory in Hopfield networks designed to solve propositional satisfiability problems"</b> ([arXiv](https://arxiv.org/abs/2307.16807)). This paper explores the process of translating the satisfiability problem in propositional logic (SAT) into Hopfield network weights and subsequently utilizing these weights with the Self-Optimization (SO) model to obtain the problem's solution.

## Content:

* **START.py** - Main file that runs the simulations and generates figures 1-4 in the paper. It sets the style for the figures using the file *ieee_conf.mplstyle*.
* **makeplots.py** - Calls for *SO_for_SAT.py* and *borders.py*, and contains various plot functions to generate figures 1-4.
* **SO_for_SAT.py** - Calls for *hebbclean.F90* when 'F=1'. Contains functions to generate the clauses for the Liars problem and map coloring problem, and all the SO simulation functions from [SO-scaled-up](https://github.com/nata-web/SO-scaled-up/tree/main) slightly modified for this work. 
* **borders.py** - Contains functions to compute a set of shapes for the map of Checkerboard, the map of South America from the [CShapes 2.0 Dataset](https://icr.ethz.ch/publications/cshapes-2/) used in the article, and a set of shapes for Japan from the [GADM 4.1 Dataset](https://gadm.org/download_country.html).
* **hebbclean.F90** - Contains the FORTRAN routine.
* **ieee_conf.mplstyle** - Style sheet for matplotlib.
* **Map coloring example.pdf** - Contains an example of converting the SAT problem of map coloring (for 4 countries and 2 colors) to the weights of the Hopfield network.
  
## Additional needed files
The shapes comprising the map of South America (used in Sec. IV-B in the article) were obtained from the [CShapes 2.0 Dataset](https://icr.ethz.ch/publications/cshapes-2/). In addition, we also imported the map of Japan in this code (not present in the paper), and you can find the shapes for that the [GADM 4.1 Dataset](https://gadm.org/download_country.html), level-1. You will need to download these both datasets to run the code. 

## To run the code from Python with FORTRAN:
Make sure your system has gfortran and f2py. Run the following commands before the execution of the python code to compile the FORTRAN file:

`f2py3 --f90flags="-g -fdefault-integer-8 -O3" -m hebbF -c hebbclean.F90`

## 

If you have any questions, feel free to open an issue or send me an email: natalya.weber (at) oist.jp.

If you use our code for your research, consider citing our paper (and [CShapes 2.0 Dataset](https://icr.ethz.ch/publications/cshapes-2/) or the [GADM 4.1 Dataset](https://gadm.org/download_country.html) if you use those datasets as well):
```
@misc{weber_use_2023,
  title = {On the Use of Associative Memory in {{Hopfield}} Networks Designed to Solve Propositional Satisfiability Problems},
  author = {Weber, Natalya and Koch, Werner and Erdem, Ozan and Froese, Tom},
  year = {2023},
  month = jul,
  number = {arXiv:2307.16807},
  eprint = {2307.16807},
  primaryclass = {nlin, q-bio},
  publisher = {{arXiv}},
  urldate = {2023-08-01},
  abstract = {Hopfield networks are an attractive choice for solving many types of computational problems because they provide a biologically plausible mechanism. The Self-Optimization (SO) model adds to the Hopfield network by using a biologically founded Hebbian learning rule, in combination with repeated network resets to arbitrary initial states, for optimizing its own behavior towards some desirable goal state encoded in the network. In order to better understand that process, we demonstrate first that the SO model can solve concrete combinatorial problems in SAT form, using two examples of the Liars problem and the map coloring problem. In addition, we show how under some conditions critical information might get lost forever with the learned network producing seemingly optimal solutions that are in fact inappropriate for the problem it was tasked to solve. What appears to be an undesirable side-effect of the SO model, can provide insight into its process for solving intractable problems.},
  archiveprefix = {arxiv},
  keywords = {Computer Science - Artificial Intelligence,Nonlinear Sciences - Adaptation and Self-Organizing Systems,Quantitative Biology - Neurons and Cognition}
}
```

