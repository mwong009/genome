<!---markdown file--->

# Genome Project

[![Documentation Status](https://readthedocs.org/projects/genome/badge/?version=latest)](https://genome.readthedocs.io/en/latest/?badge=latest)

## Overview

Genome is a Python library that allows you to develop discrete choice models with machine learning and neural network tools.
The goal is to combine cutting edge machine learning techniques with discrete choice analysis to allow better and faster model developments.
This software is specifically built for estimation of travel behaviour data sets.
It is built on the [Theano](https://github.com/Theano/Theano) deep learning library.

Genome primary interface is a basic command-line script based Python software.
Using a script allows users to describe a model 'recipe'.
The core of Genome uses the Theano engine to train neural network models.
It is adapted to work on more conventional style discrete choice models and econometric analysis.
It allows extremely fast computational speedup using tensor based computations.
This software can be configured to automatically estimate both discrete and continuous output simultaneously.

**Note:** We are currently working towards a 1.0 release.
Until it is ready, Genome is only suitable for developers who can help to find bugs and issues.
This software should **not** be used for live production.

## Features

* Specify multiple layers for a deep learning based discrete choice model
* Deep learning optimization algorithms (SGD, Adam, N-Momentum, Adagrad)
* Supports standard MNL model with reference parameters
* Generates estimated parameter tables with standard error calculations
* Loglikelihood tracking

### Under Development

* Generative model implementation
* Residual networks
* Model saving and reloading

## Installing

To install the package, run `pip3` on the project directory:

```bash
$ pip3 install --user -r 'requirements.txt' .
```

To remove package from user directory:

```bash
$ pip3 uninstall genome -y
```

### Directory Layout

`GENOME` is the project directory

* `GENOME/genome` contains the python package
* `GENOME/docs` contains the package documentation (development)
* `GENOME/scripts` contains several useful scripts for model reference
  * `scripts/data` contains sample data sets
* `GENOME/tutorials` contains examples of running the software

## Contributing

Report bugs and request features using the issue tracker, following the contributing guidelines below

* If you know exactly what the problem is, the best way to contribute is to submit a pull request, which is often very helpful in the developement of this project.
* If you do not know how to submit a pull request, create a new issue on the issue tracker page.
* If you are adding a change or a new feature to the project, make sure all the tests pass and write new test suites for the feature if possible.

### Submitting Requests

* Submit one change per pull request, do not combine features into one pull request.
* You can also discuss the new functionality or ideas before submitting the pull request in the issue tracker page.
* When writing pull requests, keep the titles short but be more descriptive.
* Please follow [PEP 8](https://www.python.org/dev/peps/pep-0008) coding style for consistency
* You can include your name/organization in your contribution to identify yourself
* All new functionality must include a test case to check if it works as expected.

### Reporting Issues

* Check open issues to see if the issue has already been reported. Check the comments and/or leave a comment if you have any useful information to help fix the issue.
* If you are not sure how to use the software or how to develop a model, start a discussion in the issue tracker page.
* Write **complete** and **informative** report on the problem or specific problem. The more well written the report, the easier and faster it can be resolved.
* Include all relevant files and/or enviroment required to reproduce the error. For example, OS: Windows; Platform: Python 3.6; Hardware: Intel i5, NVIDIA GTX 960; etc.
* Include the output of `genome version -v` so we know what version of the software it produced the issue.

## Version History

0.1.0
- 2019/05/03: Pre-development and planning stage

## Credits and Licence

Developer(s): [Melvin Wong](https://github.com/mwong009)

Organization: [LiTrans](https://litrans.ca/)

Contributors:
