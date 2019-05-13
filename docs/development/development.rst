.. _contribution:

How to contribute
-----------------

Discussion on development takes place in the
`Github project page <https://github.com/mwong009/genome/issues>`_.
If you find a bug in the code or if some component of the software is not working as expected, please report it so that it can be fixed as soon as possible.
Feedback from end-users are valuable to maintain quality of the code.

Reporting issues
~~~~~~~~~~~~~~~~

* Describe what you expected to happen.
* If possible, include a minimal, complete, and verifiable example to help us
  identify the issue. This also helps check that the issue is not with your own
  code.
* Describe what actually happened. Include the full traceback if there was an
  exception.

Github repository
~~~~~~~~~~~~~~~~~

The `source code <https://github.com/mwong009/genome/>`_ is maintained using Git_.

.. _Git: https://git-scm.com/

You can learn the basics of Git here:

* http://gitref.org/

Workflow
~~~~~~~~

* Register as a user on https://github.com
* Fork the project to get a personal 'copy' of the source code
* checkout a local branch ``git checkout -b  <branch name>``
* Make changes and/or additions on your local branch your fork. The preferred way is to make use of the command line or by using `Github Desktop <https://desktop.github.com/>`_
* Commit and push your changes to your remote::

    $ git push --set-upstream origin <branch name>

* Go to your github repository and submit a merge request to the project
  repository
* Provide a descriptive title and informative comments in the merge request
  form
* Wait for feedback from the developers
* Once the developers are satisfied with the merge request, your changes will
  be pushed onto the master branch
* (optional) delete your local branch

At some point someone will take a look at your changes and merge it to the
project branch if the change looks good.

Code Review
~~~~~~~~~~~

Before you start working on a new feature, please follow proper `Coding Conventions <https://www.python.org/dev/peps/pep-0008/>`_.
Most small changes are fine and can be merges right away, but for major
changes or new features, proper coding standards are required.

Other good coding style guide suggestions

* `Google Python style guide <http://google.github.io/styleguide/pyguide.html>`_

Documentation
~~~~~~~~~~~~~

Contribution to writing documentation is always welcomed.
This documentation is written in  reStructuredText_.

.. _reStructuredText: http://docutils.sourceforge.net/rst.html
