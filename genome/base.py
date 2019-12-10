# -*- coding: utf-8 -*-
"""Base modules"""


class BaseEnvironment:
    def __init__(self):
        pass

    def setup(self, **kwargs):
        """Setup the base components in the simulation envionment

        setup(data)

        """
        self.data = kwargs.get('data', None)
        if self.data is not None:
            pass
        else:
            raise NameError('Datafile not given! Usage: setup(data=\'filename\',...)')


class Environment(BaseEnvironment):
    """Execution envionment for the Genome model training process.

    Optimization is simulated by invoking :meth:`run()`.
    """
    def __init__(self):
        self._active_process = None

    def run(self, epochs=None):
        """Executes the step() process until a given number of epochs.

        If it is ``None`` (default), this method will return when the early stopping
        criterion is met.
        """

    def exit(self):
        """Stops the current process.
        """
        pass

    def step(self):
        """Processes one step of a training cycle.

        Ends when the number of reached or early stopping criterion is met.
        """
        pass
