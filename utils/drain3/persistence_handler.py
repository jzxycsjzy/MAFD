# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
import os


class PersistenceHandler(ABC):

    # @abstractmethod
    def save_state(self, state, servicename):
        with open('./MAFD/DrainModel/' + servicename, 'wb+') as f:
            f.write(state)

    # @abstractmethod
    def load_state(self, servicename):
        f = open('./MAFD/DrainModel/' + servicename, 'rb+')
        state = f.read()
        f.close()
        return state
