from abc import abstractmethod, ABC
from tensorflow.keras.callbacks import Callback


class Resettable(ABC):
    """Interface for objects that can be reset"""

    @abstractmethod
    def reset(self) -> None:
        """Resets the object"""
        pass


class Resetter(Callback):
    """Callback which resets the object on each epoch end"""

    def __init__(self, obj: Resettable):
        super().__init__()

        assert isinstance(obj, Resettable)

        self.obj = obj

    def on_epoch_end(self, *args, **kwargs) -> None:
        self.obj.reset()
