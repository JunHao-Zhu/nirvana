from typing import Mapping


class Field:
    def __init__(self):
        self._name: str = None
        self._type = None
        self._metadata: Mapping[str, str] = None

    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, new_name: str):
        self._name = new_name

    def from_dict(self, data: Mapping[str, str]):
        self._name = data["name"]
        self._type = data["type"]
        self._metadata = data["metadata"]
