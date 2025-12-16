from pyparsing import deque
from typing import Any, Dict, List

class Epoch_manager:
    """
    Maintains a rolling history of per-satellite data.
    Each satellite keeps at most max_history entries.
    """

    def __init__(self, max_history: int = 50) -> None:
        self.max_history = max_history
        # maps sat_identifier -> deque of data entries
        self.sat_history_: Dict[str, deque] = {}

    def add_entry(self, sat_identifier: str, data_entry: Any) -> None:
        """
        Add a new time-ordered entry for the satellite.
        Drops the oldest entry once max_history is exceeded.
        """
        if sat_identifier not in self.sat_history_:
            # deque with bounded size enforces the 50-limit automatically
            self.sat_history_[sat_identifier] = deque(maxlen=self.max_history)

        self.sat_history_[sat_identifier].append(data_entry)

    def get_history(self, sat_identifier: str) -> List[Any]:
        """
        Return all stored entries for a satellite in time order.
        Returns an empty list if not present.
        """
        if sat_identifier not in self.sat_history_:
            return []

        return list(self.sat_history_[sat_identifier])

    def batch_history(self, sats: List[str]) -> Dict[str, List[Any]]:
        """
        Given a list of satellite identifiers, return
        their histories as a dict.
        Missing satellites return an empty list.
        """
        result: Dict[str, List[Any]] = {}

        for sat_id in sats:
            result[sat_id] = self.get_history(sat_id)

        return result