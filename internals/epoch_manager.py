from pyparsing import deque
from typing import Any, Dict, List

class Epoch_manager:
    """
    Maintains a rolling history of per-satellite data.
    Each satellite keeps at most max_history entries.
    """

    def __init__(self, max_history: int = 50) -> None:
        """
        Parameters
        ----------
        max_history: int
            The maximum number of historical entries to keep per satellite.
        """
        self.max_history = max_history
        # maps sat_identifier -> deque of data entries
        self.sat_history_: Dict[str, deque] = {}

    def add_entry(self, sat_identifier: str, data_entry: Any) -> None:
        """
        Add a new time-ordered entry for the satellite.
        Drops the oldest entry once max_history is exceeded.

        Parameters
        ----------
        sat_identifier: str
            A unique identifier for the satellite (e.g., "G01" for GPS satellite 1).
        data_entry: Any
            The data entry to be stored for the satellite. Can be any type of data relevant to 
            the satellite's history (e.g., pseudorange, position, velocity, etc.).
        
        Returns
        -------
        None
        """
        if sat_identifier not in self.sat_history_:
            # deque with bounded size enforces the 50-limit automatically
            self.sat_history_[sat_identifier] = deque(maxlen=self.max_history)

        self.sat_history_[sat_identifier].append(data_entry)

    def get_history(self, sat_identifier: str) -> List[Any]:
        """
        Return all stored entries for a satellite in time order.
        Returns an empty list if not present.

        Parameters
        ----------
        sat_identifier: str
            The unique identifier for the satellite whose history is to be retrieved.

        Returns
        -------
        List[Any]
            A list of data entries for the specified satellite, ordered from oldest to newest.
            If the satellite identifier is not found, an empty list is returned.
        """
        if sat_identifier not in self.sat_history_:
            return []

        return list(self.sat_history_[sat_identifier])

    def batch_history(self, sats: List[str]) -> Dict[str, List[Any]]:
        """
        Given a list of satellite identifiers, return their histories as a dict.
        Missing satellites return an empty list.

        Parameters
        ----------
        sats: List[str]
            A list of satellite identifiers for which to retrieve histories.
            
        Returns
        -------
        Dict[str, List[Any]]
            A dictionary mapping each satellite identifier in the input list to its corresponding history list.
            If a satellite identifier is not found in the history manager, it will be mapped to an empty list.
        """
        result: Dict[str, List[Any]] = {}

        for sat_id in sats:
            result[sat_id] = self.get_history(sat_id)

        return result