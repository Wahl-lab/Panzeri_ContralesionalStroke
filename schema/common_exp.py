"""Schema for experimental information"""

import datajoint as dj
import login
from schema import common_mice
from pathlib import Path
import os
from datetime import datetime
from datetime import date as datetime_date
from typing import Union, Optional, List


schema = dj.schema('common_exp', locals(), create_tables=True)


@schema
class Anesthesia(dj.Lookup):
    definition = """ # Anesthesia Info
    anesthesia               : varchar(128)   # Anesthesia short name
    ---
    anesthesia_details       : varchar(1024)  # Longer description
    """

@schema
class Setup(dj.Lookup):
    definition = """ # Details of the setup
    setup          : varchar(128)   # Unique name of the setup
    ---
    setup_details  : varchar(1024)  # More info about the components
    """

@schema
class Task(dj.Lookup):
    definition = """ # Experimental task for the mouse
    task            : varchar(128)      # Unique name of the task
    ---
    -> common_mice.Investigator         # Keep track of whose task it is for GUI
    stage           : tinyint           # Counter for e.g. difficulty in case of learning task
    task_details    : varchar(1048)     # Task description
    """

@schema
class Session(dj.Manual):
    definition = """ # Information about the session and experimental setup
    -> common_mice.Mouse
    day             : date           # Date of the experimental session (YYYY-MM-DD)
    session_num     : tinyint        # Counter of experimental sessions on the same day (base 1)
    ---
    session_id      : varchar(128)   # Unique identifier
    session_path    : varchar(256)   # Path of this session relative to the Neurophysiology-Storage1 DATA directory
    session_counter : smallint       # Overall counter of all sessions across mice (base 0)
    experimenter    : varchar(128)   # Who actually performed the experiment, must be a username from Investigator()
    -> Anesthesia
    -> Setup
    -> Task
    session_notes   : varchar(2048)  # description of important things that happened
    """

    @staticmethod
    def create_id(investigator_name: str, mouse_id: int, date: Union[datetime, str], session_num: int) -> str:
        """
        Create unique session id with the format inv_MXXX_YYYY-MM-DD_ZZ:
        inv: investigator shortname (called 'experimenter' in Adrians GUI)
        MXXX: investigator-specific mouse number (M + 0-padded three-digit number, e.g. M018)
        YYYY-MM-DD: date of session
        ZZ: 0-padded two-digit counter of the sessions on that day
        Note that trials are with base 1
        Adrian 2019-08-12
        Adapted by Hendrik 2021-05-04
        ------------------------------------------------------------------------------------

        Args:
            investigator_name:  Shortname of the investigator for this session (from common_mice.Investigator)
            mouse_id:           Investigator-specific mouse ID (from common_mice.Mice)
            date:               Datetime object of the session date, or string with format YYYY-MM-DD
            session_num:        Iterator for number of sessions on that day

        Returns:
            Unique string ID for this session

        """

        # first part: mouse identifier
        mouse_id_str = 'M{:03d}'.format(int(mouse_id))
        first_part = 'session_' + investigator_name + '_' + mouse_id_str

        # second: Transform datetime object to string, while removing the time stamp
        if type(date) == datetime or type(date) == datetime_date:
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = date

        # third: trial with leading zeros
        trial_str = '{:02d}'.format(session_num)

        # combine and return the unique session id
        return first_part + '_' + date_str + '_' + trial_str
    
    def get_relative_path(self, absolute_path: Union[str, Path]) -> Path:
        """
        Returns path relative to session directory

        Parameters
        ----------
        absolute_path : Union[str, Path]
            absolute path to a file or directory.

        Returns
        -------
        Path
            path relative to session directory.
        """
        from util.pathfinding import get_relative_path
        if len(self) != 1:
            raise Exception("Please select only one session!")
        p_session = self.fetch1("session_path")        
        return get_relative_path(absolute_path, subfolder=p_session)
    
    def glob(self, pattern: str, recursive: bool = False) -> List[Path]:
        """
        Finds all unique files that match the specified pattern, relative to session folder.
        Automatically includes alternative data directories in the search

        Parameters
        ----------
        pattern : str
            pattern to match, standard globbing patterns and wildcards are accepted.
        recursive : bool, optional
            if True, will also search in subfolders. The default is False.

        Returns
        -------
        List[Path]
            List of absolute paths that match the pattern.
        """
        from util.pathfinding import glob
        if len(self) != 1:
            raise Exception("Please select only one session!")
        p_session = self.fetch1("session_path")
        return glob(pattern, recursive=recursive, subfolder=p_session)

    def helper_insert1(self, entry_dict: dict) -> str:
        """
        Simplified insert function that takes care of id and counter values.
        Adrian 2019-08-19

        Args:
            entry_dict: Dictionary containing all key, value pairs for the session except for
                                the id and counter

        Returns:
            Status update string confirming successful insertion.
        """
        from util.pathfinding import get_relative_path

        # Make copy so that changes do not affect original dict
        new_entry_dict = entry_dict.copy()

        sess_id = self.create_id(new_entry_dict['username'], new_entry_dict['mouse_id'], new_entry_dict['day'],
                                 new_entry_dict['session_num'])
        if len(self.fetch('session_counter')) == 0:
            counter = 0
        else:
            counter = max(self.fetch('session_counter')) + 1

        # Transform absolute path from the GUI to the relative path on the Neurophys-Storage1 server
        new_entry_dict['session_path'] = get_relative_path(entry_dict["session_path"]).as_posix()

        # add automatically computed values to the dictionary
        entry = dict(**new_entry_dict, session_id=sess_id, session_counter=counter)

        self.insert1(entry)
        # Only print out primary keys
        key_dict = {your_key: entry[your_key] for your_key in ['username', 'mouse_id', 'day', 'session_num']}
        return 'Inserted new session: {}'.format(key_dict)

    def get_key(self, counter: int) -> dict:
        """
        Return uniquely identifying primary keys that corresponds to global counter of sessions.

        Args:
            counter: Overall counter of all sessions across mice (base 0).

        Returns:
            Primary keys of the session with the provided counter.
        """
        return (self & {'session_counter': counter}).fetch1('KEY')
