"""
Information about mouse groups
"""

# connect to the database
import login
login.connect()
import datajoint as dj

# import table dependencies
from schema import common_mice

# instantiate the schema
schema = dj.schema('mpanze_exp_refactor', locals(), create_tables=True)

@schema
class Handedness(dj.Manual):
    definition = """ # table for storing mouse handedness
    -> common_mice.Mouse
    ---
    handedness  : enum('L', 'R')       # mouse dominant hand
    """

@schema
class StrokeGroupName(dj.Lookup):
    definition = """ # table for storing group names
    group : varchar(128)     # group name
    """

@schema
class StrokeGroup(dj.Lookup):
    definition = """ # table for storing stroke group of each mouse
    -> common_mice.Mouse
    ---
    -> StrokeGroupName
    """

@schema
class NewGroup(dj.Lookup):
    definition = """ # table for storing new group names
    -> common_mice.Mouse
    ---
    -> StrokeGroupName.proj(new_group='group')  # new group name
    """
