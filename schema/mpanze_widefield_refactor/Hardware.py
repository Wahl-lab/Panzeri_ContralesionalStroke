"""
Hardware-related info
"""

# connect to the database
import login
login.connect()
import datajoint as dj

# instantiate the schema
schema = dj.schema('mpanze_widefield_refactor', locals(), create_tables=True)

@schema
class Microscope(dj.Lookup):
    definition = """ # Microscope used for widefield imaging
    microscope_name             : varchar(256)  # microscope name
    ---
    microscope_details          : varchar(1048) # additional details
    """
@schema
class Objective(dj.Lookup):
    definition = """ # Objectives used for widefield imaging
    objective_name              : varchar(100)  # short name identifying the objective
    ---
    efl                         : float         # effective focal length of the objective in mm
    bfl = NULL                  : float         # back focal length of the objective in mm, if available
    f_stop = NULL               : float         # minimum F stop, if available
    na = NULL                   : float         # objective NA (numerical aperture), if available
    magnification = NULL        : float         # magnification of objective if applicable
    model = NULL                : varchar(256)  # manufacturer and model number
    objective_notes = ""        : varchar(256)  # any relevant notes about the objective
    """

@schema
class WidefieldCamera(dj.Lookup):
    definition = """ # Camera used for widefield imaging
    camera_name                : varchar(128)   # camera shorname
    ---
    manufacturer               : varchar(128)   # camera manufacturer
    model                      : varchar(128)   # camera model
    sensor_type                : varchar(64)    # type of sensor
    n_pixels_h                 : int            # height of the sensor in pixels
    n_pixels_w                 : int            # width of the sensor in pixels
    pixel_w                    : float          # pixel width in um
    pixel_h                    : float          # pixel height in um
    camera_notes               : varchar(256)   # any additional camera notes
    """
