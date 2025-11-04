"""
Information about widefield imaging sessions
"""

# connect to the database
import login
login.connect()
import datajoint as dj

# import table dependencies
from schema import common_mice
from schema import common_exp
from schema.mpanze_widefield_refactor.Hardware import Objective, Microscope, WidefieldCamera

# other dependencies
import tifffile as tif
from util import pathfinding

# instantiate the schema
schema = dj.schema('mpanze_widefield_refactor', locals(), create_tables=True)

@schema
class WidefieldSession(dj.Imported):
    definition = """ # Basic info about the widefield sesssion
    -> common_exp.Session
    ---
    -> Microscope
    -> WidefieldCamera
    -> Objective.proj(top_objective = 'objective_name')
    -> Objective.proj(bottom_objective = 'objective_name')
    exposure_time               : float         # (global) exposure time in ms
    approx_fps                  : float         # approximate framerate per channel
    binning                     : tinyint       # pixel binning used (this is already applied by the imaging camera)
    pixels_w                    : smallint      # image width, post binning (2nd matrix dimension)
    pixels_h                    : smallint      # image height, post binning (1st matrix dimension)
    n_frames                    : int           # total frames per channel in recording session
    file_path                   : varchar(512)  # relative path to raw data file
    """
    # include only joytick experiments from batch 7 onwards, as these are formatted as single files
    _key_source = common_exp.Session * common_mice.Mouse.proj(batch='batch') & dict(username='mpanze', task="Joystick_with_cue") & "batch>=7"
        
    def make(self, key):
        """
        populate WidefieldSession table for batch 7 and later
        """
        # find raw imaging file - should be only one file in folder
        p_raw = (common_exp.Session() & key).glob("*_img.tif")
        if len(p_raw) == 0:
            # check if mouse is rehab mouse, on a no-imaging day
            from schema import mpanze_exp as exp
            try:
                days_from_stroke_norm = (exp.DaysFromStrokeNorm & key).fetch1("days_from_stroke_norm")
                if days_from_stroke_norm in [8,10,15,17,22,24]:
                    print(f"Mouse {key} is a rehab mouse, no imaging on day {days_from_stroke_norm}, skipping...", flush=True)
                    return
            except Exception as e:
                raise Exception(f"No raw imaging file found for session {key}")
        elif len(p_raw) > 1:
            raise Exception(f"Multiple raw imaging files found for session {key}")
        p_raw = p_raw[0]

        # get number of frames
        memmap = tif.memmap(p_raw, mode="r")
        # check that number of frames is even
        if memmap.shape[0] % 2 != 0:
            raise Exception(f"File {p_raw} has odd number of frames")
        n_frames = memmap.shape[0] // 2
        del memmap

        # get relative path
        p_tif_rel = (common_exp.Session() & key).get_relative_path(p_raw).as_posix()
        
        # make and insert entries
        entry = dict(**key, n_frames=n_frames, file_path=p_tif_rel)
        entry["pixels_h"], entry["pixels_w"] = 512, 512
        entry["approx_fps"] = 20
        entry["exposure_time"] = 25
        entry["binning"] = 4
        entry["top_objective"] = "Navitar 50mm F0.95"
        entry["bottom_objective"] = "Navitar 50mm F0.95"
        entry["microscope_name"] = "J92 widefield"
        entry["camera_name"] = "ORCA-Flash4.0 V3"
        self.insert1(entry)
    
    def get_path(self, as_posix=False):
        """
        get absolute path to file
        params:
            as_posix: bool, if True return path as string (posix format), otherwise as Path object
        """
        assert len(self) == 1, "only one entry allowed"
        p = pathfinding.get_absolute_paths(self, "file_path")
        assert len(p) == 1, "only one file allowed"
        return p[0].as_posix() if as_posix else p[0] 
        
    def get_frame(self):
        """
        returns the first frame of the widefield recording
        """
        # get a frame, for widefield registration
        assert len(self) == 1, "only one entry allowed"
        p = self.get_path()
        return tif.imread(p, key=0)
    
    def get_stack(self):
        """
        load imaging stack into memory
        """
        assert len(self) == 1, "only one entry allowed"
        p_tif = self.get_path()
        stack = tif.imread(p_tif)
        return stack
