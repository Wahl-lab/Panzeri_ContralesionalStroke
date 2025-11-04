"""
Motion correction pipeline for widefield data
Preprocessing steps:
    1. Quick check of start and end of session to see if motion correction is needed
    2. Manual assignment of motion correction parameters to sessions
    3. Compute motion correction
"""

# connect to the database
import login
login.connect()
import datajoint as dj

# import table dependencies
from schema.mpanze_widefield_refactor.WidefieldSession import WidefieldSession

# other dependencies
from util.widefield import motion_correction
import numpy as np
from scipy.stats import pearsonr
import tifffile as tif

# instantiate the schema
schema = dj.schema('mpanze_widefield_refactor', locals(), create_tables=True)

def auto_assign_low_motion(max_shift=0.5, max_std=0.1, auto_populate=False, limit=None):
    """
    Automatically assign motion correction parameters to sessions with low motion
    Sessions with low motion can skip motion correction (method="none")
    params:
        max_shift: float, maximum acceptable shift in pixels, in either direction or channel
        max_std: float, maximum standard deviation in shifts in pixels, in either direction or channel
        auto_populate: bool, if True, automatically populate the entry in the MotionCorrection table
        limit: int, limit the number of sessions to process, if None, process all
    """
    # create conditions
    conditions = [f"max_x < {max_shift}", f"max_y < {max_shift}", f"std_x < {max_std}", f"std_y < {max_std}"]
    condition = " AND ".join(conditions)
    condition_blue = condition + " AND channel='blue'"
    condition_uv = condition + " AND channel='uv'"

    # query sessions that meet conditions and have not been assigned
    sessions_start_blue = MotionCorrectionPreCheck.aggr(MotionCorrectionPreCheck.ShiftsStart & condition_blue)
    sessions_start_uv = MotionCorrectionPreCheck.aggr(MotionCorrectionPreCheck.ShiftsStart & condition_uv)
    sessions_end_blue = MotionCorrectionPreCheck.aggr(MotionCorrectionPreCheck.ShiftsEnd & condition_blue)
    sessions_end_uv = MotionCorrectionPreCheck.aggr(MotionCorrectionPreCheck.ShiftsEnd & condition_uv)
    sessions = (sessions_start_blue & sessions_start_uv & sessions_end_blue & sessions_end_uv) - MotionCorrectionAssignment
    session_keys = sessions.fetch("KEY", limit=limit)

    # iterate over sessions
    for key in session_keys:
        # insert entry in MotionCorrectionAssignment
        entry = dict(**key, mc_param_id=0)
        MotionCorrectionAssignment.insert1(entry)

        # populate entry in MotionCorrection table
        if auto_populate:
            MotionCorrection.populate(entry)

@schema
class MotionCorrectionMethod(dj.Lookup):
    definition = """ # Supported methods for motion correction
    method          : varchar(32)   # method for motion correction
    """
    contents = [
        dict(method="cv2_phaseCorrelate"),
        dict(method="none"),
        dict(method="skimage_phase_cross_correlation"),
    ]

@schema
class MotionCorrectionParameters(dj.Lookup):
    definition = """ # Parameters for motion correction
    mc_param_id     : int           # unique id for the motion correction parameters
    ---
    -> MotionCorrectionMethod       # method for motion correction
    kwargs          : longblob      # keyword arguments for motion correction function
    """
    contents = [
        dict(mc_param_id=0, method="none", kwargs=dict()),
        dict(mc_param_id=1, method="skimage_phase_cross_correlation", kwargs=dict(n_processes=2, upsample_factor=20, normalization=None)),
        dict(mc_param_id=2, method="cv2_phaseCorrelate", kwargs=dict(n_threads=-1)),
    ]

@schema
class MotionCorrectionPreCheck(dj.Computed):
    definition = """ # quick check of start and end of session to see if motion correction is needed
    -> WidefieldSession
    ---
    time_inserted=CURRENT_TIMESTAMP           : timestamp      # automatic timestamp
    """
    class Channel(dj.Part):
        definition = """ # Motion correction for each channel
        -> MotionCorrectionPreCheck
        channel                : enum('blue', 'uv')
        ---
        template_start         : longblob       # reference image for motion correction
        template_end           : longblob       # reference image for motion correction
        """

    class ShiftsStart(dj.Part):
        definition = """ # Motion correction shifts, start of the session
        -> MotionCorrectionPreCheck.Channel
        ---
        shifts           : longblob       # shifts for each frame in the stack (n_frames, 2), in image coordinates (x,y)
        max_x            : float          # maximum absolute shift in x direction
        max_y            : float          # maximum absolute shift in y direction
        mean_x           : float          # mean shift in x direction
        mean_y           : float          # mean shift in y direction
        std_x            : float          # standard deviation of shifts in x direction
        std_y            : float          # standard deviation of shifts in y direction
        """
    
    class ShiftsEnd(dj.Part):
        definition = """ # Motion correction shifts, end of the session
        -> MotionCorrectionPreCheck.Channel
        ---
        shifts           : longblob       # shifts for each frame in the stack (n_frames, 2), in image coordinates (x,y)
        max_x            : float          # maximum absolute shift in x direction
        max_y            : float          # maximum absolute shift in y direction
        mean_x           : float          # mean shift in x direction
        mean_y           : float          # mean shift in y direction
        std_x            : float          # standard deviation of shifts in x direction
        std_y            : float          # standard deviation of shifts in y direction
        """

    def make(self, key):
        # get stack path
        p_stack = (WidefieldSession & key).get_path()
        n_frames = (WidefieldSession & key).fetch1("n_frames")

        # load first half-minute of data
        stack_memmap = tif.memmap(p_stack, mode="r")
        stack_start = np.copy(stack_memmap[:600])
        stack_end = np.copy(stack_memmap[n_frames*2-600:n_frames*2])
        stack_memmap._mmap.close()

        # split into blue and uv channels
        stack_start_blue = stack_start[::2]
        stack_start_uv = stack_start[1::2]
        stack_end_blue = stack_end[::2]
        stack_end_uv = stack_end[1::2]

        # compute template for each channel
        template_start_blue = np.mean(stack_start_blue, axis=0, dtype=np.float32)
        template_start_uv = np.mean(stack_start_uv, axis=0, dtype=np.float32)
        template_end_blue = np.mean(stack_end_blue, axis=0, dtype=np.float32)
        template_end_uv = np.mean(stack_end_uv, axis=0, dtype=np.float32)

        # create entries
        entry_master = dict(**key)
        entry_channel_blue = dict(**key, channel="blue", template_start=template_start_blue, template_end=template_end_blue)
        entry_channel_uv = dict(**key, channel="uv", template_start=template_start_uv, template_end=template_end_uv)

        # compute shifts for each channel
        entry_shifts_blue_start = dict(**key, channel="blue")
        entry_shifts_uv_start = dict(**key, channel="uv")
        entry_shifts_blue_end = dict(**key, channel="blue")
        entry_shifts_uv_end = dict(**key, channel="uv")

        # compute shifts for each channel
        entry_shifts_blue_start["shifts"] = motion_correction.get_shifts_phase_cross_corr(template_start_blue, stack_start_blue, upsample_factor=20, n_processes=2, normalization=None)
        entry_shifts_uv_start["shifts"] = motion_correction.get_shifts_phase_cross_corr(template_start_uv, stack_start_uv, upsample_factor=20, n_processes=2, normalization=None)
        entry_shifts_blue_end["shifts"] = motion_correction.get_shifts_phase_cross_corr(template_start_blue, stack_end_blue, upsample_factor=20, n_processes=2, normalization=None)
        entry_shifts_uv_end["shifts"] = motion_correction.get_shifts_phase_cross_corr(template_start_uv, stack_end_uv, upsample_factor=20, n_processes=2, normalization=None)
        
        # compute metrics for shifts
        entry_shifts_blue_start["max_x"] = np.max(np.abs(entry_shifts_blue_start["shifts"][:,0]))
        entry_shifts_blue_start["max_y"] = np.max(np.abs(entry_shifts_blue_start["shifts"][:,1]))
        entry_shifts_blue_start["mean_x"] = np.mean(entry_shifts_blue_start["shifts"][:,0])
        entry_shifts_blue_start["mean_y"] = np.mean(entry_shifts_blue_start["shifts"][:,1])
        entry_shifts_blue_start["std_x"] = np.std(entry_shifts_blue_start["shifts"][:,0])
        entry_shifts_blue_start["std_y"] = np.std(entry_shifts_blue_start["shifts"][:,1])
        entry_shifts_uv_start["max_x"] = np.max(np.abs(entry_shifts_uv_start["shifts"][:,0]))
        entry_shifts_uv_start["max_y"] = np.max(np.abs(entry_shifts_uv_start["shifts"][:,1]))
        entry_shifts_uv_start["mean_x"] = np.mean(entry_shifts_uv_start["shifts"][:,0])
        entry_shifts_uv_start["mean_y"] = np.mean(entry_shifts_uv_start["shifts"][:,1])
        entry_shifts_uv_start["std_x"] = np.std(entry_shifts_uv_start["shifts"][:,0])
        entry_shifts_uv_start["std_y"] = np.std(entry_shifts_uv_start["shifts"][:,1])
        entry_shifts_blue_end["max_x"] = np.max(np.abs(entry_shifts_blue_end["shifts"][:,0]))
        entry_shifts_blue_end["max_y"] = np.max(np.abs(entry_shifts_blue_end["shifts"][:,1]))
        entry_shifts_blue_end["mean_x"] = np.mean(entry_shifts_blue_end["shifts"][:,0])
        entry_shifts_blue_end["mean_y"] = np.mean(entry_shifts_blue_end["shifts"][:,1])
        entry_shifts_blue_end["std_x"] = np.std(entry_shifts_blue_end["shifts"][:,0])
        entry_shifts_blue_end["std_y"] = np.std(entry_shifts_blue_end["shifts"][:,1])
        entry_shifts_uv_end["max_x"] = np.max(np.abs(entry_shifts_uv_end["shifts"][:,0]))
        entry_shifts_uv_end["max_y"] = np.max(np.abs(entry_shifts_uv_end["shifts"][:,1]))
        entry_shifts_uv_end["mean_x"] = np.mean(entry_shifts_uv_end["shifts"][:,0])
        entry_shifts_uv_end["mean_y"] = np.mean(entry_shifts_uv_end["shifts"][:,1])
        entry_shifts_uv_end["std_x"] = np.std(entry_shifts_uv_end["shifts"][:,0])
        entry_shifts_uv_end["std_y"] = np.std(entry_shifts_uv_end["shifts"][:,1])

        # save entries
        self.insert1(entry_master)
        self.Channel.insert1(entry_channel_blue)
        self.Channel.insert1(entry_channel_uv)
        self.ShiftsStart.insert1(entry_shifts_blue_start)
        self.ShiftsStart.insert1(entry_shifts_uv_start)
        self.ShiftsEnd.insert1(entry_shifts_blue_end)
        self.ShiftsEnd.insert1(entry_shifts_uv_end)

@schema
class MotionCorrectionAssignment(dj.Manual):
    definition = """ # Manual assignment of motion correction parameters to sessions
    -> MotionCorrectionPreCheck
    ---
    -> MotionCorrectionParameters
    """

@schema
class MotionCorrection(dj.Computed):
    definition = """ # Computes motion correction for widefield data
    -> MotionCorrectionAssignment
    ---
    shift_correlation_x     : float          # correlation of shifts in x direction between channels
    shift_correlation_y     : float          # correlation of shifts in y direction between channels    
    time_inserted=CURRENT_TIMESTAMP           : timestamp      # automatic timestamp
    """
    def make_motion_corrected_movies(self):
        assert len(self) == 1, "Select one session"
        key = self.fetch1("KEY")
        # fetch imaging file path
        p_img = (WidefieldSession & key).get_path()
        # load stack
        stack = (WidefieldSession & key).get_stack()
        blue = stack[::2]
        uv = stack[1::2]
        # fetch shifts
        shifts_blue = (MotionCorrection.Shifts & dict(**key, channel="blue")).fetch1("shifts")
        shifts_uv = (MotionCorrection.Shifts & dict(**key, channel="uv")).fetch1("shifts")
        # compute affine matrices
        M_blue = motion_correction.shifts_to_affine_matrices(shifts_blue)
        M_uv = motion_correction.shifts_to_affine_matrices(shifts_uv)
        # apply shifts in place
        import cv2
        import tqdm
        import tifffile as tif
        n_frames, h, w = stack.shape
        for i in tqdm.tqdm(range(n_frames//2)):
            blue[i] = cv2.warpAffine(blue[i], M_blue[i], (w,h))
            uv[i] = cv2.warpAffine(uv[i], M_uv[i], (w,h))
        # save stacks
        p_out_blue = p_img.with_name(p_img.stem + f"_blue_motion_corrected.tif")
        p_out_uv = p_img.with_name(p_img.stem + f"_uv_motion_corrected.tif")
        tif.imsave(p_out_blue, blue)
        tif.imsave(p_out_uv, uv)

    class Channel(dj.Part):
        definition = """ # Motion correction for each channel
        -> MotionCorrection
        channel                : enum('blue', 'uv')
        ---
        template               : longblob       # reference image for motion correction
        """
    
    class Shifts(dj.Part):
        definition = """ # Motion correction shifts
        -> MotionCorrection.Channel
        ---
        shifts                 : longblob       # shifts for each frame in the stack (n_frames, 2), in image coordinates (x,y)
        max_x                  : float          # maximum absolute shift in x direction
        max_y                  : float          # maximum absolute shift in y direction
        mean_x                 : float          # mean shift in x direction
        mean_y                 : float          # mean shift in y direction
        std_x                  : float          # standard deviation of shifts in x direction
        std_y                  : float          # standard deviation of shifts in y direction
        """
        def get_M(self):
            assert len(self) == 1, "Select one channel"
            shifts = self.fetch1("shifts")
            return motion_correction.shifts_to_affine_matrices(shifts)
    
    def make(self, key):
        # get motion correction parameters
        motion_params = (MotionCorrectionParameters & (MotionCorrectionAssignment & key)).fetch1()
        method = motion_params["method"]
        kwargs = motion_params["kwargs"]

        # create entries
        entry_master = dict(**key)
        entry_channel_blue = dict(**key, channel="blue")
        entry_channel_uv = dict(**key, channel="uv")
        entry_shifts_blue = dict(**key, channel="blue")
        entry_shifts_uv = dict(**key, channel="uv")

        # find method for motion correction
        if method == "cv2_phaseCorrelate":
            # load stack
            stack_blue, stack_uv, template_blue, template_uv = self.load_stack_and_templates(key)
            entry_channel_blue["template"] = template_blue
            entry_channel_uv["template"] = template_uv
            # compute shifts
            shift_func = motion_correction.get_shifts_cv2_phaseCorrelate
            entry_shifts_blue["shifts"] = shift_func(template_blue, stack_blue, **kwargs)
            entry_shifts_uv["shifts"] = shift_func(template_uv, stack_uv, **kwargs)
        elif method == "skimage_phase_cross_correlation":
            # load stack
            stack_blue, stack_uv, template_blue, template_uv = self.load_stack_and_templates(key)
            entry_channel_blue["template"] = template_blue
            entry_channel_uv["template"] = template_uv
            # compute shifts
            shift_func = motion_correction.get_shifts_phase_cross_corr
            entry_shifts_blue["shifts"] = shift_func(template_blue, stack_blue, **kwargs)
            entry_shifts_uv["shifts"] = shift_func(template_uv, stack_uv, **kwargs)
        elif method == "none":
            # no motion correction
            entry_channel_blue["template"] = np.nan
            entry_channel_uv["template"] = np.nan
            n_frames = (WidefieldSession & key).fetch1("n_frames")
            entry_shifts_blue["shifts"] = np.zeros((n_frames, 2), dtype=np.float32)
            entry_shifts_uv["shifts"] = np.zeros((n_frames, 2), dtype=np.float32)
        else:
            raise ValueError(f"Unsupported motion correction method: {method}")

        # compute metrics for shifts
        entry_shifts_blue["max_x"] = np.max(np.abs(entry_shifts_blue["shifts"][:,0]))
        entry_shifts_blue["max_y"] = np.max(np.abs(entry_shifts_blue["shifts"][:,1]))
        entry_shifts_blue["mean_x"] = np.mean(entry_shifts_blue["shifts"][:,0])
        entry_shifts_blue["mean_y"] = np.mean(entry_shifts_blue["shifts"][:,1])
        entry_shifts_blue["std_x"] = np.std(entry_shifts_blue["shifts"][:,0])
        entry_shifts_blue["std_y"] = np.std(entry_shifts_blue["shifts"][:,1])
        entry_shifts_uv["max_x"] = np.max(np.abs(entry_shifts_uv["shifts"][:,0]))
        entry_shifts_uv["max_y"] = np.max(np.abs(entry_shifts_uv["shifts"][:,1]))
        entry_shifts_uv["mean_x"] = np.mean(entry_shifts_uv["shifts"][:,0])
        entry_shifts_uv["mean_y"] = np.mean(entry_shifts_uv["shifts"][:,1])
        entry_shifts_uv["std_x"] = np.std(entry_shifts_uv["shifts"][:,0])
        entry_shifts_uv["std_y"] = np.std(entry_shifts_uv["shifts"][:,1])

        # compute correlation between shifts
        if method == "none":
            entry_master["shift_correlation_x"] = 1.0
            entry_master["shift_correlation_y"] = 1.0
        else:
            entry_master["shift_correlation_x"] = pearsonr(entry_shifts_blue["shifts"][:,0], entry_shifts_uv["shifts"][:,0])[0]
            entry_master["shift_correlation_y"] = pearsonr(entry_shifts_blue["shifts"][:,1], entry_shifts_uv["shifts"][:,1])[0]

        # remove nans from correlation
        if np.isnan(entry_master["shift_correlation_x"]):
            entry_master["shift_correlation_x"] = 0
        if np.isnan(entry_master["shift_correlation_y"]):
            entry_master["shift_correlation_y"] = 0

        # save entries
        self.insert1(entry_master)
        self.Channel.insert1(entry_channel_blue)
        self.Channel.insert1(entry_channel_uv)
        self.Shifts.insert1(entry_shifts_blue)
        self.Shifts.insert1(entry_shifts_uv)

    @staticmethod
    def load_stack_and_templates(key):
        # load stack
        stack = (WidefieldSession & key).get_stack()
        stack_blue = stack[::2]
        stack_uv = stack[1::2]
        # load templates
        template_blue = np.mean(stack_blue[:100], axis=0, dtype=np.float32)
        template_uv = np.mean(stack_uv[:100], axis=0, dtype=np.float32)
        return stack_blue, stack_uv, template_blue, template_uv
