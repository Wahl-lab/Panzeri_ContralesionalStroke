"""
Movement segmentation pipeline
"""

# connect to the database
import login
login.connect()

# imports
import datajoint as dj
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import binary_closing, label

# import table dependencies
from schema.mpanze_paw_tracking_refactor.DeepLabCut import WeightedHandPosition, FilteredDLC, Features
from schema.mpanze_paw_tracking_refactor.PawRecording import PawRecording, Synchronisation, JoystickPosition
from schema.mpanze_exp_refactor import mpanze_exp_refactor as exp

# instantiate the schema
schema = dj.schema('mpanze_paw_tracking_refactor', locals(), create_tables=True)

@schema
class MovementSegmentationParams(dj.Lookup):
    definition = """ # parameters for movement segmentation
    pt_seg_id               : int           # unique identifier for parameter set
    ---
    savgol_window           : float         # window size of savgol filter, in seconds
    savgol_order            : int           # order of savgol filter
    v_thresh                : float         # velocity threshold
    structure               : float         # structure size for morphological operations
    prominence              : float         # prominence threshold for sub-epoch detection
    min_distance            : float         # minimum distance between sub-epoch local minima
    min_duration            : float         # minimum duration of epochs in seconds
    post_reward_time        : float         # time after reward to consider as post-reward epoch in seconds
    """
    contents = [
        dict(pt_seg_id=0, savgol_window=0.075, savgol_order=2, v_thresh=5, structure=0.075, min_distance=0.050, min_duration=0.05, prominence=7.5, post_reward_time=0.75),
    ]

@schema
class MovementSegmentation(dj.Computed):
    definition = """
    -> WeightedHandPosition
    -> MovementSegmentationParams
    """
    class Hand(dj.Part):
        definition = """
        -> MovementSegmentation
        hand                : enum('L', 'R')    # hand
        ---
        side                : enum('ipsi', 'contra')    # ipsilateral or contralateral to the task hand
        """
        def epoch_features(self, subtract_baseline=False, labels_to_include=None):
            from tqdm import tqdm
            import pandas as pd
            data = []
            # get features
            for s in tqdm(self.fetch("KEY"), desc='fetching epoch features'):
                df = (self & s).epoch_features_1(subtract_baseline=subtract_baseline, labels_to_include=labels_to_include)
                data.append(df)
            # concatenate dataframes
            return pd.concat(data)

        def epoch_features_1(self, subtract_baseline=False, labels_to_include=None):
            assert len(self) == 1, "only one entry allowed"
            # fetch features
            if not labels_to_include is None:
                restriction = [f"label = '{l}'" for l in labels_to_include]
                names, features = (Features.Feature & self.proj() & restriction).fetch("label", "feature")
            else:
                names, features = (Features.Feature & self.proj()).fetch("label", "feature")
            features = np.stack(features).T
            names_min = [f"{name}_min" for name in names]
            names_max = [f"{name}_max" for name in names]
            names_mean = [f"{name}_mean" for name in names]
            names = ["epoch_id"] + names_min + names_max + names_mean

            # get epoch features
            e_ids, starts, ends = (MovementSegmentation.Epoch & self.proj()).fetch("epoch_id", "start_frame", "end_frame", order_by="epoch_id")
            n_frames = (PawRecording.Hand & self.proj()).fetch1("n_frames")
            epoch_features = []

            if subtract_baseline:
                # get resting mask
                rest_mask = (RestingMask.Hand & self.proj()).fetch1("resting_mask")
                rest = np.median(features[rest_mask], axis=0)

            for eid, s, e in zip(e_ids, starts, ends):
                # get mask
                mask = MovementSegmentation.frames_to_mask(n_frames, s, e)
                # get features
                maxs = features[mask].max(axis=0)
                mins = features[mask].min(axis=0)
                means = features[mask].mean(axis=0)
                if subtract_baseline:
                    maxs -= rest
                    mins -= rest
                    means -= rest
                # append features
                epoch_features.append(np.concatenate(((eid,), mins, maxs, means)))
            epoch_features = np.stack(epoch_features)
            # create dataframe
            import pandas as pd
            df = pd.DataFrame(epoch_features, columns=names)
            df = df.set_index("epoch_id")
            # merge with indexing dataframe
            df_session = (MovementSegmentation.Epoch.proj() & self.proj()).fetch(format='frame')
            return df_session.join(df)

        def sub_epoch_features(self, subtract_baseline=False, labels_to_include=None):
            from tqdm import tqdm
            import pandas as pd
            data = []
            # get features
            for s in tqdm(self.fetch("KEY"), desc='fetching subepoch features'):
                df = (self & s).sub_epoch_features_1(subtract_baseline=subtract_baseline, labels_to_include=labels_to_include)
                data.append(df)
            # concatenate dataframes
            return pd.concat(data)

        def sub_epoch_features_1(self, subtract_baseline=False, labels_to_include=None):
            assert len(self) == 1, "only one entry allowed"
            # fetch features
            if not labels_to_include is None:
                restriction = [f"label = '{l}'" for l in labels_to_include]
                names, features = (Features.Feature & self.proj() & restriction).fetch("label", "feature")
            else:
                names, features = (Features.Feature & self.proj()).fetch("label", "feature")
            features = np.stack(features).T
            names_min = [f"{name}_min" for name in names]
            names_max = [f"{name}_max" for name in names]
            names_mean = [f"{name}_mean" for name in names]
            names = ["epoch_id", "sub_epoch_id"] + names_min + names_max + names_mean

            # get sub-epoch features
            e_ids, se_ids, starts, ends = (MovementSegmentation.SubEpoch & self.proj()).fetch("epoch_id", "sub_epoch_id", "start_frame", "end_frame", order_by=("epoch_id", "sub_epoch_id"))
            n_frames = (PawRecording.Hand & self.proj()).fetch1("n_frames")
            subepoch_features = []

            if subtract_baseline:
                # get resting mask
                rest_mask = (RestingMask.Hand & self.proj()).fetch1("resting_mask")
                rest = np.median(features[rest_mask], axis=0)

            for eid, seid, s, e in zip(e_ids, se_ids, starts, ends):
                # get mask
                mask = MovementSegmentation.frames_to_mask(n_frames, s, e)
                # get features
                maxs = features[mask].max(axis=0)
                mins = features[mask].min(axis=0)
                means = features[mask].mean(axis=0)
                if subtract_baseline:
                    maxs -= rest
                    mins -= rest
                    means -= rest
                # append features
                subepoch_features.append(np.concatenate(((eid, seid), mins, maxs, means)))
            subepoch_features = np.stack(subepoch_features)
            # create dataframe
            import pandas as pd
            df = pd.DataFrame(subepoch_features, columns=names)
            df = df.set_index(["epoch_id", "sub_epoch_id"])
            # merge with indexing dataframe
            df_session = (MovementSegmentation.SubEpoch.proj() & self.proj()).fetch(format='frame')
            return df_session.join(df)
            
    class Epoch(dj.Part):
        definition = """ # segmentation into epochs
        -> MovementSegmentation.Hand
        epoch_id            : int               # epoch identifier
        ---
        start_frame         : int               # start frame
        end_frame           : int               # end frame (exclusive)
        start_time          : float             # start time
        end_time            : float             # end time (exclusive)
        """
        def plot_skeleton(self, ax=None, lw=1, alpha=.5, cmap='winter', skip=1, joystick=True):
            import matplotlib.pyplot as plt
            assert len(self) == 1
            key, start_frame, end_frame = self.fetch1("KEY", "start_frame", "end_frame")
            x, y = (FilteredDLC.Label & key).fetch("x", "y", order_by="label")
            X, Y = np.stack(x).T, np.stack(y).T
            X_grasp = X[start_frame:end_frame:skip]
            Y_grasp = Y[start_frame:end_frame:skip]
            if ax is None:
                ax = plt.gca()
            # plot skeleton
            colormap = plt.get_cmap(cmap)
            colors = [colormap(i) for i in np.linspace(0, 1, X_grasp.shape[0])]
            for x, y, c in zip(X_grasp, Y_grasp, colors):
                self._plot_skeleton(x, y, ax, lw=lw, alpha=alpha, color=c)
            
            # get frame dimensions
            h, w = (PawRecording.Hand & key).fetch1("pixels_h", "pixels_w")
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)  # invert y-axis
            ax.set_aspect('equal')
            # add joystick
            if joystick:
                jx, jy = (JoystickPosition.Hand & key).fetch1("x", "y")
                ax.plot((jx,jx), (jy, h), color='k', lw=2)
            # add frame rectangle
            ax.set_axis_off()
            ax.add_patch(plt.Rectangle((1, 1), w-1, h-1, fill=False, edgecolor='k', lw=1))
            # add wrist trajecotry
            x_wrist = X_grasp[:,-1]
            y_wrist = Y_grasp[:,-1]
            ax.plot(x_wrist, y_wrist, color='k', lw=1, alpha=1)
        
        @staticmethod
        def _plot_skeleton(x_labels, y_labels, ax, lw, alpha, color):
            ax.plot((x_labels[0], x_labels[1]), (y_labels[0], y_labels[1]), color=color, lw=lw, alpha=alpha)   # 1_knuckle to 1_mid
            ax.plot((x_labels[1], x_labels[2]), (y_labels[1], y_labels[2]), color=color, lw=lw, alpha=alpha)   # 1_mid to 1_tip
            ax.plot((x_labels[3], x_labels[4]), (y_labels[3], y_labels[4]), color=color, lw=lw, alpha=alpha)   # 2_knuckle to 2_mid
            ax.plot((x_labels[4], x_labels[5]), (y_labels[4], y_labels[5]), color=color, lw=lw, alpha=alpha)   # 2_mid to 2_tip
            ax.plot((x_labels[6], x_labels[7]), (y_labels[6], y_labels[7]), color=color, lw=lw, alpha=alpha)   # 3_knuckle to 3_mid
            ax.plot((x_labels[7], x_labels[8]), (y_labels[7], y_labels[8]), color=color, lw=lw, alpha=alpha)   # 3_mid to 3_tip
            ax.plot((x_labels[9], x_labels[10]), (y_labels[9], y_labels[10]), color=color, lw=lw, alpha=alpha)  # 4_knuckle to 4_mid
            ax.plot((x_labels[10], x_labels[11]), (y_labels[10], y_labels[11]), color=color, lw=lw, alpha=alpha)  # 4_mid to 4_tip
            ax.plot((x_labels[0], x_labels[-1]), (y_labels[0], y_labels[-1]), color=color, lw=lw, alpha=alpha)  # 1_knuckle to wrist
            ax.plot((x_labels[3], x_labels[-1]), (y_labels[3], y_labels[-1]), color=color, lw=lw, alpha=alpha)  # 2_knuckle to wrist
            ax.plot((x_labels[6], x_labels[-1]), (y_labels[6], y_labels[-1]), color=color, lw=lw, alpha=alpha)  # 3_knuckle to wrist
            ax.plot((x_labels[9], x_labels[-1]), (y_labels[9], y_labels[-1]), color=color, lw=lw, alpha=alpha)  # 4_knuckle to wrist
            ax.plot((x_labels[-2], x_labels[-1]), (y_labels[-2], y_labels[-1]), color=color, lw=lw, alpha=alpha)  # elbow to wrist

    class SubEpoch(dj.Part):
        definition = """
        -> MovementSegmentation.Epoch
        sub_epoch_id         : int               # sub-epoch identifier (unique within the epoch)
        ---
        global_id            : int               # global epoch identifier (unique within the session)
        start_frame          : int               # start frame
        end_frame            : int               # end frame (exclusive)
        start_time           : float             # start time
        end_time             : float             # end time (exclusive)
        """

    def make(self, key):
        # get parameters
        params = (MovementSegmentationParams & key).fetch1()
       
        # iterate over hands
        entries_hands = []
        entries_epochs = []
        entries_sub_epochs = []
        hand_keys = (WeightedHandPosition.Hand & key).fetch("KEY")
        for hand_key in hand_keys:
            hand_key = {**key, **hand_key}
            side = (PawRecording.Hand & hand_key).fetch1("side")
            entry_hand = dict(**hand_key, side=side)
            entries_hands.append(entry_hand)
            
            # get frame timestamps and fps
            fps = (PawRecording.Hand & hand_key).fetch1("fps")
            t = (Synchronisation.Hand & hand_key).fetch1("frame_timestamps")
            
            # fetch velocity and filter
            v = (WeightedHandPosition.Hand & hand_key).fetch_velocity_1(norm=True)
            savgol_window = int(params["savgol_window"] * fps)
            v = savgol_filter(v, window_length=savgol_window, polyorder=params["savgol_order"])
            v[v < 0] = 0

            # extract epochs
            structure = int(params["structure"] * fps)
            v_thresh = params["v_thresh"]
            epoch_frames = self.get_epochs(v, v_thresh, structure)
            epoch_times = t[epoch_frames]
            epoch_durations = epoch_times[:,1] - epoch_times[:,0]
            # filter epochs
            epoch_frames = epoch_frames[epoch_durations >= params["min_duration"]]
            epoch_times = epoch_times[epoch_durations >= params["min_duration"]]

            # extract sub-epochs
            min_distance = int(params["min_distance"] * fps)
            prominence = params["prominence"]
            sub_epochs = self.get_sub_epochs(v, epoch_frames, min_distance, prominence)

            # create epoch entries
            for i, (start_frame, end_frame) in enumerate(epoch_frames):
                entry_epoch = dict(
                    **hand_key, epoch_id=i, start_frame=start_frame, end_frame=end_frame,
                    start_time=t[start_frame], end_time=t[end_frame],
                )
                entries_epochs.append(entry_epoch)

                for j, global_id, sub_epoch_start, sub_epoch_end in sub_epochs[i]:
                    entry_sub_epoch = dict(
                        **hand_key, epoch_id=i, sub_epoch_id=j, global_id=global_id,
                        start_frame=sub_epoch_start, end_frame=sub_epoch_end,
                        start_time=t[sub_epoch_start], end_time=t[sub_epoch_end],
                    )
                    entries_sub_epochs.append(entry_sub_epoch)
        
        # insert data
        self.insert1(key)
        self.Hand.insert(entries_hands)
        self.Epoch.insert(entries_epochs)
        self.SubEpoch.insert(entries_sub_epochs)

    @staticmethod
    def frames_to_mask(n_frames, start_frame, end_frame):
        mask = np.zeros(n_frames, dtype=bool)
        mask[start_frame:end_frame] = True
        return mask

    @staticmethod
    def get_epochs(v, v_thresh, structure):
        # find movement mask
        movement_mask = v > v_thresh
        movement_mask = binary_closing(movement_mask, structure=np.ones([structure]))
        # extract epochs
        mask_labeled, n_components = label(movement_mask)
        epoch_frames = []
        for i in range(1, n_components+1):
            start, end = np.nonzero(mask_labeled == i)[0][[0,-1]]
            epoch_frames.append([start, end+1]) # end is exclusive
        epoch_frames = np.array(epoch_frames)
        return epoch_frames
    
    @staticmethod
    def get_sub_epochs(v, epoch_frames, min_distance, prominence):
        # find sub-epochs using local minima
        sub_epochs = {}
        global_id = 0
        for i, (start, end) in enumerate(epoch_frames):
            epoch_mask = MovementSegmentation.frames_to_mask(len(v), start, end)
            v_epoch = v[epoch_mask]
            # find local minima
            local_minima, _ = find_peaks(-v_epoch, prominence=prominence, distance=min_distance)
            local_minima = [0, *local_minima, len(v_epoch)-1]
            # create sub_epochs
            sub_epochs[i] = []
            n_sub_epochs = len(local_minima) - 1
            for j in range(n_sub_epochs):
                sub_epoch_start = local_minima[j] + start
                sub_epoch_end = local_minima[j+1] + start
                sub_epochs[i].append([j, global_id, sub_epoch_start, sub_epoch_end])
                global_id += 1
            sub_epochs[i] = np.array(sub_epochs[i])
        
        return sub_epochs
    
@schema
class JoystickOcclusion(dj.Computed):
    definition = """ # compute number of hand labels in different zones
    -> FilteredDLC
    -> JoystickPosition
    ---
    """
    class Hand(dj.Part):
        definition = """ # hand occlusion data
        -> JoystickOcclusion
        hand                : enum('L', 'R')    # hand
        ---
        side                : enum('ipsi', 'contra')    # ipsilateral or contralateral to the task hand
        bound_l             : int           # threshold for left side of joystick occlusion box
        bound_r             : int           # threshold for right side of joystick occlusion box
        rest_occupancy      : longblob      # occupancy in rest zone
        search_occupancy    : longblob      # occupancy in search zone
        joystick_occupancy  : longblob      # occupancy in joystick zone
        """
    
    def make(self, key):
        # iterate over hands
        entries_hands = []
        hand_keys = (FilteredDLC.Hand * JoystickPosition.Hand.proj() & key).fetch("KEY")
        for hand_key in hand_keys:
            side = (FilteredDLC.Hand & hand_key).fetch1("side")
            # get hand position data
            x, y = (FilteredDLC.Label & hand_key & "label!='elbow'").fetch("x", "y")
            x = np.stack(x).T
            y = np.stack(y).T
            # get joystick position data
            jx = (JoystickPosition.Hand & hand_key).fetch1("x")
            # get joystick bounds
            bound_l = int(jx - 100)
            bound_r = int(jx + 60)
            # get occupancy in joystick zones
            rest_occupancy = (x < bound_l).sum(axis=1)
            search_occupancy = (x > bound_r).sum(axis=1)
            joystick_occupancy = ((x >= bound_l) & (x <= bound_r)).sum(axis=1)
            # create entry
            entry_hand = dict(
                **hand_key, side=side, bound_l=bound_l, bound_r=bound_r,
                rest_occupancy=rest_occupancy, search_occupancy=search_occupancy, joystick_occupancy=joystick_occupancy
            )
            entries_hands.append(entry_hand)

        # insert data
        self.insert1(key)
        self.Hand.insert(entries_hands)

@schema
class ReachEpoch(dj.Computed):
    definition = """ # compute reach epochs
    -> MovementSegmentation.Hand
    -> JoystickOcclusion.Hand
    ---
    """
    _key_source = (MovementSegmentation.Hand * JoystickOcclusion.Hand.proj()) & "side='ipsi'" # only ipsilateral hand
    class Epoch(dj.Part):
        definition = """ # epochs
        -> ReachEpoch
        epoch_id            : int               # epoch identifier
        ---
        """
    class SubEpoch(dj.Part):
        definition = """ # sub-epochs
        -> ReachEpoch.Epoch
        sub_epoch_id         : int               # sub-epoch identifier (unique within the epoch)
        ---
        is_reach             : bool              # is reach epoch
        """
    
    def make(self, key):
        # fetch occlusion data
        rest_occupancy, search_occupancy, joystick_occupancy = (JoystickOcclusion.Hand & key).fetch1("rest_occupancy", "search_occupancy", "joystick_occupancy")
        # fetch subepoch data
        se_starts, se_ends, se_ids, e_ids = (MovementSegmentation.SubEpoch & key).fetch("start_frame", "end_frame", "sub_epoch_id", "epoch_id", order_by=('epoch_id', 'sub_epoch_id'))

        # find reaches
        crossover_frames = np.nonzero(np.diff(np.sign(rest_occupancy - (joystick_occupancy + search_occupancy)), prepend=0) <= -1)[0]
        epoch_classes = {}
        for cr in crossover_frames:
            for s, e, seid, eid in zip(se_starts, se_ends, se_ids, e_ids):
                if (s <= cr <= e) and (rest_occupancy[s] > joystick_occupancy[s] + search_occupancy[s]):
                    epoch_classes[(eid, seid)] = True
                    break
        # create entries
        entries_epochs = [dict(**key, epoch_id=e_id) for e_id in np.unique(e_ids)]
        entries_sub_epochs = []
        for seid, eid in zip(se_ids, e_ids):
            entry_sub_epoch = dict(
                **key,
                epoch_id=eid,
                sub_epoch_id=seid,
                is_reach = (eid, seid) in epoch_classes.keys()
            )
            entries_sub_epochs.append(entry_sub_epoch)
        
        # insert data
        self.insert1(key)
        self.Epoch.insert(entries_epochs)
        self.SubEpoch.insert(entries_sub_epochs)

@schema
class RetractEpoch(dj.Computed):
    definition = """ # compute retract epochs
    -> MovementSegmentation.Hand
    -> JoystickOcclusion.Hand
    ---
    """
    _key_source = (MovementSegmentation.Hand * JoystickOcclusion.Hand.proj()) & "side='ipsi'" # only ipsilateral hand
    class Epoch(dj.Part):
        definition = """ # epochs
        -> RetractEpoch
        epoch_id            : int               # epoch identifier
        ---
        """
    class SubEpoch(dj.Part):
        definition = """ # sub-epochs
        -> RetractEpoch.Epoch
        sub_epoch_id         : int               # sub-epoch identifier (unique within the epoch)
        ---
        is_retract           : bool              # is retract epoch
        """
    
    def make(self, key):
        # fetch occlusion data
        rest_occupancy, search_occupancy, joystick_occupancy = (JoystickOcclusion.Hand & key).fetch1("rest_occupancy", "search_occupancy", "joystick_occupancy")
        # fetch subepoch data
        se_starts, se_ends, se_ids, e_ids = (MovementSegmentation.SubEpoch & key).fetch("start_frame", "end_frame", "sub_epoch_id", "epoch_id", order_by=('epoch_id', 'sub_epoch_id'))

        # find reaches
        crossover_frames = np.nonzero(np.diff(np.sign(rest_occupancy - (joystick_occupancy + search_occupancy)), prepend=0) >= 1)[0]
        epoch_classes = {}
        for cr in crossover_frames:
            for s, e, seid, eid in zip(se_starts, se_ends, se_ids, e_ids):
                if (s <= cr <= e) and (rest_occupancy[e] > joystick_occupancy[e] + search_occupancy[e]):
                    epoch_classes[(eid, seid)] = True
                    break
        # create entries
        entries_epochs = [dict(**key, epoch_id=e_id) for e_id in np.unique(e_ids)]
        entries_sub_epochs = []
        for seid, eid in zip(se_ids, e_ids):
            entry_sub_epoch = dict(
                **key,
                epoch_id=eid,
                sub_epoch_id=seid,
                is_retract = (eid, seid) in epoch_classes.keys()
            )
            entries_sub_epochs.append(entry_sub_epoch)
        
        # insert data
        self.insert1(key)
        self.Epoch.insert(entries_epochs)
        self.SubEpoch.insert(entries_sub_epochs)

@schema
class SearchEpoch(dj.Computed):
    definition = """ # compute search epochs
    -> MovementSegmentation.Hand
    -> JoystickOcclusion.Hand
    ---
    """
    _key_source = (MovementSegmentation.Hand * JoystickOcclusion.Hand.proj()) & "side='ipsi'" # only ipsilateral hand
    class Epoch(dj.Part):
        definition = """ # epochs
        -> SearchEpoch
        epoch_id            : int               # epoch identifier
        ---
        """
    class SubEpoch(dj.Part):
        definition = """ # sub-epochs
        -> SearchEpoch.Epoch
        sub_epoch_id         : int               # sub-epoch identifier (unique within the epoch)
        ---
        is_search            : bool              # is search epoch
        """
    
    def make(self, key):
        # fetch occlusion data
        rest_occupancy, search_occupancy, joystick_occupancy = (JoystickOcclusion.Hand & key).fetch1("rest_occupancy", "search_occupancy", "joystick_occupancy")
        # fetch subepoch data
        se_starts, se_ends, se_ids, e_ids = (MovementSegmentation.SubEpoch & key).fetch("start_frame", "end_frame", "sub_epoch_id", "epoch_id", order_by=('epoch_id', 'sub_epoch_id'))

        # find searches
        epoch_classes = {}
        for s, e, seid, eid in zip(se_starts, se_ends, se_ids, e_ids):
            if (rest_occupancy[s] <= joystick_occupancy[s] + search_occupancy[s]) and (rest_occupancy[e] <= joystick_occupancy[e] + search_occupancy[e]):
                epoch_classes[(eid, seid)] = True
        
        # create entries
        entries_epochs = [dict(**key, epoch_id=e_id) for e_id in np.unique(e_ids)]
        entries_sub_epochs = []
        for seid, eid in zip(se_ids, e_ids):
            entry_sub_epoch = dict(
                **key,
                epoch_id=eid,
                sub_epoch_id=seid,
                is_search = (eid, seid) in epoch_classes.keys()
            )
            entries_sub_epochs.append(entry_sub_epoch)
        
        # insert data
        self.insert1(key)
        self.Epoch.insert(entries_epochs)
        self.SubEpoch.insert(entries_sub_epochs)

@schema
class SubEpochClassification(dj.Computed):
    definition = """ # classify sub-epochs
    -> ReachEpoch
    -> RetractEpoch
    -> SearchEpoch
    -> exp.JoystickPresence
    -> Synchronisation.Hand
    ---
    t_to_exclude            : float            # time from joystick_out to exclude
    """
    epoch_labels = ['reach', 'search', 'retract', 'excluded', 'other']
    class Epoch(dj.Part):
        definition = """ # epochs
        -> SubEpochClassification
        epoch_id            : int               # epoch identifier
        ---
        """
    class SubEpoch(dj.Part):
        definition = """ # sub-epochs
        -> SubEpochClassification.Epoch
        sub_epoch_id         : int               # sub-epoch identifier (unique within the epoch)
        ---
        sub_epoch_class      : enum('reach', 'search', 'retract', 'excluded', 'other')
        joystick_frames      : int               # number of frames with joystick present
        joystick_percentage  : float             # percentage of frames with joystick present
        """
    
    def make(self, key):
        t_to_exclude = 1
        # get joystick presence data
        t_joystick_in, t_joystick_out = (exp.JoystickPresence.Trial & key).fetch("t_joystick_in", "t_joystick_out", order_by="trial_id")
        t = (Synchronisation.Hand & key).fetch1("frame_timestamps")
        joystick_present_mask = np.zeros_like(t, dtype=bool)
        for t_in, t_out in zip(t_joystick_in, t_joystick_out):
            joystick_present_mask[(t >= t_in) & (t <= t_out)] = True

        # get subepoch data
        f_starts, f_ends, t_starts, t_ends, se_ids, e_ids, is_reach, is_search, is_retract = (
            MovementSegmentation.SubEpoch
            * ReachEpoch.SubEpoch
            * RetractEpoch.SubEpoch
            * SearchEpoch.SubEpoch
            & key
        ).fetch("start_frame", "end_frame", "start_time", "end_time", \
                "sub_epoch_id", "epoch_id", \
                "is_reach", "is_search", "is_retract", \
                order_by=('epoch_id', 'sub_epoch_id')
        )

        # iterate over sub_epochs
        entries_epochs = [dict(**key, epoch_id=e_id) for e_id in np.unique(e_ids)]
        entries_sub_epochs = []
        for i, (eid, seid) in enumerate(zip(e_ids, se_ids)):
            # get sub_epoch data
            t_start, t_end = t_starts[i], t_ends[i]
            f_start, f_end = f_starts[i], f_ends[i]
            reach, search, retract = is_reach[i], is_search[i], is_retract[i]

            # compute joystick presence statistics
            joystick_frames = (joystick_present_mask[f_start:f_end]).sum()
            joystick_percentage = joystick_frames / (f_end - f_start) * 100

            # check if sub_epoch includes a joystick_out, with small tolerance
            joystick_out_included = [(t_start - 0.075 <= t_out <= t_end) for t_out in t_joystick_out]
            joystick_out_included = np.any(joystick_out_included)
            # check if sub_epoch starts in excluded time
            in_excluded_time = [(t_out <= t_start <= t_out + t_to_exclude) for t_out in t_joystick_out]
            in_excluded_time = np.any(in_excluded_time)
            if joystick_out_included or in_excluded_time:
                sub_epoch_class = 'excluded'
                entries_sub_epochs.append(dict(
                    **key, epoch_id=eid, sub_epoch_id=seid, sub_epoch_class=sub_epoch_class,
                    joystick_frames=joystick_frames, joystick_percentage=joystick_percentage
                ))
                continue

            # classify sub_epoch
            if reach:
                sub_epoch_class = 'reach'
            elif search:
                sub_epoch_class = 'search'
            elif retract:
                sub_epoch_class = 'retract'
            else:
                sub_epoch_class = 'other'
            
            # create entry
            entry_sub_epoch = dict(
                **key, epoch_id=eid, sub_epoch_id=seid, sub_epoch_class=sub_epoch_class,
                joystick_frames=joystick_frames, joystick_percentage=joystick_percentage
            )
            entries_sub_epochs.append(entry_sub_epoch)

        # insert data
        self.insert1(dict(**key, t_to_exclude=t_to_exclude))
        self.Epoch.insert(entries_epochs)
        self.SubEpoch.insert(entries_sub_epochs)

@schema
class EpochClassification(dj.Computed):
    definition = """ # classify epochs based on task-related outcomes
    -> SubEpochClassification
    """
    class Epoch(dj.Part):
        definition = """
        -> EpochClassification
        epoch_id            : int               # epoch identifier
        ---
        epoch_class         : enum('rewarded', 'miss', 'other', 'excluded')
        n_sub_epochs        : int               # number of sub-epochs
        n_reach             : int               # number of reach sub-epochs
        frames_reach        : int               # number of frames in reach sub-epochs
        n_search            : int               # number of search sub-epochs
        frames_search       : int               # number of frames in search sub-epochs
        n_retract           : int               # number of retract sub-epochs
        frames_retract      : int               # number of frames in retract sub-epochs
        n_excluded          : int               # number of excluded sub-epochs
        frames_excluded     : int               # number of frames in excluded sub-epochs
        n_other             : int               # number of other sub-epochs
        frames_other        : int               # number of frames in other sub-epochs
        n_joystick_frames   : int               # number of joystick frames
        """

    def make(self, key):
        # fetch rewarded timestamps
        t_rew = (exp.JoystickExperiment.Trials & dict(**key, successful=1)).fetch("t_servo_out")

        # fetch epoch ids
        epoch_keys, epoch_ids, start_times = (SubEpochClassification.Epoch * MovementSegmentation.Epoch & key).fetch("KEY", "epoch_id", "start_time", order_by="epoch_id")

        # get rewarded epochs
        rewarded_ids = []
        for t_r in t_rew:
            diff = t_r - start_times
            diff[diff <= 0.01] = np.nan
            if not np.isnan(diff).all():
                tr_idx = np.nanargmin(diff)
                epoch_id = epoch_ids[tr_idx]
                rewarded_ids.append(epoch_id)
            
        # iterate over epochs
        entries_epochs = []
        for epoch_key, epoch_id in zip(epoch_keys, epoch_ids):
            # fetch sub_epoch data
            sub_epoch_classes, joystick_frames, sub_epoch_frames = (SubEpochClassification.SubEpoch * MovementSegmentation.SubEpoch.proj(n_frames='end_frame-start_frame') & epoch_key).fetch("sub_epoch_class", "joystick_frames", "n_frames")
            # compute sub-epoch statistics
            n_sub_epochs = len(sub_epoch_classes)
            n_reach = np.sum(sub_epoch_classes == 'reach')
            frames_reach = np.sum(sub_epoch_frames[sub_epoch_classes == 'reach'])
            n_search = np.sum(sub_epoch_classes == 'search')
            frames_search = np.sum(sub_epoch_frames[sub_epoch_classes == 'search'])
            n_retract = np.sum(sub_epoch_classes == 'retract')
            frames_retract = np.sum(sub_epoch_frames[sub_epoch_classes == 'retract'])
            n_excluded = np.sum(sub_epoch_classes == 'excluded')
            frames_excluded = np.sum(sub_epoch_frames[sub_epoch_classes == 'excluded'])
            n_other = np.sum(sub_epoch_classes == 'other')
            frames_other = np.sum(sub_epoch_frames[sub_epoch_classes == 'other'])

            # classify epoch
            if epoch_id in rewarded_ids:
                epoch_class = 'rewarded'
            elif n_excluded > n_sub_epochs/3:
                epoch_class = 'excluded'
            elif n_reach + n_search > 0:
                epoch_class = 'miss'
            else:
                epoch_class = 'other'

            # create entries
            entry_epoch = dict(
                **epoch_key, epoch_class = epoch_class,
                n_sub_epochs = n_sub_epochs, n_reach = n_reach, frames_reach = frames_reach,
                n_search = n_search, frames_search = frames_search,
                n_retract = n_retract, frames_retract = frames_retract,
                n_excluded = n_excluded, frames_excluded = frames_excluded,
                n_other = n_other, frames_other = frames_other,
                n_joystick_frames = np.sum(joystick_frames),
            )
            entries_epochs.append(entry_epoch)
        
        # insert data
        self.insert1(key)
        self.Epoch.insert(entries_epochs)

@schema
class RestingMask(dj.Computed):
    definition = """ # compute resting mask
    -> MovementSegmentation
    -> JoystickOcclusion
    ---
    """
    class Hand(dj.Part):
        definition = """ # hand resting mask
        -> RestingMask
        hand                : enum('L', 'R')    # hand
        ---
        side                : enum('ipsi', 'contra')    # ipsilateral or contralateral to the task hand
        moving_mask         : longblob          # moving mask
        resting_mask        : longblob          # resting mask, including occlusion
        paw_still_mask      : longblob          # resting mask, excluding occlusion 
        percent_still       : float             # percentage of frames in still mask
        percent_resting     : float             # percentage of frames in resting mask
        """

    def make(self, key):
        # fetch hand keys
        hand_keys = (MovementSegmentation.Hand * JoystickOcclusion.Hand.proj() & key).fetch("KEY")
        # iterate over hands
        entries_hands = []
        for hand_key in hand_keys:
            side = (PawRecording.Hand & hand_key).fetch1("side")
            # fetch number of frames
            n_frames = (PawRecording.Hand & hand_key).fetch1("n_frames")
            # fetch joystick occlusion data
            rest_occupancy, search_occupancy, joystick_occupancy = (JoystickOcclusion.Hand & hand_key).fetch1("rest_occupancy", "search_occupancy", "joystick_occupancy")
            # fetch subepoch data
            f_starts, f_ends = (MovementSegmentation.Epoch & hand_key).fetch("start_frame", "end_frame", order_by='epoch_id')
            # create paw still mask
            moving_mask = np.zeros(n_frames, dtype=bool)
            for s, e in zip(f_starts, f_ends):
                moving_mask[s:e] = True
            paw_still_mask = ~moving_mask
            # create resting mask
            is_in_rest_zone = (rest_occupancy > joystick_occupancy + search_occupancy)
            resting_mask = is_in_rest_zone & paw_still_mask
            # compute quality control metrics
            percent_still = paw_still_mask.sum() / n_frames * 100
            percent_resting = resting_mask.sum() / n_frames * 100
            # create entry
            entry_hand = dict(
                **hand_key, side=side,
                moving_mask=moving_mask, resting_mask=resting_mask, paw_still_mask=paw_still_mask,
                percent_still=percent_still, percent_resting=percent_resting
            )
            entries_hands.append(entry_hand)
        
        # insert data
        self.insert1(key)
        self.Hand.insert(entries_hands)

@schema
class HoldEpoch(dj.Computed):
    definition = """ # compute hold epochs
    -> JoystickOcclusion.Hand
    -> RestingMask.Hand
    -> exp.JoystickPresence
    -> Synchronisation.Hand
    -> MovementSegmentationParams
    ---
    hold_mask            : longblob          # hold mask
    deflection_threshold : float             # threshold for deflection mask
    """
    _key_source = (
        JoystickOcclusion.Hand 
        * RestingMask.Hand.proj() 
        * exp.JoystickPresence 
        * Synchronisation.Hand.proj()
        * MovementSegmentationParams) & "side='ipsi'" # only ipsilateral hand
    
    class Epoch(dj.Part):
        definition = """ # holding epochs, distinct from MovementSegmentation epochs)
        -> HoldEpoch
        hold_epoch_id       : int               # hold epoch identifier
        ---
        start_frame         : int               # start frame
        end_frame           : int               # end frame (exclusive)
        start_time          : float             # start time
        end_time            : float             # end time (exclusive) 
        """
    
    def make(self, key):
        deflection_threshold = -1.63 # threshold for deflection mask

        # fetch joystick presence mask
        t_joystick_in, t_joystick_out = (exp.JoystickPresence.Trial & key).fetch("t_joystick_in", "t_joystick_out", order_by="trial_id")
        t = (Synchronisation.Hand & key).fetch1("frame_timestamps")
        joystick_present_mask = np.zeros_like(t, dtype=bool)
        for t_in, t_out in zip(t_joystick_in, t_joystick_out):
            joystick_present_mask[(t >= t_in) & (t <= t_out)] = True

        # fetch hand movement data
        still_mask = (RestingMask.Hand & key).fetch1("paw_still_mask")

        # fetch joystick occlusion data
        rest_occupancy, search_occupancy, joystick_occupancy = (JoystickOcclusion.Hand & key).fetch1("rest_occupancy", "search_occupancy", "joystick_occupancy")
        joystick_location_mask = (joystick_occupancy >= rest_occupancy) & (joystick_occupancy >= search_occupancy)
        
        # compute deflection mask
        t_joystick, y_joystick = (exp.JoystickReadouts.Data & key).fetch1("t", "y")
        y_joystick_resample = np.interp(t, t_joystick, y_joystick)
        deflection_mask = y_joystick_resample < deflection_threshold

        # compute hold mask and fill holes
        hold_mask = (joystick_location_mask | deflection_mask) & still_mask & joystick_present_mask
        structure = (MovementSegmentationParams & key).fetch1("structure")
        fps = (PawRecording.Hand & key).fetch1("fps")
        structure = int(structure * fps)
        hold_mask = binary_closing(hold_mask, structure=np.ones([structure]))

        # segment into epochs
        epochs_entries = []
        hold_mask_labeled, n_components = label(hold_mask)
        for i in range(1, n_components+1):
            f_start, f_end = np.nonzero(hold_mask_labeled == i)[0][[0,-1]]
            t_start, t_end = t[f_start], t[f_end]
            # create entry
            entry_epoch = dict(
                **key, hold_epoch_id=i, start_frame=f_start, end_frame=f_end+1,
                start_time=t_start, end_time=t_end
            )
            epochs_entries.append(entry_epoch)
        
        # insert data
        self.insert1(dict(**key, deflection_threshold=deflection_threshold, hold_mask=hold_mask))
        self.Epoch.insert(epochs_entries)
