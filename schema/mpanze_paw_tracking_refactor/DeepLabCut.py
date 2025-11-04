"""
DeepLabCut processing pipeline
"""

# connect to the database
import login
login.connect()

# imports
import datajoint as dj
import numpy as np
from scipy.signal import savgol_filter

# optional imports
from util.optional_import import import_optional
dlc = import_optional('deeplabcut', behavior='warn')

# import table dependencies
from schema.mpanze_paw_tracking_refactor.PawRecording import PawRecording, JoystickPosition

# instantiate the schema
schema = dj.schema('mpanze_paw_tracking_refactor', locals(), create_tables=True)

@schema
class DeepLabCut(dj.Computed):
    definition = """ # deeplabcut processing results
    -> PawRecording
    ---
    dlc_model_path          : varchar(512)      # relative path to the deeplabcut config file
    iteration               : int               # iteration of the deeplabcut model
    time_insert=CURRENT_TIMESTAMP : timestamp       # automatically inserted timestamp
    """

    class Hand(dj.Part):
        definition = """ # results for each hand
        -> DeepLabCut
        hand                : enum('L', 'R')            # left or right forelimb of the mouse
        ---
        side                : enum('ipsi', 'contra')    # ipsilateral or contralateral to the task hand
        time_insert_hand=CURRENT_TIMESTAMP : timestamp       # automatically inserted timestamp
        """

        def get_path(self, as_posix=False):
            assert len(self) == 1, "only one entry allowed"
            p_vid = (PawRecording.Hand & self.proj()).get_path(as_posix=False)
            p_vid = list(p_vid.parent.glob(p_vid.stem+"DLC_*.mp4"))[0]
            return p_vid.as_posix() if as_posix else p_vid

    class Label(dj.Part):
        definition = """ # coordinates of each labelled body part
        -> DeepLabCut.Hand
        label               : varchar(32)             # name of the labelled body part
        ---
        x                   : longblob                # x coordinates of the labelled body part
        y                   : longblob                # y coordinates of the labelled body part
        p                   : longblob                # confidence of the labelled body part
        """

    def make(self, key):
        raise NotImplementedError("DeepLabCut processing with new version is not implemented yet")
    
@schema
class FilteredDLC(dj.Computed):
    definition = """ # interpolates and filters the DLC data
    -> DeepLabCut
    ---
    p_cutoff            : float         # cutoff for the confidence of the labelled body part
    savgol_window       : float         # window size for the Savitzky-Golay filter (in seconds)
    savgol_order        : float         # order of the Savitzky-Golay filter
    """
    
    class Hand(dj.Part):
        definition = """ # results for each hand
        -> FilteredDLC
        hand                : enum('L', 'R')            # left or right forelimb of the mouse
        ---
        side                : enum('ipsi', 'contra')    # ipsilateral or contralateral to the task hand
        """

        def fetch_coordinate_matrix(self):
            key = self.fetch1("KEY")
            x, y, label = (FilteredDLC.Label & key).fetch("x", "y", "label", order_by="label")
            names = [f"{l}_x" for l in label] + [f"{l}_y" for l in label]
            return np.stack([*x, *y]).T.astype(np.float32), names
        
    class Label(dj.Part):
        definition = """ # coordinates of each labelled body part
        -> FilteredDLC.Hand
        label               : varchar(32)             # name of the labelled body part
        ---
        x                   : longblob                # x coordinates of the labelled body part
        y                   : longblob                # y coordinates of the labelled body part
        """
    
    def make(self, key):
        # hardcoded params
        p_cutoff = 0.5
        savgol_window = 0.075  # 75ms window
        savgol_order = 2
        entry_master = dict(**key, p_cutoff=p_cutoff, savgol_window=savgol_window, savgol_order=savgol_order)
    
        # iterate over DLC data
        entries_hand = []
        entries_label = []
        for key_hand in (DeepLabCut.Hand & key).fetch("KEY"):
            side = (DeepLabCut.Hand & key_hand).fetch1("side")
            entries_hand.append(dict(**key_hand, side=side))

            # get window in frames
            fps, pixels_w, pixels_h = (PawRecording.Hand & key_hand).fetch1("fps", "pixels_w", "pixels_h")
            window = int(savgol_window * fps)

            # iterate over labels
            for key_label in (DeepLabCut.Label & key_hand).fetch("KEY"):
                x, y, p = (DeepLabCut.Label & key_label).fetch1("x", "y", "p")
                # threshold data
                t = np.arange(len(x)) / fps
                x_thresh = x[p > p_cutoff]
                y_thresh = y[p > p_cutoff]
                t_thresh = t[p > p_cutoff]
                # interpolate data
                x_interp = np.interp(t, t_thresh, x_thresh)
                y_interp = np.interp(t, t_thresh, y_thresh)
                # filter data
                x_filt = savgol_filter(x_interp, window, savgol_order)
                y_filt = savgol_filter(y_interp, window, savgol_order)
                # clip data
                x_filt = np.clip(x_filt, a_min=0, a_max=pixels_w-1).astype(np.float32)
                y_filt = np.clip(y_filt, a_min=0, a_max=pixels_h-1).astype(np.float32)
                # append entries
                entries_label.append(dict(**key_label, x=x_filt, y=y_filt))

        # insert entries
        self.insert1(entry_master)
        self.Hand.insert(entries_hand)
        self.Label.insert(entries_label)

@schema
class WeightedHandPosition(dj.Computed):
    definition = """ # probability-weighted hand position
    -> DeepLabCut
    ---
    """

    class Hand(dj.Part):
        definition = """ # results for each hand
        -> WeightedHandPosition
        hand                : enum('L', 'R')            # left or right forelimb of the mouse
        ---
        side                : enum('ipsi', 'contra')    # ipsilateral or contralateral to the task hand
        x                   : longblob                  # x coordinates of the hand
        y                   : longblob                  # y coordinates of the hand
        """
        def fetch_velocity_1(self, norm=False):
            assert len(self) == 1, "only one entry allowed"
            # fetch coordinates
            x, y = self.fetch1("x", "y")
            coords = np.stack([x,y]).T
            v = np.gradient(coords, axis=0)
            return np.linalg.norm(v, axis=1) if norm else v
    
    def make(self, key):
        # process each hand
        entries_hand = []
        for hand_key in (DeepLabCut.Hand & key).fetch("KEY"):
            # get data
            x, y, p = (DeepLabCut.Label & "label!='elbow'" & hand_key).fetch("x", "y", "p")
            x = np.stack(x).T
            y = np.stack(y).T
            p = np.stack(p).T
            pixels_h, pixels_w = (PawRecording.Hand & hand_key).fetch1("pixels_h", "pixels_w")

            # compute weighted mean
            x = np.average(x, axis=1, weights=p).squeeze()
            y = np.average(y, axis=1, weights=p).squeeze()

            # clip data
            x = np.clip(x, a_min=0, a_max=pixels_w-1).astype(np.float32)
            y = np.clip(y, a_min=0, a_max=pixels_h-1).astype(np.float32)

            # append entries
            entry_hand = dict(**hand_key, x=x, y=y)
            entries_hand.append(entry_hand)

        # insert data
        self.insert1(key)
        self.Hand.insert(entries_hand)

@schema
class Features(dj.Computed):
    definition = """ # computes continuous movement features from deeplabcut traces
    -> FilteredDLC
    -> JoystickPosition
    -> WeightedHandPosition
    ---
    """
    
    class Hand(dj.Part):
        definition = """ # movement features for individual hands
        -> Features
        hand            : enum('L', 'R')    # limb
        ---
        side            : enum('ipsi', 'contra')    # ipsilateral or contralateral to the task hand
        """
        def fetch_feature_matrix(self, labels_to_include=None):
            key = self.fetch1("KEY")
            if labels_to_include is not None:
                restriction = [f"label='{label}'" for label in labels_to_include]
                features, names = (Features.Feature & key & restriction).fetch("feature", "label", order_by="label")
            else:
                features, names = (Features.Feature & key).fetch("feature", "label", order_by="label")
            features = np.stack(features).T.astype(np.float32)
            return features, names

    class Feature(dj.Part):
        definition = """ # movement features
        -> Features.Hand
        label           : varchar(128)      # name of the feature
        ---
        feature         : longblob          # 1-D array of the feature
        """
    
    @staticmethod
    def coarsefeatures(key):
        data = {}
        # compute features from weighted hand position
        x, y = (WeightedHandPosition.Hand & key).fetch1("x", "y")
        xj, yj = (JoystickPosition.Hand & key).fetch1("x", "y")

        # filter x, y
        fps = (PawRecording.Hand & key).fetch1("fps")
        savgol_window, savgol_order = (FilteredDLC & key).fetch1("savgol_window", "savgol_order")
        window = int(savgol_window * fps)
        x = savgol_filter(x, window, int(savgol_order))
        y = savgol_filter(y, window, int(savgol_order))

        # compute distance to joystick
        dx = x - xj
        dy = y - yj
        d = np.sqrt(dx**2 + dy**2)
        data["distance_x"] = dx.astype(np.float32)
        data["distance_y"] = dy.astype(np.float32)
        data["distance"] = d.astype(np.float32)
        # compute velocity, acceleration
        vx = np.gradient(x)
        vy = np.gradient(y)
        v = np.sqrt(vx**2 + vy**2)
        ax = np.gradient(vx)
        ay = np.gradient(vy)
        a = np.sqrt(ax**2 + ay**2)
        data["velocity_x"] = vx.astype(np.float32)
        data["velocity_y"] = vy.astype(np.float32)
        data["velocity"] = v.astype(np.float32)
        data["acceleration_x"] = ax.astype(np.float32)
        data["acceleration_y"] = ay.astype(np.float32)
        data["acceleration"] = a.astype(np.float32)
        return data
    
    @staticmethod
    def finefeatures(key):
        # compute bending of fingers
        data = {}
        for i in range(1,5):
            k = np.stack((FilteredDLC.Label & dict(**key, label=f"{i}_knuckle")).fetch1("x", "y")).T
            m = np.stack((FilteredDLC.Label & dict(**key, label=f"{i}_mid")).fetch1("x", "y")).T
            t = np.stack((FilteredDLC.Label & dict(**key, label=f"{i}_tip")).fetch1("x", "y")).T
            # compute bend, rotation
            km = 90-np.rad2deg(np.arctan2(m[:,1]-k[:,1], m[:,0]-k[:,0]))
            mt = 90-np.rad2deg(np.arctan2(t[:,1]-m[:,1], t[:,0]-m[:,0]))
            # quadrant correction - flip point is at -90/+270, realign it to 180
            km[km >= 180] -= 360
            mt[mt >= 180] -= 360
            bend = mt - km
            #rotation = (mt + km)/2
            rotation = km
            data[f"bend_{i}"] = bend.astype(np.float32)
            data[f"bend_alt_{i}"] = np.abs(bend).astype(np.float32)
            data[f"rotation_{i}"] = rotation.astype(np.float32)
        
        # compute opening
        for i in range(1,4):
            op = data[f"rotation_{i+1}"] - data[f"rotation_{i}"]
            data[f"open_{i}"] = np.abs(op)
        # compute alternate opening
        data["open_alt_24"] = np.abs(data["rotation_4"] - data["rotation_2"])
        
        # compute finger 2-4 averages
        data['bend_24'] = np.stack([data[f"bend_{i}"] for i in range(2,5)]).mean(axis=0)
        data['bend_alt_24'] = np.stack([data[f"bend_alt_{i}"] for i in range(2,5)]).mean(axis=0)
        data['rotation_24'] = np.stack([data[f"rotation_{i}"] for i in range(2,5)]).mean(axis=0)
        data['open_24'] = np.stack([data[f"open_{i}"] for i in range(2,4)]).mean(axis=0)
        return data

    def make(self, key):
        # iterate over data
        entries_hand = []
        entries_feature = []
        for key_hand in (FilteredDLC.Hand & key).fetch("KEY"):
            # insert hand entry
            side = (FilteredDLC.Hand & key_hand).fetch1("side")
            entries_hand.append(dict(**key_hand, side=side))

            # compute features
            data_fine = self.finefeatures(key_hand)
            data_coarse = self.coarsefeatures(key_hand)
            data = {**data_fine, **data_coarse}

            # insert data
            for label, feature in data.items():
                entries_feature.append(dict(**key_hand, label=label, feature=feature))

        # insert data
        self.insert1(key)
        Features.Hand.insert(entries_hand)
        Features.Feature.insert(entries_feature)

def import_from_old_pipeline():
    from tqdm import tqdm
    from schema import mpanze_paw_tracking as pt

    keys = PawRecording.fetch("KEY")
    for key in tqdm(keys):
        dlc_entry = (pt.DeepLabCut & key).fetch1()
        hand_entries = (pt.DeepLabCut.Hand & key).fetch(as_dict=True)
        label_entries = (pt.DeepLabCut.Label & key).fetch(as_dict=True)
        for entry in label_entries:
            entry.pop("file_id")
            entry.pop("time_insert_label")
        DeepLabCut.insert1(dlc_entry, allow_direct_insert=True)
        DeepLabCut.Hand.insert(hand_entries, allow_direct_insert=True)
        DeepLabCut.Label.insert(label_entries, allow_direct_insert=True)
