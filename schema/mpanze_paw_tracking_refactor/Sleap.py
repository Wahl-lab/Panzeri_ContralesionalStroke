"""
Keypoint tracking using SLEAP
"""

# database connection and imports
import login
login.connect()
import datajoint as dj

# imports
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import binary_closing, label

# optional imports
try:
    import sleap_io as io
except ImportError:
    print("sleap_io not found, SLEAP functionality will not work. Please install sleap_io to use this feature.")

# import table dependencies
from schema.mpanze_paw_tracking_refactor.PawRecording import PawRecording, Synchronisation, JoystickPosition
from schema.mpanze_exp_refactor import mpanze_exp_refactor as exp
from schema.mpanze_paw_tracking_refactor.DeepLabCut import WeightedHandPosition

# instantiate the schema
schema = dj.schema('mpanze_paw_tracking_refactor', locals(), create_tables=True)

@schema
class Sleap(dj.Computed):
    definition = """ # keypoint tracking using SLEAP
    -> PawRecording
    ---
    sleap_model_dir     : varchar(255)  # absolute path to the directory containing the SLEAP model
    batch_size         : int            # batch size used for inference
    max_instances      : int            # maximum number of instances to track per frame
    peak_threshold     : float          # threshold for peak detection
    time_insert=CURRENT_TIMESTAMP : timestamp # automatic timestamp for when the entry was inserted
    """

    class Hand(dj.Part):
        definition = """ # results for each hand
        -> Sleap
        hand            : enum('L', 'R')  # which hand is used
        ---
        side            : enum('ipsi', 'contra')  # whether the hand is dominant (ipsi) or non-dominant (contra)
        """

        def get_path(self, as_posix=False):
            """get path to the rendered video for this hand"""
            raise NotImplementedError("This method is not implemented yet")
    
    class Label(dj.Part):
        definition = """ # coordinates for each labelled body part
        -> Sleap.Hand
        label            : varchar(32)       # name of the body part
        ---
        x                : longblob           # x coordinates of the body part
        y                : longblob           # y coordinates of the body part
        p                : longblob           # confidence of the body part detection (0 or 1, just here for compatibility with DLC)
        """
    
    def make(self, key):
        # various hardcoded parameters
        sleap_model_dir = '/home/ubuntu/neurophys_3/paw_tracking/exports/prod_v1'  # hardcoded for use on sciencecloud
        batch_size = 16
        max_instances = 1       # one paw per frame
        peak_threshold = 0.15   # threshold for peak detection

        # define label mapping from sleap to dlc format
        label_mapping = {
            'wrist': 'wrist',
            'elbow': 'elbow',
            'knuckle_1': '1_knuckle',
            'knuckle_2': '2_knuckle',
            'knuckle_3': '3_knuckle',
            'knuckle_4': '4_knuckle',
            'mid_1': '1_mid',
            'mid_2': '2_mid',
            'mid_3': '3_mid',
            'mid_4': '4_mid',
            'tip_1': '1_tip',
            'tip_2': '2_tip',
            'tip_3': '3_tip',
            'tip_4': '4_tip',
            'support' : 'paw_2'
        }

        # sanity checks
        assert login.get_computer_name() == "science_cloud_sleap_instance"        # for now we run this only on science cloud
        from schema.common_mice import Mouse
        batch = (Mouse() & key).fetch1('batch')
        assert batch >= 7, "batches before 7 are not supported"

        # iterate over both hands
        entries_hand = []
        entries_label = []
        for hand_key in (PawRecording.Hand & key).fetch("KEY"):
            # get side
            side = (PawRecording.Hand & hand_key).fetch1('side')
            entries_hand.append(dict(**hand_key, side=side))

            # get number of frames in video for sanity checks
            n_frames = (PawRecording.Hand & hand_key).fetch1('n_frames')

            # get video path
            p_video = (PawRecording.Hand & hand_key).get_path()

            # generate output path
            p_output = p_video.with_suffix('.predictions.slp')

            # run sleap inference via the command line interface
            cmd = f'sleap-nn predict {sleap_model_dir} {p_video.resolve().as_posix()} -o {p_output.resolve().as_posix()} '\
            f'--runtime tensorrt --batch-size {batch_size} --peak-conf-threshold {peak_threshold} --max-instances {max_instances}'
            import subprocess
            subprocess.run(cmd, shell=True, check=True)

            # load the predictions
            predictions = io.load_file(p_output.resolve().as_posix())
            # convert to dataframe
            df = (
                predictions
                .to_dataframe()
                .drop(columns=["track", "track_score", "instance_score", "score"])
            )

            for node, df_node in df.groupby("node"):
                df_to_enter = df_node.drop(columns=["node"]).set_index("frame_idx").sort_index()
                # fill missing frames with NaN
                df_to_enter = df_to_enter.reindex(np.arange(n_frames), fill_value=np.nan)

                # compute confidence as 1 if x is not NaN, else 0
                df_node["p"] = 1 - df_node["x"].isna().astype(float)  # confidence is 1 if x is not NaN, else 0  
                df_node = df_node.fillna(0)  # replace NaN with 0 for x and y

                # create entry
                entry = dict(
                    **hand_key,
                    label=label_mapping[node],
                    x=df_node.x.to_numpy().astype(np.float32),
                    y=df_node.y.to_numpy().astype(np.float32),
                    p=df_node.p.to_numpy().astype(np.float32)
                )
                entries_label.append(entry)

        # insert into the database
        self.insert1(dict(**key, sleap_model_dir=sleap_model_dir, batch_size=batch_size, max_instances=max_instances, peak_threshold=peak_threshold))
        self.Hand.insert(entries_hand)
        self.Label.insert(entries_label)

@schema
class RenderedVideo(dj.Computed):
    definition = """ # renders of the videos with the tracked keypoints overlaid, for quality control
    -> Sleap.Hand
    ---
    """
    def make(self, key):
        p_video = (PawRecording.Hand & key).get_path()
        p_slp = p_video.with_suffix('.predictions.slp')

        # generate output path
        p_out = p_video.with_suffix('.rendered.mp4')

        # run sleap render via the command line interface
        cmd = f'sleap-io render -i {p_slp.resolve().as_posix()} -o {p_out.resolve().as_posix()}'\
        f' --preset draft --crf 15'
        # run command
        print(f"Running command: {cmd}")
        import subprocess
        subprocess.run(cmd, shell=True, check=True)

        # sanity check that the output video was created
        assert p_out.exists()

        # insert into the database
        self.insert1(key)

@schema
class FilteredSleap(dj.Computed):
    definition = """ # interpolates and filters the SLEAP data
    -> Sleap
    ---
    p_cutoff            : float         # cutoff for the confidence of the labelled body part
    savgol_window       : float         # window size for the Savitzky-Golay filter (in seconds)
    savgol_order        : float         # order of the Savitzky-Golay filter
    """
    
    class Hand(dj.Part):
        definition = """ # results for each hand
        -> FilteredSleap
        hand                : enum('L', 'R')            # left or right forelimb of the mouse
        ---
        side                : enum('ipsi', 'contra')    # ipsilateral or contralateral to the task hand
        """

        def fetch_coordinate_matrix(self):
            key = self.fetch1("KEY")
            x, y, label = (FilteredSleap.Label & key).fetch("x", "y", "label", order_by="label")
            names = [f"{l}_x" for l in label] + [f"{l}_y" for l in label]
            return np.stack([*x, *y]).T.astype(np.float32), names
        
    class Label(dj.Part):
        definition = """ # coordinates of each labelled body part
        -> FilteredSleap.Hand
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
        for key_hand in (Sleap.Hand & key).fetch("KEY"):
            side = (Sleap.Hand & key_hand).fetch1("side")
            entries_hand.append(dict(**key_hand, side=side))

            # get window in frames
            fps, pixels_w, pixels_h = (PawRecording.Hand & key_hand).fetch1("fps", "pixels_w", "pixels_h")
            window = int(savgol_window * fps)

            # iterate over labels - exclude elbow as tracking is often really poor
            for key_label in (Sleap.Label & key_hand & "label != 'elbow'").fetch("KEY"):
                x, y, p = (Sleap.Label & key_label).fetch1("x", "y", "p")
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
class HandPositionSleap(dj.Computed):
    definition = """ # averagehand position
    -> FilteredSleap
    ---
    """

    class Hand(dj.Part):
        definition = """ # results for each hand
        -> HandPositionSleap
        hand                : enum('L', 'R')            # left or right forelimb of the mouse
        ---
        side                : enum('ipsi', 'contra')    # ipsilateral or contralateral to the task hand
        x                   : longblob                  # x coordinates of the hand
        y                   : longblob                  # y coordinates of the hand
        corrcoef_x=NULL     : float                     # correlation coefficient with dlc pipeline
        corrcoef_y=NULL     : float                     # correlation coefficient with dlc pipeline
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
        for hand_key in (FilteredSleap.Hand & key).fetch("KEY"):
            # get data
            x, y = (FilteredSleap.Label & "label!='elbow'" & hand_key).fetch("x", "y")
            x = np.stack(x).T
            y = np.stack(y).T
            pixels_h, pixels_w = (PawRecording.Hand & hand_key).fetch1("pixels_h", "pixels_w")

            # compute weighted mean
            x = np.average(x, axis=1).squeeze()
            y = np.average(y, axis=1).squeeze()

            # clip data
            x = np.clip(x, a_min=0, a_max=pixels_w-1).astype(np.float32)
            y = np.clip(y, a_min=0, a_max=pixels_h-1).astype(np.float32)

            # if x and y are available in dlc, compute correlation coefficient
            if len(WeightedHandPosition.Hand & hand_key) == 1:
                x_dlc, y_dlc = (WeightedHandPosition.Hand & hand_key).fetch1("x", "y")
                corrcoef_x = np.corrcoef(x, x_dlc)[0, 1]
                corrcoef_y = np.corrcoef(y, y_dlc)[0, 1]
            else:
                corrcoef_x = np.nan
                corrcoef_y = np.nan

            # append entries
            entry_hand = dict(**hand_key, x=x, y=y, corrcoef_x=corrcoef_x, corrcoef_y=corrcoef_y)
            entries_hand.append(entry_hand)

        # insert data
        self.insert1(key)
        self.Hand.insert(entries_hand)


@schema
class FeaturesSleap(dj.Computed):
    definition = """ # computes continuous movement features from SLEAP traces
    -> FilteredSleap
    -> JoystickPosition
    -> HandPositionSleap
    ---
    """

    class Hand(dj.Part):
        definition = """ # movement features for individual hands
        -> FeaturesSleap
        hand            : enum('L', 'R')
        ---
        side            : enum('ipsi', 'contra')
        """

        def fetch_feature_matrix(self, labels_to_include=None):
            key = self.fetch1("KEY")
            if labels_to_include is not None:
                restriction = [f"label='{label}'" for label in labels_to_include]
                features, names = (FeaturesSleap.Feature & key & restriction).fetch("feature", "label", order_by="label")
            else:
                features, names = (FeaturesSleap.Feature & key).fetch("feature", "label", order_by="label")
            features = np.stack(features).T.astype(np.float32)
            return features, names

    class Feature(dj.Part):
        definition = """ # movement features
        -> FeaturesSleap.Hand
        label           : varchar(128)
        ---
        feature         : longblob
        """

    @staticmethod
    def coarsefeatures(key):
        data = {}
        x, y = (HandPositionSleap.Hand & key).fetch1("x", "y")
        xj, yj = (JoystickPosition.Hand & key).fetch1("x", "y")

        fps = (PawRecording.Hand & key).fetch1("fps")
        savgol_window, savgol_order = (FilteredSleap & key).fetch1("savgol_window", "savgol_order")
        window = int(savgol_window * fps)
        x = savgol_filter(x, window, int(savgol_order))
        y = savgol_filter(y, window, int(savgol_order))

        dx = x - xj
        dy = y - yj
        d = np.sqrt(dx**2 + dy**2)
        data["distance_x"] = dx.astype(np.float32)
        data["distance_y"] = dy.astype(np.float32)
        data["distance"] = d.astype(np.float32)
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
        data = {}
        for i in range(1, 5):
            k = np.stack((FilteredSleap.Label & dict(**key, label=f"{i}_knuckle")).fetch1("x", "y")).T
            m = np.stack((FilteredSleap.Label & dict(**key, label=f"{i}_mid")).fetch1("x", "y")).T
            t = np.stack((FilteredSleap.Label & dict(**key, label=f"{i}_tip")).fetch1("x", "y")).T
            km = 90 - np.rad2deg(np.arctan2(m[:, 1] - k[:, 1], m[:, 0] - k[:, 0]))
            mt = 90 - np.rad2deg(np.arctan2(t[:, 1] - m[:, 1], t[:, 0] - m[:, 0]))
            km[km >= 180] -= 360
            mt[mt >= 180] -= 360
            bend = mt - km
            rotation = km
            data[f"bend_{i}"] = bend.astype(np.float32)
            data[f"bend_alt_{i}"] = np.abs(bend).astype(np.float32)
            data[f"rotation_{i}"] = rotation.astype(np.float32)

        for i in range(1, 4):
            data[f"open_{i}"] = np.abs(data[f"rotation_{i+1}"] - data[f"rotation_{i}"])
        data["open_alt_24"] = np.abs(data["rotation_4"] - data["rotation_2"])
        data["bend_24"] = np.stack([data[f"bend_{i}"] for i in range(2, 5)]).mean(axis=0)
        data["bend_alt_24"] = np.stack([data[f"bend_alt_{i}"] for i in range(2, 5)]).mean(axis=0)
        data["rotation_24"] = np.stack([data[f"rotation_{i}"] for i in range(2, 5)]).mean(axis=0)
        data["open_24"] = np.stack([data[f"open_{i}"] for i in range(2, 4)]).mean(axis=0)
        return data

    def make(self, key):
        entries_hand = []
        entries_feature = []
        for key_hand in (FilteredSleap.Hand & key).fetch("KEY"):
            side = (FilteredSleap.Hand & key_hand).fetch1("side")
            entries_hand.append(dict(**key_hand, side=side))

            data = {**self.finefeatures(key_hand), **self.coarsefeatures(key_hand)}
            for label, feature in data.items():
                entries_feature.append(dict(**key_hand, label=label, feature=feature))

        self.insert1(key)
        self.Hand.insert(entries_hand)
        self.Feature.insert(entries_feature)


@schema
class MovementSegmentationParamsSleap(dj.Lookup):
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
class MovementSegmentationSleap(dj.Computed):
    definition = """
    -> HandPositionSleap
    -> MovementSegmentationParamsSleap
    """

    class Hand(dj.Part):
        definition = """
        -> MovementSegmentationSleap
        hand                : enum('L', 'R')    # hand
        ---
        side                : enum('ipsi', 'contra')    # ipsilateral or contralateral to the task hand
        """

        def epoch_features(self, subtract_baseline=False, labels_to_include=None):
            from tqdm import tqdm
            import pandas as pd
            data = []
            for s in tqdm(self.fetch("KEY"), desc='fetching epoch features'):
                df = (self & s).epoch_features_1(subtract_baseline=subtract_baseline, labels_to_include=labels_to_include)
                data.append(df)
            return pd.concat(data)

        def epoch_features_1(self, subtract_baseline=False, labels_to_include=None):
            assert len(self) == 1, "only one entry allowed"
            if labels_to_include is not None:
                restriction = [f"label = '{l}'" for l in labels_to_include]
                names, features = (FeaturesSleap.Feature & self.proj() & restriction).fetch("label", "feature")
            else:
                names, features = (FeaturesSleap.Feature & self.proj()).fetch("label", "feature")
            features = np.stack(features).T
            names_min = [f"{name}_min" for name in names]
            names_max = [f"{name}_max" for name in names]
            names_mean = [f"{name}_mean" for name in names]
            names = ["epoch_id"] + names_min + names_max + names_mean

            e_ids, starts, ends = (MovementSegmentationSleap.Epoch & self.proj()).fetch("epoch_id", "start_frame", "end_frame", order_by="epoch_id")
            n_frames = (PawRecording.Hand & self.proj()).fetch1("n_frames")
            epoch_features = []

            if subtract_baseline:
                rest_mask = (RestingMaskSleap.Hand & self.proj()).fetch1("resting_mask")
                rest = np.median(features[rest_mask], axis=0)

            for eid, s, e in zip(e_ids, starts, ends):
                mask = MovementSegmentationSleap.frames_to_mask(n_frames, s, e)
                maxs = features[mask].max(axis=0)
                mins = features[mask].min(axis=0)
                means = features[mask].mean(axis=0)
                if subtract_baseline:
                    maxs -= rest
                    mins -= rest
                    means -= rest
                epoch_features.append(np.concatenate(((eid,), mins, maxs, means)))
            epoch_features = np.stack(epoch_features)
            import pandas as pd
            df = pd.DataFrame(epoch_features, columns=names)
            df = df.set_index("epoch_id")
            df_session = (MovementSegmentationSleap.Epoch.proj() & self.proj()).fetch(format='frame')
            return df_session.join(df)

        def sub_epoch_features(self, subtract_baseline=False, labels_to_include=None):
            from tqdm import tqdm
            import pandas as pd
            data = []
            for s in tqdm(self.fetch("KEY"), desc='fetching subepoch features'):
                df = (self & s).sub_epoch_features_1(subtract_baseline=subtract_baseline, labels_to_include=labels_to_include)
                data.append(df)
            return pd.concat(data)

        def sub_epoch_features_1(self, subtract_baseline=False, labels_to_include=None):
            assert len(self) == 1, "only one entry allowed"
            if labels_to_include is not None:
                restriction = [f"label = '{l}'" for l in labels_to_include]
                names, features = (FeaturesSleap.Feature & self.proj() & restriction).fetch("label", "feature")
            else:
                names, features = (FeaturesSleap.Feature & self.proj()).fetch("label", "feature")
            features = np.stack(features).T
            names_min = [f"{name}_min" for name in names]
            names_max = [f"{name}_max" for name in names]
            names_mean = [f"{name}_mean" for name in names]
            names = ["epoch_id", "sub_epoch_id"] + names_min + names_max + names_mean

            e_ids, se_ids, starts, ends = (MovementSegmentationSleap.SubEpoch & self.proj()).fetch("epoch_id", "sub_epoch_id", "start_frame", "end_frame", order_by=("epoch_id", "sub_epoch_id"))
            n_frames = (PawRecording.Hand & self.proj()).fetch1("n_frames")
            subepoch_features = []

            if subtract_baseline:
                rest_mask = (RestingMaskSleap.Hand & self.proj()).fetch1("resting_mask")
                rest = np.median(features[rest_mask], axis=0)

            for eid, seid, s, e in zip(e_ids, se_ids, starts, ends):
                mask = MovementSegmentationSleap.frames_to_mask(n_frames, s, e)
                maxs = features[mask].max(axis=0)
                mins = features[mask].min(axis=0)
                means = features[mask].mean(axis=0)
                if subtract_baseline:
                    maxs -= rest
                    mins -= rest
                    means -= rest
                subepoch_features.append(np.concatenate(((eid, seid), mins, maxs, means)))
            subepoch_features = np.stack(subepoch_features)
            import pandas as pd
            df = pd.DataFrame(subepoch_features, columns=names)
            df = df.set_index(["epoch_id", "sub_epoch_id"])
            df_session = (MovementSegmentationSleap.SubEpoch.proj() & self.proj()).fetch(format='frame')
            return df_session.join(df)

    class Epoch(dj.Part):
        definition = """ # segmentation into epochs
        -> MovementSegmentationSleap.Hand
        epoch_id            : int               # epoch identifier
        ---
        start_frame         : int               # start frame
        end_frame           : int               # end frame (exclusive)
        start_time          : float             # start time
        end_time            : float             # end time (exclusive)
        """

    class SubEpoch(dj.Part):
        definition = """
        -> MovementSegmentationSleap.Epoch
        sub_epoch_id         : int               # sub-epoch identifier (unique within the epoch)
        ---
        global_id            : int               # global epoch identifier (unique within the session)
        start_frame          : int               # start frame
        end_frame            : int               # end frame (exclusive)
        start_time           : float             # start time
        end_time             : float             # end time (exclusive)
        """

    def make(self, key):
        params = (MovementSegmentationParamsSleap & key).fetch1()
        entries_hands = []
        entries_epochs = []
        entries_sub_epochs = []
        hand_keys = (HandPositionSleap.Hand & key).fetch("KEY")
        for hand_key in hand_keys:
            hand_key = {**key, **hand_key}
            side = (PawRecording.Hand & hand_key).fetch1("side")
            entries_hands.append(dict(**hand_key, side=side))

            fps = (PawRecording.Hand & hand_key).fetch1("fps")
            t = (Synchronisation.Hand & hand_key).fetch1("frame_timestamps")

            v = (HandPositionSleap.Hand & hand_key).fetch_velocity_1(norm=True)
            savgol_window = int(params["savgol_window"] * fps)
            v = savgol_filter(v, window_length=savgol_window, polyorder=params["savgol_order"])
            v[v < 0] = 0

            structure = int(params["structure"] * fps)
            v_thresh = params["v_thresh"]
            epoch_frames = self.get_epochs(v, v_thresh, structure)
            if len(epoch_frames) > 0:
                epoch_times = t[epoch_frames]
                epoch_durations = epoch_times[:, 1] - epoch_times[:, 0]
                epoch_frames = epoch_frames[epoch_durations >= params["min_duration"]]

            min_distance = int(params["min_distance"] * fps)
            prominence = params["prominence"]
            sub_epochs = self.get_sub_epochs(v, epoch_frames, min_distance, prominence)

            for i, (start_frame, end_frame) in enumerate(epoch_frames):
                entries_epochs.append(dict(
                    **hand_key, epoch_id=i, start_frame=start_frame, end_frame=end_frame,
                    start_time=t[start_frame], end_time=t[end_frame],
                ))

                for j, global_id, sub_epoch_start, sub_epoch_end in sub_epochs[i]:
                    entries_sub_epochs.append(dict(
                        **hand_key, epoch_id=i, sub_epoch_id=j, global_id=global_id,
                        start_frame=sub_epoch_start, end_frame=sub_epoch_end,
                        start_time=t[sub_epoch_start], end_time=t[sub_epoch_end],
                    ))

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
        movement_mask = v > v_thresh
        movement_mask = binary_closing(movement_mask, structure=np.ones([structure]))
        mask_labeled, n_components = label(movement_mask)
        epoch_frames = []
        for i in range(1, n_components + 1):
            start, end = np.nonzero(mask_labeled == i)[0][[0, -1]]
            epoch_frames.append([start, end + 1])
        return np.array(epoch_frames)

    @staticmethod
    def get_sub_epochs(v, epoch_frames, min_distance, prominence):
        sub_epochs = {}
        global_id = 0
        for i, (start, end) in enumerate(epoch_frames):
            epoch_mask = MovementSegmentationSleap.frames_to_mask(len(v), start, end)
            v_epoch = v[epoch_mask]
            local_minima, _ = find_peaks(-v_epoch, prominence=prominence, distance=min_distance)
            local_minima = [0, *local_minima, len(v_epoch)-1]
            sub_epochs[i] = []
            for j in range(len(local_minima) - 1):
                sub_epoch_start = local_minima[j] + start
                sub_epoch_end = local_minima[j+1] + start
                sub_epochs[i].append([j, global_id, sub_epoch_start, sub_epoch_end])
                global_id += 1
            sub_epochs[i] = np.array(sub_epochs[i])
        return sub_epochs


@schema
class JoystickOcclusionSleap(dj.Computed):
    definition = """ # compute number of hand labels in different zones
    -> FilteredSleap
    -> JoystickPosition
    ---
    """

    class Hand(dj.Part):
        definition = """ # hand occlusion data
        -> JoystickOcclusionSleap
        hand                : enum('L', 'R')
        ---
        side                : enum('ipsi', 'contra')
        bound_l             : int
        bound_r             : int
        rest_occupancy      : longblob
        search_occupancy    : longblob
        joystick_occupancy  : longblob
        """

    def make(self, key):
        entries_hands = []
        hand_keys = (FilteredSleap.Hand * JoystickPosition.Hand.proj() & key).fetch("KEY")
        for hand_key in hand_keys:
            side = (FilteredSleap.Hand & hand_key).fetch1("side")
            x, y = (FilteredSleap.Label & hand_key & "label!='elbow'").fetch("x", "y")
            x = np.stack(x).T
            y = np.stack(y).T
            jx = (JoystickPosition.Hand & hand_key).fetch1("x")
            bound_l = int(jx - 100)
            bound_r = int(jx + 60)
            rest_occupancy = (x < bound_l).sum(axis=1)
            search_occupancy = (x > bound_r).sum(axis=1)
            joystick_occupancy = ((x >= bound_l) & (x <= bound_r)).sum(axis=1)
            entries_hands.append(dict(
                **hand_key, side=side, bound_l=bound_l, bound_r=bound_r,
                rest_occupancy=rest_occupancy, search_occupancy=search_occupancy, joystick_occupancy=joystick_occupancy,
            ))

        self.insert1(key)
        self.Hand.insert(entries_hands)


@schema
class ReachEpochSleap(dj.Computed):
    definition = """ # compute reach epochs
    -> MovementSegmentationSleap.Hand
    -> JoystickOcclusionSleap.Hand
    ---
    """
    _key_source = (MovementSegmentationSleap.Hand * JoystickOcclusionSleap.Hand.proj()) & "side='ipsi'"

    class Epoch(dj.Part):
        definition = """
        -> ReachEpochSleap
        epoch_id            : int
        ---
        """

    class SubEpoch(dj.Part):
        definition = """
        -> ReachEpochSleap.Epoch
        sub_epoch_id         : int
        ---
        is_reach             : bool
        """

    def make(self, key):
        rest_occupancy, search_occupancy, joystick_occupancy = (JoystickOcclusionSleap.Hand & key).fetch1("rest_occupancy", "search_occupancy", "joystick_occupancy")
        se_starts, se_ends, se_ids, e_ids = (MovementSegmentationSleap.SubEpoch & key).fetch("start_frame", "end_frame", "sub_epoch_id", "epoch_id", order_by=('epoch_id', 'sub_epoch_id'))
        crossover_frames = np.nonzero(np.diff(np.sign(rest_occupancy - (joystick_occupancy + search_occupancy)), prepend=0) <= -1)[0]
        epoch_classes = {}
        for cr in crossover_frames:
            for s, e, seid, eid in zip(se_starts, se_ends, se_ids, e_ids):
                if (s <= cr <= e) and (rest_occupancy[s] > joystick_occupancy[s] + search_occupancy[s]):
                    epoch_classes[(eid, seid)] = True
                    break
        entries_epochs = [dict(**key, epoch_id=e_id) for e_id in np.unique(e_ids)]
        entries_sub_epochs = []
        for seid, eid in zip(se_ids, e_ids):
            entries_sub_epochs.append(dict(**key, epoch_id=eid, sub_epoch_id=seid, is_reach=(eid, seid) in epoch_classes.keys()))
        self.insert1(key)
        self.Epoch.insert(entries_epochs)
        self.SubEpoch.insert(entries_sub_epochs)


@schema
class RetractEpochSleap(dj.Computed):
    definition = """ # compute retract epochs
    -> MovementSegmentationSleap.Hand
    -> JoystickOcclusionSleap.Hand
    ---
    """
    _key_source = (MovementSegmentationSleap.Hand * JoystickOcclusionSleap.Hand.proj()) & "side='ipsi'"

    class Epoch(dj.Part):
        definition = """
        -> RetractEpochSleap
        epoch_id            : int
        ---
        """

    class SubEpoch(dj.Part):
        definition = """
        -> RetractEpochSleap.Epoch
        sub_epoch_id         : int
        ---
        is_retract           : bool
        """

    def make(self, key):
        rest_occupancy, search_occupancy, joystick_occupancy = (JoystickOcclusionSleap.Hand & key).fetch1("rest_occupancy", "search_occupancy", "joystick_occupancy")
        se_starts, se_ends, se_ids, e_ids = (MovementSegmentationSleap.SubEpoch & key).fetch("start_frame", "end_frame", "sub_epoch_id", "epoch_id", order_by=('epoch_id', 'sub_epoch_id'))
        crossover_frames = np.nonzero(np.diff(np.sign(rest_occupancy - (joystick_occupancy + search_occupancy)), prepend=0) >= 1)[0]
        epoch_classes = {}
        for cr in crossover_frames:
            for s, e, seid, eid in zip(se_starts, se_ends, se_ids, e_ids):
                if (s <= cr <= e) and (rest_occupancy[e] > joystick_occupancy[e] + search_occupancy[e]):
                    epoch_classes[(eid, seid)] = True
                    break
        entries_epochs = [dict(**key, epoch_id=e_id) for e_id in np.unique(e_ids)]
        entries_sub_epochs = []
        for seid, eid in zip(se_ids, e_ids):
            entries_sub_epochs.append(dict(**key, epoch_id=eid, sub_epoch_id=seid, is_retract=(eid, seid) in epoch_classes.keys()))
        self.insert1(key)
        self.Epoch.insert(entries_epochs)
        self.SubEpoch.insert(entries_sub_epochs)


@schema
class SearchEpochSleap(dj.Computed):
    definition = """ # compute search epochs
    -> MovementSegmentationSleap.Hand
    -> JoystickOcclusionSleap.Hand
    ---
    """
    _key_source = (MovementSegmentationSleap.Hand * JoystickOcclusionSleap.Hand.proj()) & "side='ipsi'"

    class Epoch(dj.Part):
        definition = """
        -> SearchEpochSleap
        epoch_id            : int
        ---
        """

    class SubEpoch(dj.Part):
        definition = """
        -> SearchEpochSleap.Epoch
        sub_epoch_id         : int
        ---
        is_search            : bool
        """

    def make(self, key):
        rest_occupancy, search_occupancy, joystick_occupancy = (JoystickOcclusionSleap.Hand & key).fetch1("rest_occupancy", "search_occupancy", "joystick_occupancy")
        se_starts, se_ends, se_ids, e_ids = (MovementSegmentationSleap.SubEpoch & key).fetch("start_frame", "end_frame", "sub_epoch_id", "epoch_id", order_by=('epoch_id', 'sub_epoch_id'))
        epoch_classes = {}
        for s, e, seid, eid in zip(se_starts, se_ends, se_ids, e_ids):
            if (rest_occupancy[s] <= joystick_occupancy[s] + search_occupancy[s]) and (rest_occupancy[e] <= joystick_occupancy[e] + search_occupancy[e]):
                epoch_classes[(eid, seid)] = True
        entries_epochs = [dict(**key, epoch_id=e_id) for e_id in np.unique(e_ids)]
        entries_sub_epochs = []
        for seid, eid in zip(se_ids, e_ids):
            entries_sub_epochs.append(dict(**key, epoch_id=eid, sub_epoch_id=seid, is_search=(eid, seid) in epoch_classes.keys()))
        self.insert1(key)
        self.Epoch.insert(entries_epochs)
        self.SubEpoch.insert(entries_sub_epochs)


@schema
class SubEpochClassificationSleap(dj.Computed):
    definition = """ # classify sub-epochs
    -> ReachEpochSleap
    -> RetractEpochSleap
    -> SearchEpochSleap
    -> exp.JoystickPresence
    -> Synchronisation.Hand
    ---
    t_to_exclude            : float
    """
    epoch_labels = ['reach', 'search', 'retract', 'excluded', 'other']

    class Epoch(dj.Part):
        definition = """
        -> SubEpochClassificationSleap
        epoch_id            : int
        ---
        """

    class SubEpoch(dj.Part):
        definition = """
        -> SubEpochClassificationSleap.Epoch
        sub_epoch_id         : int
        ---
        sub_epoch_class      : enum('reach', 'search', 'retract', 'excluded', 'other')
        joystick_frames      : int
        joystick_percentage  : float
        """

    def make(self, key):
        t_to_exclude = 1
        t_joystick_in, t_joystick_out = (exp.JoystickPresence.Trial & key).fetch("t_joystick_in", "t_joystick_out", order_by="trial_id")
        t = (Synchronisation.Hand & key).fetch1("frame_timestamps")
        joystick_present_mask = np.zeros_like(t, dtype=bool)
        for t_in, t_out in zip(t_joystick_in, t_joystick_out):
            joystick_present_mask[(t >= t_in) & (t <= t_out)] = True

        f_starts, f_ends, t_starts, t_ends, se_ids, e_ids, is_reach, is_search, is_retract = (
            MovementSegmentationSleap.SubEpoch
            * ReachEpochSleap.SubEpoch
            * RetractEpochSleap.SubEpoch
            * SearchEpochSleap.SubEpoch
            & key
        ).fetch("start_frame", "end_frame", "start_time", "end_time", "sub_epoch_id", "epoch_id", "is_reach", "is_search", "is_retract", order_by=('epoch_id', 'sub_epoch_id'))

        entries_epochs = [dict(**key, epoch_id=e_id) for e_id in np.unique(e_ids)]
        entries_sub_epochs = []
        for i, (eid, seid) in enumerate(zip(e_ids, se_ids)):
            t_start, t_end = t_starts[i], t_ends[i]
            f_start, f_end = f_starts[i], f_ends[i]
            reach, search, retract = is_reach[i], is_search[i], is_retract[i]
            joystick_frames = (joystick_present_mask[f_start:f_end]).sum()
            joystick_percentage = joystick_frames / (f_end - f_start) * 100
            joystick_out_included = np.any([(t_start - 0.075 <= t_out <= t_end) for t_out in t_joystick_out])
            in_excluded_time = np.any([(t_out <= t_start <= t_out + t_to_exclude) for t_out in t_joystick_out])
            if joystick_out_included or in_excluded_time:
                sub_epoch_class = 'excluded'
            elif reach:
                sub_epoch_class = 'reach'
            elif search:
                sub_epoch_class = 'search'
            elif retract:
                sub_epoch_class = 'retract'
            else:
                sub_epoch_class = 'other'
            entries_sub_epochs.append(dict(
                **key, epoch_id=eid, sub_epoch_id=seid, sub_epoch_class=sub_epoch_class,
                joystick_frames=joystick_frames, joystick_percentage=joystick_percentage,
            ))

        self.insert1(dict(**key, t_to_exclude=t_to_exclude))
        self.Epoch.insert(entries_epochs)
        self.SubEpoch.insert(entries_sub_epochs)


@schema
class EpochClassificationSleap(dj.Computed):
    definition = """ # classify epochs based on task-related outcomes
    -> SubEpochClassificationSleap
    """

    class Epoch(dj.Part):
        definition = """
        -> EpochClassificationSleap
        epoch_id            : int
        ---
        epoch_class         : enum('rewarded', 'miss', 'other', 'excluded')
        n_sub_epochs        : int
        n_reach             : int
        frames_reach        : int
        n_search            : int
        frames_search       : int
        n_retract           : int
        frames_retract      : int
        n_excluded          : int
        frames_excluded     : int
        n_other             : int
        frames_other        : int
        n_joystick_frames   : int
        """

    def make(self, key):
        t_rew = (exp.JoystickExperiment.Trials & dict(**key, successful=1)).fetch("t_servo_out")
        epoch_keys, epoch_ids, start_times = (SubEpochClassificationSleap.Epoch * MovementSegmentationSleap.Epoch & key).fetch("KEY", "epoch_id", "start_time", order_by="epoch_id")

        rewarded_ids = []
        for t_r in t_rew:
            diff = t_r - start_times
            diff[diff <= 0.01] = np.nan
            if not np.isnan(diff).all():
                tr_idx = np.nanargmin(diff)
                rewarded_ids.append(epoch_ids[tr_idx])

        entries_epochs = []
        for epoch_key, epoch_id in zip(epoch_keys, epoch_ids):
            sub_epoch_classes, joystick_frames, sub_epoch_frames = (SubEpochClassificationSleap.SubEpoch * MovementSegmentationSleap.SubEpoch.proj(n_frames='end_frame-start_frame') & epoch_key).fetch("sub_epoch_class", "joystick_frames", "n_frames")
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

            if epoch_id in rewarded_ids:
                epoch_class = 'rewarded'
            elif n_excluded > n_sub_epochs/3:
                epoch_class = 'excluded'
            elif n_reach + n_search > 0:
                epoch_class = 'miss'
            else:
                epoch_class = 'other'

            entries_epochs.append(dict(
                **epoch_key, epoch_class=epoch_class,
                n_sub_epochs=n_sub_epochs, n_reach=n_reach, frames_reach=frames_reach,
                n_search=n_search, frames_search=frames_search,
                n_retract=n_retract, frames_retract=frames_retract,
                n_excluded=n_excluded, frames_excluded=frames_excluded,
                n_other=n_other, frames_other=frames_other,
                n_joystick_frames=np.sum(joystick_frames),
            ))

        self.insert1(key)
        self.Epoch.insert(entries_epochs)


@schema
class RestingMaskSleap(dj.Computed):
    definition = """ # compute resting mask
    -> MovementSegmentationSleap
    -> JoystickOcclusionSleap
    ---
    """

    class Hand(dj.Part):
        definition = """ # hand resting mask
        -> RestingMaskSleap
        hand                : enum('L', 'R')
        ---
        side                : enum('ipsi', 'contra')
        moving_mask         : longblob
        resting_mask        : longblob
        paw_still_mask      : longblob
        percent_still       : float
        percent_resting     : float
        """

    def make(self, key):
        hand_keys = (MovementSegmentationSleap.Hand * JoystickOcclusionSleap.Hand.proj() & key).fetch("KEY")
        entries_hands = []
        for hand_key in hand_keys:
            side = (PawRecording.Hand & hand_key).fetch1("side")
            n_frames = (PawRecording.Hand & hand_key).fetch1("n_frames")
            rest_occupancy, search_occupancy, joystick_occupancy = (JoystickOcclusionSleap.Hand & hand_key).fetch1("rest_occupancy", "search_occupancy", "joystick_occupancy")
            f_starts, f_ends = (MovementSegmentationSleap.Epoch & hand_key).fetch("start_frame", "end_frame", order_by='epoch_id')
            moving_mask = np.zeros(n_frames, dtype=bool)
            for s, e in zip(f_starts, f_ends):
                moving_mask[s:e] = True
            paw_still_mask = ~moving_mask
            is_in_rest_zone = (rest_occupancy > joystick_occupancy + search_occupancy)
            resting_mask = is_in_rest_zone & paw_still_mask
            percent_still = paw_still_mask.sum() / n_frames * 100
            percent_resting = resting_mask.sum() / n_frames * 100
            entries_hands.append(dict(
                **hand_key, side=side,
                moving_mask=moving_mask, resting_mask=resting_mask, paw_still_mask=paw_still_mask,
                percent_still=percent_still, percent_resting=percent_resting,
            ))

        self.insert1(key)
        self.Hand.insert(entries_hands)


@schema
class HoldEpochSleap(dj.Computed):
    definition = """ # compute hold epochs
    -> JoystickOcclusionSleap.Hand
    -> RestingMaskSleap.Hand
    -> exp.JoystickPresence
    -> Synchronisation.Hand
    -> MovementSegmentationParamsSleap
    ---
    hold_mask            : longblob
    deflection_threshold : float
    """
    _key_source = (
        JoystickOcclusionSleap.Hand
        * RestingMaskSleap.Hand.proj()
        * exp.JoystickPresence
        * Synchronisation.Hand.proj()
        * MovementSegmentationParamsSleap) & "side='ipsi'"

    class Epoch(dj.Part):
        definition = """ # holding epochs, distinct from MovementSegmentation epochs)
        -> HoldEpochSleap
        hold_epoch_id       : int
        ---
        start_frame         : int
        end_frame           : int
        start_time          : float
        end_time            : float
        """

    def make(self, key):
        deflection_threshold = -1.63
        t_joystick_in, t_joystick_out = (exp.JoystickPresence.Trial & key).fetch("t_joystick_in", "t_joystick_out", order_by="trial_id")
        t = (Synchronisation.Hand & key).fetch1("frame_timestamps")
        joystick_present_mask = np.zeros_like(t, dtype=bool)
        for t_in, t_out in zip(t_joystick_in, t_joystick_out):
            joystick_present_mask[(t >= t_in) & (t <= t_out)] = True

        still_mask = (RestingMaskSleap.Hand & key).fetch1("paw_still_mask")
        rest_occupancy, search_occupancy, joystick_occupancy = (JoystickOcclusionSleap.Hand & key).fetch1("rest_occupancy", "search_occupancy", "joystick_occupancy")
        joystick_location_mask = (joystick_occupancy >= rest_occupancy) & (joystick_occupancy >= search_occupancy)

        t_joystick, y_joystick = (exp.JoystickReadouts.Data & key).fetch1("t", "y")
        y_joystick_resample = np.interp(t, t_joystick, y_joystick)
        deflection_mask = y_joystick_resample < deflection_threshold

        hold_mask = (joystick_location_mask | deflection_mask) & still_mask & joystick_present_mask
        structure = (MovementSegmentationParamsSleap & key).fetch1("structure")
        fps = (PawRecording.Hand & key).fetch1("fps")
        structure = int(structure * fps)
        hold_mask = binary_closing(hold_mask, structure=np.ones([structure]))

        epochs_entries = []
        hold_mask_labeled, n_components = label(hold_mask)
        for i in range(1, n_components + 1):
            f_start, f_end = np.nonzero(hold_mask_labeled == i)[0][[0, -1]]
            t_start, t_end = t[f_start], t[f_end]
            epochs_entries.append(dict(**key, hold_epoch_id=i, start_frame=f_start, end_frame=f_end + 1, start_time=t_start, end_time=t_end))

        self.insert1(dict(**key, deflection_threshold=deflection_threshold, hold_mask=hold_mask))
        self.Epoch.insert(epochs_entries)
