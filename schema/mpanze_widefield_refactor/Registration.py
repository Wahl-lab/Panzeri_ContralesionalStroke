"""
tables related to image registration and masking
"""

# connect to the database
import login
login.connect()
import datajoint as dj

# import table dependencies
from schema.mpanze_widefield_refactor.WidefieldSession import WidefieldSession
from schema import common_mice

# other dependencies
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tifffile as tif
from matplotlib.widgets import LassoSelector

# instantiate the schema
schema = dj.schema('mpanze_widefield_refactor', locals(), create_tables=True)

@schema
class ReferenceSession(dj.Computed):
    definition = """ # reference session for each mouse
    -> WidefieldSession.proj(day_ref="day", num_ref="session_num")
    ---
    """
    # include only the 1st session for each mouse
    _key_source = WidefieldSession.proj(day_ref="day", num_ref="session_num") * common_mice.Mouse.aggr(WidefieldSession, day_ref="min(day)")
    def make(self, key):
        # failsafe, only one entry allowed per mouse
        assert len(self & dict(mouse_id=key["mouse_id"])) == 0, "only one entry allowed per mouse"
        self.insert1(key)


@schema
class RegisteredSession(dj.Computed):
    definition = """ # Affine registration of a session to a reference session
    -> WidefieldSession
    ---
    -> ReferenceSession
    affine_matrix       : longblob      # affine transformation matrix to reference image
    pts_ref             : longblob      # reference points
    pts_current         : longblob      # points in current session
    """
    def make(self, key):
        # get frame from reference session
        ref_key = (ReferenceSession & key).proj(day="day_ref", session_num="num_ref").fetch1("KEY")
        ref_frame = (WidefieldSession & ref_key).get_frame()

        # get frame from current session
        current_frame = (WidefieldSession & key).get_frame()

        # prepare figure for plotting
        f, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].set_axis_off()
        ax[1].set_axis_off()
        ax[0].imshow(ref_frame, cmap="gray")
        ax[1].imshow(current_frame, cmap="gray")
        ax[0].set_title("Reference frame")
        ax[1].set_title("Current frame")
        f.canvas.manager.full_screen_toggle()
        f.tight_layout()

        # select 7 points in each frame
        pts_ref = []
        pts_current = []
        for i in range(7):
            pts = plt.ginput(2, timeout=-1, show_clicks=True)
            pts = np.array(pts)
            pts_ref.append(pts[0])
            pts_current.append(pts[1])
            ax[0].plot(pts[0,0], pts[0,1], 'ro')
            ax[1].plot(pts[1,0], pts[1,1], 'ro')
            ax[0].text(pts[0,0], pts[0,1], str(i), color='r')
            ax[1].text(pts[1,0], pts[1,1], str(i), color='r')
            f.canvas.draw()
        plt.close(f)
        pts_ref = np.array(pts_ref, dtype=np.float32)
        pts_current = np.array(pts_current, dtype=np.float32)

        # compute affine transformation
        M, _ = cv2.estimateAffinePartial2D(pts_current, pts_ref)
        h, w = ref_frame.shape
        warped_frame = cv2.warpAffine(current_frame, M, (w, h))

        # plot results
        f, ax = plt.subplots(1, 3, figsize=(12, 5), sharex=True, sharey=True)
        for a in ax:
            a.set_axis_off()
            a.grid(True, color='w')
        ax[0].set_axis_off()
        ax[1].set_axis_off()
        ax[2].set_axis_off()
        ax[0].imshow(ref_frame, cmap="gray")
        ax[1].imshow(warped_frame, cmap="gray")
        ax[2].imshow(ref_frame, cmap="Reds_r", alpha=0.5)
        ax[2].imshow(warped_frame, cmap="Blues_r", alpha=0.5)
        ax[0].set_title("Reference frame")
        ax[1].set_title("Warped frame")
        ax[2].set_title("Overlay")
        f.canvas.manager.full_screen_toggle()
        f.tight_layout()

        # get user input
        plt.show(block=False)
        user_input = input("Accept registration? [y/n]: ")
        plt.close(f)
        if user_input != "y":
            raise Exception("Registration rejected")
        
        # insert entry
        self.insert1(dict(**key, day_ref= ref_key["day"], num_ref=ref_key["session_num"], affine_matrix=M, pts_ref=pts_ref, pts_current=pts_current))


@schema
class MaskedSession(dj.Computed):
    definition = """ # combined mask of valid pixels and autofluorescence mask
    -> WidefieldSession
    ---
    mask_session        : longblob      # combined mask for session (255 = valid, 0 = invalid)
    mask_fov            : longblob      # mask of valid pixels in FOV (255 = valid, 0 = invalid)
    mask_saturation     : longblob      # mask of saturated pixels (255 = saturated, 0 = not saturated)
    """
    def make(self, key):
        #sigma = 35
        # load saturation mask
        mask_saturation = self.load_saturation_mask(key)
        # smooth saturation mask
        #mask_saturation = cv2.GaussianBlur(mask_saturation, (sigma,sigma), 5*sigma, 5*sigma)
        #mask_saturation = (mask_saturation > 127).astype(np.uint8) * 255
        # create rgba mask to overlay on top of frame
        mask_saturation_rgb = np.zeros((mask_saturation.shape[0], mask_saturation.shape[1], 4), dtype=np.uint8)
        mask_saturation_rgb[:,:,0] = mask_saturation
        mask_saturation_rgb[:,:,1] = 255 - mask_saturation
        mask_saturation_rgb[:,:,2] = 255 - mask_saturation
        mask_saturation_rgb[:,:,3] = mask_saturation // 2

        # get first frame
        frame = (WidefieldSession & key).get_frame()
        # initialize masks
        mask_fov = np.zeros_like(frame, dtype=np.uint8)
        mask_session = ((mask_fov == 255) & (mask_saturation == 0)).astype(np.uint8) * 255

        # set up figure
        f, ax = plt.subplots(1,2)
        ax[0].set_axis_off()
        ax[1].set_axis_off()
        ax[0].imshow(frame, cmap="gray")
        ax[0].imshow(mask_saturation_rgb)
        ax[1].imshow(mask_session, cmap="gray")
        ax[0].set_title("Select mask area")
        ax[1].set_title("masked frame")
        f.canvas.manager.full_screen_toggle()
        f.tight_layout()

        # create callback function
        def update_mask(vertices):
            vertices = np.array(vertices, dtype=np.int32)
            # convert vertices to mask
            mask_part = np.zeros_like(mask_fov, dtype=np.uint8)
            cv2.fillPoly(mask_part, [vertices], 255)
            #mask_fov[:] = cv2.GaussianBlur(mask_fov, (sigma, sigma), 5*sigma, 5*sigma)
            mask_fov[:] = ((mask_fov == 255) | (mask_part == 255)).astype(np.uint8) * 255
            #mask_fov[:] = (mask_fov > 127).astype(np.uint8) * 255
            mask_session[:] = ((mask_fov == 255) & (mask_saturation == 0)).astype(np.uint8) * 255
            ax[1].imshow(mask_session, cmap="gray")
            f.canvas.draw()
            f.canvas.flush_events()

        # add callback for button press on right click - clear mask
        def clear_mask(event):
            if event.button == 3:
                mask_fov[:] = 0
                mask_session[:] = 0
                ax[1].imshow(mask_session, cmap="gray")
                f.canvas.draw()
                f.canvas.flush_events()
        
        # connect callback
        f.canvas.mpl_connect('button_press_event', clear_mask)
        # create lasso selector
        lasso = LassoSelector(ax[0], update_mask)
        plt.show()
        lasso.disconnect_events()
        del lasso
        plt.close('all')

        # insert data
        self.insert1(dict(**key, mask_session=mask_session, mask_fov=mask_fov, mask_saturation=mask_saturation))
    
    def load_saturation_mask(self, key):
        # load approx 1/2 minute of recording
        p_file = (WidefieldSession & key).get_path()
        n_frames = int(60 * 20)
        stack = tif.memmap(p_file, mode='r')
        # compute mask from max projection
        max_proj = np.max(stack[:n_frames], axis=0)
        del stack
        mask = (max_proj > 65500).astype(np.uint8) * 255 # assuming 16-bit tif
        return mask


@schema
class AllenRegistration(dj.Manual):
    definition = """ # Affine transformation from reference session to Allen atlas
    -> ReferenceSession
    ---
    allen_matrix        : longblob      # affine transformation matrix to atlas
    pts_ref             : longblob      # points in reference image
    pts_allen           : longblob      # points in Allen atlas
    img_aligned         : longblob      # aligned first frame in reference session
    """