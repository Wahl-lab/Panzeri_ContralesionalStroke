import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path

p_pickle = "/util/allenDorsalMapSplit.pkl"

def get_roi_edge(roi):
    """
    Get edge of roi
    params:
        roi: roi mask
    returns:
        edge: edge of roi
    """
    # get edge of roi using opencv findContours
    roi = roi.astype(np.uint8)
    contours, hierarchy = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    edge = np.squeeze(contours[0])
    return edge

def AP_split(mask, bound):
    mask_A, mask_P = np.copy(mask), np.copy(mask)
    mask_A[bound:]=0
    mask_P[:bound]=0
    return mask_A, mask_P
    
def ML_split(mask, bound):
    mask_M, mask_L = np.copy(mask), np.copy(mask)
    mask_M[:,:bound] = 0
    mask_L[:,bound:] = 0
    return mask_M, mask_L
    
def split_mask(mask, bregma, bound_ml, bound_ap, side = 'L'):
    am, pm, al, pl = np.copy(mask), np.copy(mask), np.copy(mask), np.copy(mask)
    h  = bregma[1]-bound_ap
    am[h:] = 0
    pm[:h] = 0
    al[h:] = 0
    pl[:h] = 0
    if side == 'L':
        # left hemisphere
        m = bregma[0] - bound_ml
        am[:, :m] = 0
        pm[:, :m] = 0
        al[:, m:] = 0
        pl[:, m:] = 0
    else:
        # right hemisphere
        m = bregma[0] + bound_ml
        am[:, m:] = 0
        pm[:, m:] = 0
        al[:, :m] = 0
        pl[:, :m] = 0
    return am.astype(np.uint8), pm.astype(np.uint8), al.astype(np.uint8), pl.astype(np.uint8)
        

def resize_allen(target_resolution):
    # load original allen rois
    with open(p_pickle, "rb") as f:
        d = pickle.load(f)
    
    interp_samples = 1000 # interpolate edges to this many samples (should be enough for any resolution)
    bregma = (294, 264)

    # N.B. this only applies to Hamatsu ORCA Flash 4.0 V3 and 1x magnification
    hamamatsu_sensor_size = (13.312, 13.312)  # (h,w) in mm
    magnification = 1.
    pix_per_mm_x = 586/hamamatsu_sensor_size[1] * magnification
    pix_per_mm_y = 586/hamamatsu_sensor_size[0] * magnification
    
    # anterior/lateral motor cortex boundaries, adapted from  Yang et al., 2023, Cell 186, 162-177
    AP_boundary_mm = 1.5
    LM_boundary_mm = 2
    AP_boundary_pixels = int(AP_boundary_mm * pix_per_mm_y)
    LM_boundary_pixels = int(LM_boundary_mm * pix_per_mm_x)
    
    # create combined motor cortex mask
    MO_R = d["masks"]["MOp_R"] + d["masks"]["MOs_R"]
    MO_L = d["masks"]["MOp_L"] + d["masks"]["MOs_L"]
    MO_R = cv2.morphologyEx(MO_R, cv2.MORPH_CLOSE, np.ones((5,5)))
    MO_L = cv2.morphologyEx(MO_L, cv2.MORPH_CLOSE, np.ones((5,5)))

    # split masks
    d["masks"]["AMM_R"], d["masks"]["PMM_R"], d["masks"]["ALM_R"], d["masks"]["PLM_R"] = split_mask(MO_R, bregma, LM_boundary_pixels, AP_boundary_pixels, side='R')
    d["masks"]["AMM_L"], d["masks"]["PMM_L"], d["masks"]["ALM_L"], d["masks"]["PLM_L"] = split_mask(MO_L, bregma, LM_boundary_pixels, AP_boundary_pixels, side='L')

    # get edges
    d["edges"]["AMM_R"] = get_roi_edge(d["masks"]["AMM_R"])
    d["edges"]["PMM_R"] = get_roi_edge(d["masks"]["PMM_R"])
    d["edges"]["ALM_R"] = get_roi_edge(d["masks"]["ALM_R"])
    d["edges"]["PLM_R"] = get_roi_edge(d["masks"]["PLM_R"])
    d["edges"]["AMM_L"] = get_roi_edge(d["masks"]["AMM_L"])
    d["edges"]["PMM_L"] = get_roi_edge(d["masks"]["PMM_L"])
    d["edges"]["ALM_L"] = get_roi_edge(d["masks"]["ALM_L"])
    d["edges"]["PLM_L"] = get_roi_edge(d["masks"]["PLM_L"])

    # add names to area names
    area_names = list((*d["area_names"][:-2], "AMM_R", "PMM_R", "ALM_R", "PLM_R", "AMM_L", "PMM_L", "ALM_L", "PLM_L"))

    # create additional composite masks
    PPC_R = d["masks"]["VISa_R"] + d["masks"]["VISrl_R"]
    PPC_L = d["masks"]["VISa_L"] + d["masks"]["VISrl_L"]
    SSp_nosemouth_R = d["masks"]["SSp-n_R"] + d["masks"]["SSp-m_R"]
    SSp_nosemouth_L = d["masks"]["SSp-n_L"] + d["masks"]["SSp-m_L"]
    RSP_R = d["masks"]["RSPagl_R"] + d["masks"]["RSPd_R"] + d["masks"]["RSPv_R"]
    RSP_L = d["masks"]["RSPagl_L"] + d["masks"]["RSPd_L"] + d["masks"]["RSPv_L"]
    VIS_medial_R = d["masks"]["VISam_R"] + d["masks"]["VISpm_R"]
    VIS_medial_L = d["masks"]["VISam_L"] + d["masks"]["VISpm_L"]
    # close holes in composite masks
    d["masks"]["PPC_R"] = cv2.morphologyEx(PPC_R, cv2.MORPH_CLOSE, np.ones((5,5)))
    d["masks"]["PPC_L"] = cv2.morphologyEx(PPC_L, cv2.MORPH_CLOSE, np.ones((5,5)))
    d["masks"]["SSp-nosemouth_R"] = cv2.morphologyEx(SSp_nosemouth_R, cv2.MORPH_CLOSE, np.ones((5,5)))
    d["masks"]["SSp-nosemouth_L"] = cv2.morphologyEx(SSp_nosemouth_L, cv2.MORPH_CLOSE, np.ones((5,5)))
    d["masks"]["RSP_R"] = cv2.morphologyEx(RSP_R, cv2.MORPH_CLOSE, np.ones((5,5)))
    d["masks"]["RSP_L"] = cv2.morphologyEx(RSP_L, cv2.MORPH_CLOSE, np.ones((5,5)))
    d["masks"]["VIS-medial_R"] = cv2.morphologyEx(VIS_medial_R, cv2.MORPH_CLOSE, np.ones((5,5)))
    d["masks"]["VIS-medial_L"] = cv2.morphologyEx(VIS_medial_L, cv2.MORPH_CLOSE, np.ones((5,5)))

    # get edges of composite masks
    d["edges"]["PPC_R"] = get_roi_edge(d["masks"]["PPC_R"])
    d["edges"]["PPC_L"] = get_roi_edge(d["masks"]["PPC_L"])
    d["edges"]["SSp-nosemouth_R"] = get_roi_edge(d["masks"]["SSp-nosemouth_R"])
    d["edges"]["SSp-nosemouth_L"] = get_roi_edge(d["masks"]["SSp-nosemouth_L"])
    d["edges"]["RSP_R"] = get_roi_edge(d["masks"]["RSP_R"])
    d["edges"]["RSP_L"] = get_roi_edge(d["masks"]["RSP_L"])
    d["edges"]["VIS-medial_R"] = get_roi_edge(d["masks"]["VIS-medial_R"])
    d["edges"]["VIS-medial_L"] = get_roi_edge(d["masks"]["VIS-medial_L"])

    # add composite masks to area names
    area_names.extend(["PPC_R", "PPC_L", "SSp-nosemouth_R", "SSp-nosemouth_L", "RSP_R", "RSP_L", "VIS-medial_R", "VIS-medial_L"])

    # create circle-split masks
    radius = 100
    center = 294, 280
    circ = np.zeros((679, 586), dtype=np.uint8)
    cv2.circle(circ, center, radius, 1, -1)
    d["masks"]["MOs-medial_R"] = ((d["masks"]["MOs_R"] > 0) & (circ > 0)).astype(np.uint8) * 255
    d["masks"]["MOs-lateral_R"] = ((d["masks"]["MOs_R"] > 0) & (circ == 0)).astype(np.uint8) * 255
    d["masks"]["MOs-medial_L"] = ((d["masks"]["MOs_L"] > 0) & (circ > 0)).astype(np.uint8) * 255
    d["masks"]["MOs-lateral_L"] = ((d["masks"]["MOs_L"] > 0) & (circ == 0)).astype(np.uint8) * 255
    d["masks"]["MOp-medial_R"] = ((d["masks"]["MOp_R"] > 0) & (circ > 0)).astype(np.uint8) * 255
    d["masks"]["MOp-lateral_R"] = ((d["masks"]["MOp_R"] > 0) & (circ == 0)).astype(np.uint8) * 255
    d["masks"]["MOp-medial_L"] = ((d["masks"]["MOp_L"] > 0) & (circ > 0)).astype(np.uint8) * 255
    d["masks"]["MOp-lateral_L"] = ((d["masks"]["MOp_L"] > 0) & (circ == 0)).astype(np.uint8) * 255
    d["masks"]["RSP-anterior_R"] = ((d["masks"]["RSP_R"] > 0) & (circ > 0)).astype(np.uint8) * 255
    d["masks"]["RSP-posterior_R"] = ((d["masks"]["RSP_R"] > 0) & (circ == 0)).astype(np.uint8) * 255
    d["masks"]["RSP-anterior_L"] = ((d["masks"]["RSP_L"] > 0) & (circ > 0)).astype(np.uint8) * 255
    d["masks"]["RSP-posterior_L"] = ((d["masks"]["RSP_L"] > 0) & (circ == 0)).astype(np.uint8) * 255
    # get edges of circle-split masks
    d["edges"]["MOs-medial_R"] = get_roi_edge(d["masks"]["MOs-medial_R"])
    d["edges"]["MOs-lateral_R"] = get_roi_edge(d["masks"]["MOs-lateral_R"])
    d["edges"]["MOs-medial_L"] = get_roi_edge(d["masks"]["MOs-medial_L"])
    d["edges"]["MOs-lateral_L"] = get_roi_edge(d["masks"]["MOs-lateral_L"])
    d["edges"]["MOp-medial_R"] = get_roi_edge(d["masks"]["MOp-medial_R"])
    d["edges"]["MOp-lateral_R"] = get_roi_edge(d["masks"]["MOp-lateral_R"])
    d["edges"]["MOp-medial_L"] = get_roi_edge(d["masks"]["MOp-medial_L"])
    d["edges"]["MOp-lateral_L"] = get_roi_edge(d["masks"]["MOp-lateral_L"])
    d["edges"]["RSP-anterior_R"] = get_roi_edge(d["masks"]["RSP-anterior_R"])
    d["edges"]["RSP-posterior_R"] = get_roi_edge(d["masks"]["RSP-posterior_R"])
    d["edges"]["RSP-anterior_L"] = get_roi_edge(d["masks"]["RSP-anterior_L"])
    d["edges"]["RSP-posterior_L"] = get_roi_edge(d["masks"]["RSP-posterior_L"])
    # add circle-split masks to area names
    area_names.extend(["MOs-medial_R", "MOs-lateral_R", "MOs-medial_L", "MOs-lateral_L",
                       "MOp-medial_R", "MOp-lateral_R", "MOp-medial_L", "MOp-lateral_L",
                       "RSP-anterior_R", "RSP-posterior_R", "RSP-anterior_L", "RSP-posterior_L"])

    # create outputs
    masks_resized = np.empty((len(area_names), *target_resolution), dtype=np.uint8)
    edges_resized = np.zeros((len(area_names), 2, interp_samples))
    mask_total = np.zeros(target_resolution, dtype=np.uint8)

    for i, area in enumerate(area_names):
        # resize mask
        mask_crop = d["masks"][area][:586]
        masks_resized[i] = cv2.resize(mask_crop, target_resolution, interpolation=cv2.INTER_AREA)
        if i < len(d["area_names"]):
            mask_total += masks_resized[i]
        # resize edge
        x_resized = d["edges"][area][:,0]*(target_resolution[1]/586)
        y_resized = d["edges"][area][:,1]*(target_resolution[0]/586)
        t = np.linspace(0, 1, len(x_resized))
        t_interp = np.linspace(0, 1, interp_samples)
        x_interp = np.interp(t_interp, t, x_resized)
        y_interp = np.interp(t_interp, t, y_resized)
        edge_resized = np.stack((x_interp, y_interp), axis=1).T
        edges_resized[i] = edge_resized

    area_names = np.array(area_names)
    bregma = np.array((294*target_resolution[1]/586, 264*target_resolution[0]/586))
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, np.ones((5,5)))
    mask_total[mask_total > 127] = 255
    mask_total[mask_total <= 127] = 0
    masks_resized[masks_resized > 127] = 255
    masks_resized[masks_resized <= 127] = 0

    p_out = f"util/allen_masks_{target_resolution[0]}x{target_resolution[1]}.npz"
    np.savez(p_out,masks=masks_resized, area_names=area_names, edges=edges_resized, mask_total=mask_total, bregma=bregma)

def load_allen(target_resloution):
    p_allen = Path(f"util/allen_masks_{target_resloution[0]}x{target_resloution[1]}.npz")
    if p_allen.exists():
        f = np.load(p_allen)
        return f["masks"], f["area_names"], f["edges"], f["mask_total"], f["bregma"]
    else:
        print("allen masks not found, generating...")
        resize_allen(target_resloution)
        return load_allen(target_resloution)
    
def overlay_allen(ax, areas_to_overlay=None, areas_to_dot=None, res=(512, 512), show_bregma=False, remove_axis=True, line_kw={'color':'k', 'lw':0.5}, bregma_kw={"ls":'None', "marker":'.'}):
    masks, area_names, edges, mask_total, bregma = load_allen(res)
    if areas_to_overlay is None:
        areas_to_overlay = area_names
    if areas_to_dot is None:
        areas_to_dot = []

    
    for i in range(len(area_names)):
        area = area_names[i]
        if (area in areas_to_overlay) and (area not in areas_to_dot):
            ax.plot(edges[i,0], edges[i,1], **line_kw)
        if (area in areas_to_overlay) and (area in areas_to_dot):
            kw_2 = {**line_kw}
            kw_2["ls"] = (0, (5, 10))  # loosely dashed
            ax.plot(edges[i,0], edges[i,1], **kw_2)

    if show_bregma:
        ax.plot(bregma[0], bregma[1], **bregma_kw)

    if remove_axis:
        ax.set_axis_off()

    
if __name__ == "__main__":
    # test allen masks
    for res in [(128,128), (256,256), (512,512)]:
        masks, area_names, edges, mask_total, bregma = load_allen(res)
        plt.figure()
        plt.title(f"resolution {res}")
        plt.imshow(masks[3], cmap="gray_r")
        for i,a in enumerate(area_names):
            plt.plot(edges[i,0], edges[i,1], 'r', linewidth=0.5)
        plt.plot(bregma[0], bregma[1], 'bo')
        plt.xlim(0, res[1])
        plt.ylim(res[0], 0)
        plt.axis("off")
    plt.show()