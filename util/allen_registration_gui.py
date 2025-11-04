import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Qt5Agg")
import numpy as np
import cv2
from pathlib import Path
from util.plotting import overlay_allen
import tifffile as tif
from schema.mpanze_widefield_refactor import mpanze_widefield_refactor as wf

# params
mouse_id = int(input("insert mouse id: "))
# get reference session
key_ref = (wf.ReferenceSession() & dict(mouse_id=mouse_id)).fetch1()
img_ref = (wf.WidefieldSession & dict(mouse_id=mouse_id, day=key_ref['day_ref'])).get_frame()
h, w = img_ref.shape
interaction_distance = 10

# create figure 1
fig1, ax1 = plt.subplots(1,1, figsize = (8,8))
ax1.set_title('Reference image')
ax1.imshow(img_ref, "Greys_r")
ax1.set_axis_off()
plt.tight_layout()

# create figure 2
fig2, ax2 = plt.subplots(1,1, figsize = (8,8))
ax2.set_title('Allen outlines')
ax2.set_axis_off()
ax2 = overlay_allen(ax2, res=(h,w))
ax2.set_xlim([0,w])
ax2.set_ylim([0,h])
ax2.set_aspect("equal")
ax2.invert_yaxis()
ax2.grid()
plt.tight_layout()

# create figure 3
fig3, ax3 = plt.subplots(1,1, figsize = (8,8))
ax3.set_title('Registration')
imsh_ref = ax3.imshow(img_ref, "Greys_r")
ax3 = overlay_allen(ax3, line_color="r-", res=(h,w))
plt.tight_layout()

# temp variables
current_selection_1 = -1
pts_1 = []
txt_1 = []
current_selection_2 = -1
pts_2 = []
txt_2 = []
M = []
img_ref_affine = []

# function definitions
distance = lambda p1, p2: np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
def get_data(line2D):
    data = line2D.get_data()
    return data[0], data[1]
def select_point(x, y, points):
    for i, point in enumerate(points):
        p2 = get_data(point)
        if distance((x,y), p2) <= interaction_distance:
            return i
    return -1
def update_point(x, y, selection, figure):
    if figure==1:
        pts_1[selection].set_data([x], [y])
        txt_1[selection].set_position((x+5, y+5))
    else:
        pts_2[selection].set_data([x],[y])
        txt_2[selection].set_position((x+5, y+5))
def add_point(x, y, figure):
    if figure==1:
        p, = ax1.plot(x,y,"c.")
        #p.set_data([x,y])
        N = len(pts_1) + 1
        t = ax1.text(x+5, y+5, f"{N}", color="c", fontsize=10)
        pts_1.append(p)
        txt_1.append(t)
    else:
        p, = ax2.plot(x,y,"c.")
        #p.set_data(x,y)
        N = len(pts_2) + 1
        t = ax2.text(x+5, y+5, f"{N}", color="c", fontsize=10)
        pts_2.append(p)
        txt_2.append(t)
def get_points():
    src = []
    dst = []
    for pt in pts_1:
        src.append(get_data(pt))
    for pt in pts_2:
        dst.append(get_data(pt))
    return np.array(src), np.array(dst)
def update_fig_3(src, dst):
    global M
    #M, inliers = cv2.estimateAffinePartial2D(src,dst,method=cv2.LMEDS)
    M, inliers = cv2.estimateAffinePartial2D(src,dst,ransacReprojThreshold=100)
    img_ref_affine = cv2.warpAffine(img_ref, M, (h,w))
    imsh_ref.set_data(img_ref_affine)
    fig3.canvas.draw()
    fig3.canvas.flush_events()
# event handlers
def onclick(event, figure):
    ix, iy = event.xdata, event.ydata
    if figure == 1:
        global current_selection_1
        current_selection_1 = select_point(ix,iy, pts_1)
    else:
        global current_selection_2
        current_selection_2 = select_point(ix,iy, pts_2)
def onrelease(event, figure):
    ix, iy = event.xdata, event.ydata
    if figure==1:
        global current_selection_1
        if current_selection_1 == -1:
            add_point(ix, iy, figure)
        else:
            update_point(ix, iy, current_selection_1, figure)
    else:
        global current_selection_2
        if current_selection_2 == -1:
            add_point(ix, iy, figure)
        else:
            update_point(ix, iy, current_selection_2, figure)
    fig1.canvas.draw()
    fig1.canvas.flush_events()
    fig2.canvas.draw()
    fig2.canvas.flush_events()
    # compute transform
    src, dst = get_points()
    if (len(src) == len(dst)) and (len(dst) >= 2):
        update_fig_3(src, dst)
        
fig1.canvas.mpl_connect('button_press_event', lambda event: onclick(event,1))
fig2.canvas.mpl_connect('button_press_event', lambda event: onclick(event,2))
fig1.canvas.mpl_connect('button_release_event', lambda event: onrelease(event, 1))
fig2.canvas.mpl_connect('button_release_event', lambda event: onrelease(event, 2))
plt.show()

response = input("save data? [y/n]")
if response == "y":
    from schema import mpanze_widefield
    img_entry = cv2.warpAffine(img_ref, M, (512,512))
    src, dst = get_points()
    entry = dict(**key_ref, allen_matrix=M, pts_ref=src, pts_allen=dst, img_aligned=img_entry)
    wf.AllenRegistration().insert1(entry)
else:
    raise ValueError("Registration not saved")