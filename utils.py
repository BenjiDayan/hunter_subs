import os
import numpy as np

# pysubs2 for subtitles?
# opencv for image processing
import cv2 as cv
import subprocess

import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from skimage.filters import threshold_otsu

import opencc
import easyocr

from tqdm import tqdm
from unittest.mock import MagicMock

from segment_anything import SamAutomaticMaskGenerator

# does OCR
reader = easyocr.Reader(['ch_tra'])


torrents_path = '/Users/benjidayan/Documents/torrents/hunter/'
fn = torrents_path + 'LR_Chinese_001_720P[52KHD].ts'

# command to slice out the right section of video:
# ffmpeg -i ./LR_Chinese_001_720P\[52KHD\].ts -filter:v "crop=530:93:380:555" -c:a copy out3.mp4


video_subtitle_slice_fn = 'out3.mp4'
cap = cv.VideoCapture(video_subtitle_slice_fn)

def get_frame_n(cap, frame_n):
    # our frames are np.ndarrays of shape (92, 550, 3)
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_n)
    ret, frame = cap.read()
    return ret, frame

def binary_process_frame(frame):
    white = frame.sum(axis=2) >= 255*3-54

    white = white.astype(np.uint8)

    # This helps to remove little blobs around, and fill in some holes
    blur = cv.GaussianBlur(white,(3,3),0)
    thresh = cv.threshold(blur, 100, 255, cv.THRESH_BINARY)[1]


    # Our text is too thin because it came from the inner white that is surrounded by black.
    # So pump it up a little bit. This is probably similar to the previous action
    kernel = np.ones((3,3)) / 9  # A simple averaging kernel
    convolved_image = convolve(blur.astype(np.float64), kernel)
    blur
    # plt.imshow(convolved_image)
    thresh = threshold_otsu(convolved_image)
    binary = convolved_image > thresh

    # Binary is 1 for white (text) and 0 for rest.
    # We flip to have background white and text is black, and normal 0 - 255 scale
    out = 255 - binary*255
    return out


def grow_thin_binary(frame, threshold_num_pixels=None):
    kernel = np.ones((3,3)) / 9  # A simple averaging kernel
    convolved_image = convolve(frame.astype(np.float64), kernel)
    if threshold_num_pixels is None:
        thresh = threshold_otsu(convolved_image)
    else:
        thresh = threshold_num_pixels/9 - 1e-6
    binary = convolved_image > thresh
    return binary.astype(np.uint8)


# This is worse than easyocr it seems so don't use it!
def ocr_frame_tesseract(binary_frame):
    """Frame should be good looking text"""
    text_frame = 255 - 255*binary_frame
    cv.imwrite('temp.png', text_frame)
    p = subprocess.Popen(['tesseract', '-l', 'chi_tra', 'temp.png', '-'], stdout=subprocess.PIPE)
    out, err = p.communicate()
    out = out.decode('utf-8')

    # traditional chinese to simplified
    cc = opencc.OpenCC('t2s')
    out_simplified = cc.convert(out)
    return out_simplified

def ocr_frame(binary_frame, trad2simple=True):
    """Frame should be good looking text"""
    text_frame = 255 - 255*binary_frame
    cv.imwrite('temp.png', text_frame)

    result = reader.readtext('temp.png')

    result = '\n'.join([res[1] for res in result])

    if not trad2simple:
        return result
    
    # traditional chinese to simplified
    cc = opencc.OpenCC('t2s')
    out_simplified = cc.convert(result)
    return out_simplified



def process_frame(frame):
    # Get exact part of frame that could have subtitles
    frame = frame[3:-1, :]  # is now 88 Y 550 X

    # # Divide into top and bottom
    # top = frame[:44, :]
    # bottom = frame[44:, :]

    text, binary_text_map = frame_to_text(frame)

    return text, binary_text_map


def blacken_colored_pixels(frame):
    """Finds pixels that aren't clearly some shade of grey, i.e. are colored.
    Sets them to black."""
    frame_r, frame_g, frame_b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    # otherwise we get wrap around issues when doing subtraction
    frame_r, frame_g, frame_b = frame_r.astype(np.float32), frame_g.astype(np.float32), frame_b.astype(np.float32)
    frame_max_pixel = np.max(frame, axis=2)
    np.abs(frame_r - frame_g).shape
    dists = np.stack([np.abs(frame_r - frame_g), np.abs(frame_r - frame_b), np.abs(frame_g - frame_b)], axis=2)
    max_abs_dist = np.max(dists, axis=2)
    out = max_abs_dist/frame_max_pixel

    frame2 = frame.copy()
    frame2[out > 0.07] = np.array([0, 0, 0])
    return frame2

def quick_process_frame(sam, frame, **kwargs):
    output_mask = frame_to_binary_text_pixels(frame)
    output_mask_sam = frame_to_binary_text_pixels_with_SAM(frame, sam, **kwargs)
    # output_mask_sam_grow_direct = grow_thin_binary(output_mask_sam)
    output_mask_sam_grow = frame_to_binary_text_pixels_with_SAM(frame, sam, grow=True)

    plt.figure(figsize=(15, 5))
    gs = plt.GridSpec(2, 2)
    gs.update(wspace=0.01, hspace=0.01)

    plt.subplot(gs[0]).imshow(frame)
    plt.subplot(gs[1]).imshow(output_mask)
    plt.subplot(gs[2]).imshow(output_mask_sam)
    plt.subplot(gs[3]).imshow(output_mask_sam_grow)

    



    for i in range(2):
        for j in range(2):
            ax = plt.subplot(gs[i, j])
            ax.set_axis_off()
            ax.set_aspect('equal')

    plt.show()


    ocr = ocr_frame(frame)
    print(f'#### Base:\n{ocr}')
    ocr_cv = ocr_frame(output_mask)
    print(f'#### CV:\n{ocr_cv}')
    ocr_sam = ocr_frame(output_mask_sam)
    print(f'#### SAM:\n{ocr_sam}')
    ocr_sam_grow = ocr_frame(output_mask_sam_grow)
    print(f'#### SAM:\n{ocr_sam_grow}')


    # label each axis with the ocr
    # plt.subplot(gs[0]).set_title(ocr)
    # plt.subplot(gs[1]).set_title(ocr_cv)
    # plt.subplot(gs[2]).set_title(ocr_sam)
    # plt.subplot(gs[3]).set_title(ocr_sam_grow)
    # plt.show()

    return ocr, ocr_cv, ocr_sam, ocr_sam_grow

def quick_process_frame_n2(sam, cap, n, rgb2bgr=True, **kwargs):
    """Like quick_process_frame_n, but also displays original frame as an extra plot,
    and has all 3 plots in one subplot 1x3 grid"""
    ret, frame = get_frame_n(cap, n)
    if rgb2bgr:
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    return quick_process_frame(sam, frame, **kwargs)


def frame_to_binary_text_pixels(img):
    """1's where it thinks text is, and 0's everywhere else"""
    # convert to grayscale
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # get high pass filter. This is a better version of edge detection?
    size = 8
    if not size%2:
        size +=1


    kernel = np.ones((size,size),np.float32)/(size*size)
    filtered= cv.filter2D(img_gray,-1,kernel)
    filtered = img_gray.astype('float32') - filtered.astype('float32')
    filtered = filtered + 127*np.ones(img_gray.shape, np.uint8)
   
    # do thresholding to get just the "bright" stuff
    HIGHPASS_THRESH = 180
    # BRIGHT_THRESH = 240
    BRIGHT_THRESH = 230

    white = img_gray > BRIGHT_THRESH
    white_hp = filtered > HIGHPASS_THRESH
    
    # Take logical and.
    out = white & white_hp

    # This helps to remove little blobs around, and fill in some holes
    # Do growing
    out = grow_thin_binary(out)
    return out

BOX_W = 38
BOX_H = 40
Y_MID1 = 21
Y_MID2 = 66

def check_box_size(bbox, upper_bound=True, lower_bound=True):
    w, h = bbox[2], bbox[3]
    out = True
    if upper_bound:
        out = out and w < 42 and h < 44
    if lower_bound:
        out = out and w > 34 and h > 36
    return out

def check_box_within_y_tramlines(bbox, y_mid):
    """Checks that the boxes y extremes are within y_mid +- 22"""
    x, y, w, h = bbox
    y1, y2 = y, y+h
    DIFF = BOX_H/2 + 4
    return np.abs(y1 - y_mid) <= DIFF and np.abs(y2 - y_mid) <= DIFF

def check_box_ypos(bbox, possible_ys, dist=10):
    """checks which of possible ys (line y height) that the box is on.
    -1 means none of them"""
    # for i, y in enumerate(possible_ys):
    #     if check_box_within_y_tramlines(bbox, y):
    #         return i
    # return -1

    mid = get_box_midpoint(bbox)
    mid_y = mid[1]
    for i, y in enumerate(possible_ys):
        if abs(mid_y - y) < dist:
            return i
    return -1

def get_contiguous_centered_masks(masks, x_center):
    """masks are a set of masks that seem to be on a line of same y.
    If its subtitles, it should be a central line with maybe some outliers on either side.
    We will extract the central contiguous line"""
    masks = masks.copy()
    # sorted by left x lim
    masks.sort(key=lambda x: x['bbox'][0])

    # left and right x lims of each mask
    xlims = [(m['bbox'][0], m['bbox'][0] + m['bbox'][2]) for m in masks]

    # iterate through and cut where there is a gap
    # e.g. make contiguous_idxs = [[0], [1,2], [3,4,5,6,7,8], [9]]
    contiguous_idxs = [[0]]
    contiguous_xlims = [[xlims[0][0], None]]  # will have l/r x lims of each line
    x_r_prev = xlims[0][1]
    for i, (x_l, x_r) in enumerate(xlims[1:]):
        i = i + 1
        if x_l - x_r_prev > BOX_W:  # non-contiguous: start a new line
            contiguous_xlims[-1][1] = x_r_prev
            contiguous_xlims.append([x_l, None])
            contiguous_idxs.append([i])
            x_r_prev = x_r
        else:
            x_r_prev = max(x_r_prev, x_r)
            contiguous_idxs[-1].append(i)
    contiguous_xlims[-1][1] = x_r_prev

    # find the line which overlaps with the centre of the image
    centre_line_masks = []
    for (x_l, x_r), idxs in zip(contiguous_xlims, contiguous_idxs):
        if x_l <= x_center <= x_r:
            centre_line_masks = idxs
            break  # there is only one centre line anyway

    return [masks[i] for i in centre_line_masks], [xlims[i] for i in centre_line_masks]

def remove_subset_masks(line_masks, line_xlims):
    """
    (mask, (x_l, x_r)) pairs, sorted by x_l

    we shall remove any masks that are subsets of others:
    """
    pass

def frame_to_binary_text_pixels_with_SAM(img, sam, bright_thresh=230, grow=False):
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(img)

    line1_masks = []
    line2_masks = []
    for mask in masks:
        bbox = mask['bbox']
        x, y, w, h = bbox
        # no idea why but sometimes the bbox is a float??
        x, y, w, h = int(x), int(y), int(w), int(h)
        if check_box_size(bbox, upper_bound=True, lower_bound=False):
            if check_box_within_y_tramlines(bbox, Y_MID1):
                line1_masks.append(mask)
            elif check_box_within_y_tramlines(bbox, Y_MID2):
                line2_masks.append(mask)

    # get central contiguous bits
    x_center = img.shape[1]//2
    if len(line1_masks) == 0:
        line1_masks, line1_xlims = [], []
    else:
        line1_masks, line1_xlims = get_contiguous_centered_masks(line1_masks, x_center)
    
    if len(line2_masks) == 0:
        line2_masks, line2_xlims = [], []
    else:
        line2_masks, line2_xlims = get_contiguous_centered_masks(line2_masks, x_center)
    
    potential_masks = line1_masks + line2_masks
    output_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for mask in potential_masks:
        # mask is a nice output dict
        seg_mask = mask['segmentation']
        bbox = mask['bbox']
        x, y, w, h = bbox
        # no idea why but sometimes the bbox is a float??
        x, y, w, h = int(x), int(y), int(w), int(h)
        bbox_extract_seg_mask = seg_mask[y:y+h, x:x+w]
        bbox_extract = img[y:y+h, x:x+w]
        bbox_extract_gray = cv.cvtColor(bbox_extract, cv.COLOR_RGB2GRAY)
        bbox_extract_img_bright = bbox_extract_gray > bright_thresh
        bbox_extract_char_mask = bbox_extract_img_bright & bbox_extract_seg_mask

        # output_mask[y:y+h, x:x+w] = bbox_extract_char_mask
        # actually we want to or the bits, not replace them
        # This is because we might have multiple overlapping bbox masks
        output_mask[y:y+h, x:x+w] = output_mask[y:y+h, x:x+w] | bbox_extract_char_mask

    # There's some weird bug if we do it in the loop above??
    if grow:
        output_mask = grow_thin_binary(output_mask)

    return output_mask
    


def frame_to_text(img, sam=None, trad2simple=True):
    """The frame may contain a horizontal line of text, or perhaps nothing.
    Actually that was for tesseract --psm 7. Now we are using easyocr, which
    is better at detecting text in images, and we just pass in everything."""

    # img = blacken_colored_pixels(img)

    if sam is None:
        binary_text_map = frame_to_binary_text_pixels(img)
    else:
        binary_text_map = frame_to_binary_text_pixels_with_SAM(img, sam, grow=True)

    # Do OCR
    ocr = ocr_frame(binary_text_map, trad2simple=trad2simple)
    return ocr, binary_text_map


class VideoSubExtractor:

    def __init__ (self, video_fn, sam=None):
        self.video_fn = video_fn
        self.cap = cv.VideoCapture(video_fn)
        self.frame_n = 0
        self.frame_max = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.sam=sam

        # it's roughly 30 fps. We should probably check for text every second or so?
        self.frame_skip = 20

        self.text_ts = []  # ordered list of (frame_n_start, frame_n_end, text) triples

    # def get_subs(self, max_n=None, use_tqdm=False, save_frames=False, out_folder=None):
    def get_subs(self, max_n=None, use_tqdm=False, out_file=None, trad2simple=True):
        # os.makedirs(out_folder, exist_ok=True)
    
        self.subs = []
        # make tqdm bar, setting to mock object if not tqdm
        pbar = tqdm(total=self.frame_max) if use_tqdm else MagicMock()
        pbar.n = self.frame_n
        while self.frame_n < (self.frame_max if max_n is None else max_n):
            ret, frame = get_frame_n(self.cap, self.frame_n)
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            # frame[3:-1, :]  # is now 88 Y 550 X
            if not ret:
                break
            text, binary_text_map = frame_to_text(frame, sam=self.sam, trad2simple=trad2simple)
            # if save_frames:
            #     if out_folder:
            #         cv.imwrite(os.path.join(out_folder, 'frame_%d.png' % self.frame_n), frame)
            #         cv.imwrite(os.path.join(out_folder, 'frame_%d_binary.png' % self.frame_n), binary_text_map)
            if text:
                # remove white trailing \n, just in case.
                # e.g. '那孩子眼中的光芒\n就跟他老爸是一个样'

                text = text.rstrip()
                start_n, end_n = self.get_start_end_n(self.frame_n)
                self.subs.append((start_n, end_n, text))
                # if out_folder:
                #     with open(os.path.join(out_folder, 'out.sub'), 'a') as f:
                with open(out_file, 'a') as f:
                    f.write('{%s}{%s}%s\n' % (start_n, end_n, text))

                self.frame_n = end_n  # we will then add frame_skip to this anyway
                pbar.n = self.frame_n
    
            self.frame_n += self.frame_skip
            pbar.update(self.frame_skip)
    
    def get_start_end_n(self, frame_n):
        """Finds the earliest frame that has text, and the latest frame that has text"""
        # optimal delta can be estimated by solving some equations with expected sub lengths
        start_n = self.get_last_same_frame(frame_n, delta=-5)
        end_n = self.get_last_same_frame(frame_n, delta=16)
        return start_n, end_n
        
    def get_last_same_frame(self, frame_n, delta=-10):
        """Returns the first frame that is different from the current frame.
        Skips delta (forward/back) frames at a time until it gets a different frame.
        It then backtracks one delta step to get the first frame that is different."""
        frame = get_frame_n(self.cap, frame_n)[1]
        binary = frame_to_binary_text_pixels(frame)

        bound = self.get_first_frame_delta_differing(binary, frame_n, delta)
        # we backtrack to start at prev which is guaranteed to be same as frame_n
        prev = bound - delta
        delta_neg = delta < 0
        # we seek the first differing frame, incrementing frame by frame.
        bound_closer = self.get_first_frame_delta_differing(binary, prev, frame_delta=-1 if delta_neg else 1)
        if delta_neg:
            return bound_closer + 1
        else:
            return bound_closer - 1
        
        

    def frames_different(self, binary1, binary2):
        log_and = binary1 & binary2
        num_pixels = max(binary1.sum(), binary2.sum())
        perc_equal = log_and.sum() / num_pixels
        return perc_equal < 0.8


    def get_first_frame_delta_differing(self, binary, frame_start, frame_delta=-10):
        frame_current = frame_start
        while frame_current >= 0 and frame_current < self.frame_max:
            frame_current += frame_delta
            frame = get_frame_n(self.cap, frame_current)[1]
            binary2 = frame_to_binary_text_pixels(frame)

            if self.frames_different(binary, binary2):
                return frame_current
        
        
        # offset by one, as really then the first differing frame is the NULL frame capping either end.
        if frame_current <= 0:
            return -1
        elif frame_current >= self.frame_max:
            return self.frame_max
        
    def show_frame(self, frame_n):
        ret, frame = get_frame_n(self.cap, frame_n)
        if not ret:
            return
        plt.imshow(frame)
        plt.show()
            



def process_output_text(fn):
    """Output was of form frame_start frame_end text.
    Unfortunately text itself can contain some \n.
    """
    import re

    file = 'ep1.txt'
    with open(fn, 'r') as f:
        stuff = f.readlines()

    # get indices of lines which begin with \d* \d* .*
    indices = [i for i, line in enumerate(stuff) if re.match('\d* \d* .*', line)]
    outputs = []

    for i in range(len(indices)):
        start_idx = indices[i]
        end_idx = indices[i+1] if i+1 < len(indices) else len(stuff)
        text = re.match('[{]\d*[@] \d* (.*)', stuff[start_idx]).group(1)
        start_frame, end_frame = re.match('(\d*) (\d*) .*', stuff[start_idx]).groups()
        for j in range(start_idx+1, end_idx):
            text += stuff[j]

        outputs.append((int(start_frame), int(end_frame), text))

    return outputs



def show_anns(anns, number=False):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True) if not number else anns
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for i, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
        if number:
            # add a little text in the middle
            midpoint = get_box_midpoint(ann['bbox'])
            ax.text(midpoint[0], midpoint[1], str(i), color='red', fontsize=10, ha='center', va='center', bbox=dict(facecolor=color_mask, edgecolor='none', alpha=1.0))

    ax.imshow(img)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax, color=None, marker=None):
    x0, y0 = box[0], box[1]
    w, h = box[2], box[3]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color if color else 'green', facecolor=(0,0,0,0), lw=2))  
    # add a little text in the middle
    if marker:
        ax.text(x0+w/2, y0+h/2, marker, color='red', fontsize=8, ha='center', va='center', bbox=dict(facecolor=color if color else 'green', edgecolor='none', alpha=1.0))

def get_box_midpoint(box):
    x0, y0 = box[0], box[1]
    w, h = box[2], box[3]
    return x0+w/2, y0+h/2

def mask_to_char(img, mask, bright_thresh=230):
    """
    Extracts the character from the image, given the mask.
    note that a valid character should have roughly a bbox of w, h = 38, 40
    problem is that sometimes only half of a two radical character is detected.

    """
    # mask is a nice output dict
    seg_mask = mask['segmentation']
    bbox = mask['bbox']
    x, y, w, h = bbox
    bbox_extract_seg_mask = seg_mask[y:y+h, x:x+w]
    bbox_extract = img[y:y+h, x:x+w]
    bbox_extract_gray = cv.cvtColor(bbox_extract, cv.COLOR_RGB2GRAY)
    img_bright = bbox_extract_gray > bright_thresh
    char_mask = img_bright & bbox_extract_seg_mask
    return char_mask  # binary mask of character



# ffmpeg .ts to .mp4
# ffmpeg -i input.ts output.mp4

if __name__  == '__main__':
    import sys
    if len(sys.argv) <2:
        print('Usage: python3 utils.py <input_video> <output_folder>')
        exit()

    infile = sys.argv[1]
    outfile = sys.argv[2]
    vse = VideoSubExtractor(infile)
    vse.get_subs(use_tqdm=True, out_file=outfile)




