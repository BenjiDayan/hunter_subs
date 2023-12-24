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


def grow_thin_binary(frame):
    kernel = np.ones((3,3)) / 9  # A simple averaging kernel
    convolved_image = convolve(frame.astype(np.float64), kernel)

    thresh = threshold_otsu(convolved_image)
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

def ocr_frame(binary_frame):
    """Frame should be good looking text"""
    text_frame = 255 - 255*binary_frame
    cv.imwrite('temp.png', text_frame)

    result = reader.readtext('temp.png')

    result = '\n'.join([res[1] for res in result])

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


def quick_process_frame_n2(sam, cap, n, rgb2bgr=False, **kwargs):
    """Like quick_process_frame_n, but also displays original frame as an extra plot,
    and has all 3 plots in one subplot 1x3 grid"""
    ret, frame = get_frame_n(cap, n)
    if rgb2bgr:
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

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


    ocr = ocr_frame(output_mask)
    print(ocr)
    print('##########')
    ocr_sam = ocr_frame(output_mask_sam)
    print(ocr_sam)
    # print('##########')
    # ocr_sam_grow_direct = ocr_frame(output_mask_sam_grow_direct)
    # print(ocr_sam_grow_direct)
    print('##########')
    ocr_sam_grow = ocr_frame(output_mask_sam_grow)
    print(ocr_sam_grow)
    # return ocr, ocr_sam, ocr_sam_grow_direct, ocr_sam_grow
    return ocr, ocr_sam, ocr_sam_grow


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
    BRIGHT_THRESH = 240

    white = img_gray > BRIGHT_THRESH
    white_hp = filtered > HIGHPASS_THRESH
    
    # Take logical and.
    out = white & white_hp

    # This helps to remove little blobs around, and fill in some holes
    # Do growing
    out = grow_thin_binary(out)
    return out


def frame_to_binary_text_pixels_with_SAM(img, sam, bright_thresh=230, grow=False):
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(img)

    potential_masks = []

    for mask in masks:
        bbox = mask['bbox']
        x, y, w, h = bbox
        if 34 < w < 42 and 36 < h < 44:
            potential_masks.append(mask)

    output_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    for mask in potential_masks:
        # mask is a nice output dict
        seg_mask = mask['segmentation']
        bbox = mask['bbox']
        x, y, w, h = bbox

        bbox_extract_seg_mask = seg_mask[y:y+h, x:x+w]
        bbox_extract = img[y:y+h, x:x+w]
        bbox_extract_gray = cv.cvtColor(bbox_extract, cv.COLOR_RGB2GRAY)
        bbox_extract_img_bright = bbox_extract_gray > bright_thresh
        bbox_extract_char_mask = bbox_extract_img_bright & bbox_extract_seg_mask

        output_mask[y:y+h, x:x+w] = bbox_extract_char_mask

    # There's some weird bug if we do it in the loop above??
    if grow:
        output_mask = grow_thin_binary(output_mask)

    return output_mask
    


def frame_to_text(img, sam=None):
    """The frame may contain a horizontal line of text, or perhaps nothing.
    Actually that was for tesseract --psm 7. Now we are using easyocr, which
    is better at detecting text in images, and we just pass in everything."""
    if sam is None:
        binary_text_map = frame_to_binary_text_pixels(img)
    else:
        binary_text_map = frame_to_binary_text_pixels_with_SAM(img, sam, grow=True)

    # Do OCR
    ocr = ocr_frame(binary_text_map)
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

    def get_subs(self, max_n=None, use_tqdm=False, save_frames=False, out_folder=None):
        self.subs = []
        # make tqdm bar, setting to mock object if not tqdm
        pbar = tqdm(total=self.frame_max) if use_tqdm else MagicMock()
        pbar.n = self.frame_n
        while self.frame_n < (self.frame_max if max_n is None else max_n):
            ret, frame = get_frame_n(self.cap, self.frame_n)
            frame[3:-1, :]  # is now 88 Y 550 X
            if not ret:
                break
            text, binary_text_map = frame_to_text(frame, sam=self.sam)
            if save_frames:
                if out_folder:
                    cv.imwrite(os.path.join(out_folder, 'frame_%d.png' % self.frame_n), frame)
                    cv.imwrite(os.path.join(out_folder, 'frame_%d_binary.png' % self.frame_n), binary_text_map)
            if text:
                # remove white trailing \n, just in case.
                # e.g. '那孩子眼中的光芒\n就跟他老爸是一个样'

                text = text.rstrip()
                start_n, end_n = self.get_start_end_n(self.frame_n)
                self.subs.append((start_n, end_n, text))
                if out_folder:
                    with open(os.path.join(out_folder, 'out.sub'), 'a') as f:
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



def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
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
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2], box[3]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  


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
    vse = VideoSubExtractor('./out3.mp4')
    vse.get_subs(use_tqdm=True, out_file_fn='ep1.txt')




