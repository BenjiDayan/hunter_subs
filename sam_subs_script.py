import argparse
from utils import VideoSubExtractor

from segment_anything import SamPredictor, sam_model_registry
import os

# sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cpu"


# predictor = SamPredictor(sam)

# ffmpeg .ts to .mp4
# ffmpeg -i input.ts output.mp4

if __name__  == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='Input video file')
    parser.add_argument('output_folder', help='Output folder name')
    parser.add_argument('--sam_model', help='Path to segmentation model, if empty then no SAM', default='')
    args = parser.parse_args()

    print(args.input_file)
    print(args.output_folder)


    if args.sam_model:  # e.g. ./sam_vit_h_4b8939.pth
        model_type = "vit_h"
        sam = sam_model_registry[model_type](args.sam_model)
        device = 0
        sam.to(device=device)


    os.makedirs(args.output_folder, exist_ok=True)


    vse = VideoSubExtractor(args.input_file, sam=sam)
    vse.get_subs(use_tqdm=True, save_frames=True, out_folder=args.output_folder)
