import argparse
from utils import VideoSubExtractor

from segment_anything import SamPredictor, sam_model_registry
# sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cpu"


# predictor = SamPredictor(sam)

# ffmpeg .ts to .mp4
# ffmpeg -i input.ts output.mp4

if __name__  == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='Input video file')
    parser.add_argument('output_file', help='Output video file')
    parser.add_argument('--sam_model', help='Path to segmentation model, if empty then no SAM', default='')
    args = parser.parse_args()

    print(args.input_file)
    print(args.output_file)


    # vse = VideoSubExtractor('./out3.mp4')
    # vse.get_subs(use_tqdm=True, out_file_fn='ep1.txt')