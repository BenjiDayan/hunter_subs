import os
import time
import utils
API_KEY = os.environ['GEMINI_API_KEY']
import google.generativeai as genai
model = genai.GenerativeModel("gemini-1.5-flash")

import subprocess
TEMP_SLICE_FN = 'temp_subtitle_slice.mp4'
def extract_subtitle_slice(fn):
    mp4_fn = fn.replace('.ts', '.mp4')
    # ffmpeg -i ./LR_Chinese_001_720P\[52KHD\].ts -filter:v "crop=530:93:380:555" -c:a copy out3.mp4
    subprocess.run(['ffmpeg', '-y', '-i', mp4_fn, '-filter:v', 'crop=530:93:380:555', '-c:a', 'copy', TEMP_SLICE_FN])

import re



def split_text_lines(text: str) -> list[str]:
    """Converts {s}{e}sub\nsub to {s}{e}sub\\nsub"""
    if not '{' in text:
        return ''
    text = text[text.index('{'):]
    lines = text.split('\n')
    out = []
    for line in lines:
        if line == '':
            continue
        if line[0] == '{':
            out.append(line)
        else:
            out[-1] += '\\n' + line
    return out


def extract_frame_subset(contents, start_frame=3400, end_frame=41000):
    """extracts a subset of frames from a subtitle file and returns the subset as a string
    """
    out = []

    s_e_sub = extract_frames(contents)
    for s, e, sub in s_e_sub:
        if start_frame <= s <= end_frame:
            out.append(f"{{{s}}}{{{e}}}{sub}")

    return ''.join(out)

def extract_frames(contents):
    """converts frames to a list of tuples (start_frame, end_frame, subtitle)"""
    import re
    ptrn = re.compile(r'([{]\d*[}][{]\d*[}])(.*[\r\n][^{]*)',re.MULTILINE)
    frame_sub_pairs = re.findall(ptrn, contents)
    out = []
    for frame, sub in frame_sub_pairs:
        frame_num = int(frame[1:-1].split('}{')[0])
        frame_num_end = int(frame[1:-1].split('}{')[1])
        out.append((frame_num, frame_num_end, sub))
    return out


if __name__ == '__main__':
    # example usage:
    # python gemini_test.py ep1_out_26_08_2024_subset.sub vids/LR_Chinese_001_720P[52KHD].mp4 ep1_out_26_08_2024_subset_gemini_simplified.sub
    # python gemini_test.py ./vids/LR_Chinese_003_720P\[52KHD\].ts ./vids/ep3_ocr.sub ./vids/ep3_gemini.sub   


    # above as a script with argparse for file names
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_mp4_file', type=str)
    parser.add_argument('input_sub_file', type=str)
    parser.add_argument('output_sub_file', type=str)
    args = parser.parse_args()

    fn = args.input_mp4_file
    mp4_fn = args.input_mp4_file.replace('.ts', '.mp4')
    print(mp4_fn)
    subprocess.run(['ffmpeg', '-y', '-i', fn, '-c:a', 'copy', mp4_fn])

    whole_vid = genai.upload_file(mp4_fn)
    print(whole_vid.name)

    extract_subtitle_slice(args.input_mp4_file)
    foo = utils.VideoSubExtractor(TEMP_SLICE_FN)
    foo.get_subs(max_n=None, use_tqdm=True, out_file=args.input_sub_file, trad2simple=False)


    with open(args.input_sub_file) as file:
        my_subs = file.read()
        my_subs = extract_frame_subset(my_subs)


    # # wait for whole_vid to be processed
    while True:
        if genai.get_file(whole_vid.name).state.value == 2:
            break
        print('waiting for video to be processed...')
        time.sleep(5)

    prompt = """
    Please help me extract the subtitles of this video in traditional chinese.
    I already have an initial transcription using OCR, but it has some mistakes - some characters may be missing or incorrectly transcribed.
    Please correct the transcribed subtitles by matching to the audio and video; change and add characters as needed, but keep the timestamps the same.
    The characters are in traditional chinese, no need to convert them to simplified chinese.
    
    I.e. given input:
    {start_frame1}{end_frame1}{subtitle1}
    {start_frame2}{end_frame2}{subtitle2}
    ...

    Please output:
    {start_frame1}{end_frame1}{corrected_subtitle1}
    {start_frame2}{end_frame2}{corrected_subtitle2}
    ...

    **Initial OCR transcription**:
    """ + my_subs


    result_final = model.generate_content(
        [whole_vid, "\n\n", prompt]
    )
    print(f"{result_final.text=}")

    # from unittest.mock import MagicMock
    # result_final = MagicMock()
    # with open('temp.txt') as file:
    #     result_final.text = file.read()
    #     # convert \\n in the output to \n
    #     result_final.text = result_final.text.replace('\\n', '\n')

    import opencc
    # traditional chinese to simplified
    cc = opencc.OpenCC('t2s')
    out_simplified = cc.convert(result_final.text)
    
    a = extract_frames('\n'.join(split_text_lines(out_simplified)))
    b = extract_frames('\n'.join(split_text_lines(my_subs)))
    c = []
    i, j = (0, 0)
    while i < len(a) and j < len(b):
        s1, e1, sub1 = a[i]
        s2, e2, sub2 = b[j]
        if s1 == s2 and e1 == e2:
            c.append((s1, e1, sub1))
            i += 1
            j += 1
        else:
            # make sure the LLM hasn't gone backwards
            if len(c) > 0 and s1 < c[-1][0]:
                i += 1
            elif s1 < s2:
                i += 1
                c.append((s1, e1, sub1))
            elif s1 > s2:
                j += 1
            else:  # s1 == s2 but e1 != e2
                i += 1
                j += 1
                c.append((s1, e1, sub1))
    out = ''.join([f'{{{s}}}{{{e}}}{sub}' for s, e, sub in c])

    with open(args.output_sub_file, 'w') as file:
        file.write(out)