- should we try to train a mini segmenter?
- should we try to fix some of the issues in SAM: for instance yi (one) bounding box too small, or xiao (small) split bounding boxes
    - some kind of detection mechanism whereby if there's a horizontal line of text, then any area that isn't captured but is either in the middle or either end of the line should be maybe reconsidered
    

    - argument specifying (x,y) centering of subtitle lines
    - check box width and height function to check if bbox is right shape
    - check box centering function to check if bbox is on a line
    - group tgt bboxes of same line
    - within a line of bboxes:
      - Want to remove intersecting bboxes, if these exist
      - Want to fill in holes in the line, if these exist


# First goal: write code to plot all bboxes that have <= box sized width and within line tramlines.

