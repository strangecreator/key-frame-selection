# local imports
from key_frame_selection import (
    select_from_video,
    select_from_frames_dir,
)
from key_frame_selection.types import PipelineConfig


config = PipelineConfig()


indices = select_from_video("path/to/video.mp4", cfg=config, max_frames=16)
print(indices)

# OR

indices = select_from_frames_dir("path/to/frames_dir", cfg=config, max_frames=16)
print(indices)