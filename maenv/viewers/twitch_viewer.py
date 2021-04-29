from __future__ import print_function
from twitchstream.outputvideo import TwitchBufferedOutputStream

import time

class TwitchViewer:
    def __init__(self, stream_key, width=640, height=480):
        self.width = width
        self.height = height
        self.buffered_output_stream = TwitchBufferedOutputStream(
            twitch_stream_key=stream_key,
            width=width,
            height=height,
            fps=30.,
            verbose=False)

    def send_frame(self, frame_data):
        if frame_data.shape != (self.height, self.width, 3):
            frame_data = frame_data.transpose((1, 0, 2))
        if self.buffered_output_stream.get_video_frame_buffer_state() < 30:
            self.buffered_output_stream.send_video_frame(frame_data)
        else:
            time.sleep(.001)
