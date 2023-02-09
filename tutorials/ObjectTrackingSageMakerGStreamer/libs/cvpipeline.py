# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import gi
import threading
import numpy as np
gi.require_version('Gst', '1.0')
from gi.repository import Gst

class CVPipeline(threading.Thread):
    '''Base class for a gstreamer pipeline'''
    def __init__(self, pipeline):
        threading.Thread.__init__(self)
        self.running = False
        self.gst_pipeline = pipeline

    def stop(self):
        self.running = False

    def is_running(self):
        return self.running
    
    def run(self):
        '''Invoked as a thread to execute the gstreamer loop'''
        self.running = True

        Gst.init(None)
        self.pipeline = Gst.parse_launch(self.gst_pipeline)
        self.pipeline.get_by_name('input').get_static_pad('sink').add_probe(
            Gst.PadProbeType.BUFFER,
            self.__on_frame_probe__
        )
        self.pipeline.set_state(Gst.State.PLAYING)
        self.bus = self.pipeline.get_bus()
        try:
            while self.running:
                msg = self.bus.timed_pop_filtered(
                    Gst.SECOND,
                    Gst.MessageType.EOS | Gst.MessageType.ERROR
                )
                if msg:
                    text = msg.get_structure().to_string() if msg.get_structure() else ''
                    msg_type = Gst.message_type_get_name(msg.type)
                    print(f'{msg.src.name}: [{msg_type}] {text}')
                    self.stop()
        finally:
            self.pipeline.set_state(Gst.State.NULL)

    def __on_frame_probe__(self, pad, info):
        '''Handler that reads a buffer from gstreamer and loads a numpy rgb frame'''
        buf = info.get_buffer()
        caps = pad.get_current_caps()
        caps_structure = caps.get_structure(0)
        height, width = caps_structure.get_value('height'), caps_structure.get_value('width')
        pixel_bytes = 3
        is_mapped, map_info = buf.map(Gst.MapFlags.READ)
        if is_mapped:
            try:
                image_array = np.ndarray(
                    (height, width, pixel_bytes), dtype=np.uint8, buffer=map_info.data
                ).copy()
                self.process_frame(image_array, buf.pts)
            finally:
                buf.unmap(map_info)

        return Gst.PadProbeReturn.OK

    def process_frame(self, frame, timestamp):
        pass
