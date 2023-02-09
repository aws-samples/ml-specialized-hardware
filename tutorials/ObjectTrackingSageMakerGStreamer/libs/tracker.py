# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import cv2
import sys
import numpy as np
from yolox.tracker.byte_tracker import BYTETracker

# Helper class that emulates argparse
class AllMyFields:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)

class Tracker(object):
    def __init__(self, frame_rate=25, track_tresh=0.25, track_buffer=30, match_tresh=0.5, min_box_area=10):
        self.args = AllMyFields({
            'track_thresh': track_tresh,
            'track_buffer': track_buffer,
            'match_thresh': match_tresh,
            'mot20': False,
            'min_box_area': min_box_area
        })
        self.tracker = BYTETracker(self.args, frame_rate=frame_rate)
        self.online_targets = None

    def render(self, frame, objects):
        '''Render BBoxes & ID to an image'''
        for obj_id,xyxy,score in objects:
            x1,y1,x2,y2 = xyxy
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,0), 3)
            cv2.putText(frame, f'{obj_id}', (x1+50,y1+50), cv2.FONT_HERSHEY_SIMPLEX,
                   2, (0, 0, 255), 2, cv2.LINE_AA)
            
    def step(self, detections, img_size=(960,540)):
        '''Update the tracker based on predictions
        Detections[ [x1,y1,x2,y2,conf] ]
        '''
        self.online_targets = self.tracker.update(detections, [img_size[1], img_size[0]], [img_size[1], img_size[0]])
        results = []
        for t in self.online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                x1,y1,bw,bh = tlwh.astype(np.int32)
                xyxy = [x1,y1,x1+bw,y1+bh]
                # obj_id, bbox, conf
                results.append((tid, xyxy, t.score))
        return results
