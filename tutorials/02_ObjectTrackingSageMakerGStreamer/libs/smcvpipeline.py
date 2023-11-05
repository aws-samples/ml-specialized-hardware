# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import io
import os
import time
import json
import boto3
import queue
import tracker
import threading
import cvpipeline
import numpy as np
from turbojpeg import TurboJPEG

class SageMakerCVPipeline(cvpipeline.CVPipeline):
    def __init__(self, pipeline, endpoint_name, region_name, max_cams_per_batch, output_dir, 
                 tile_size=(960,540), conf_thres=0.15, iou_thres=0.45, max_workers=5, 
                 preds_per_file=100, jpeg_quality=90, enable_tracking=False):
        super().__init__(pipeline)
        
        self.endpoint_name = endpoint_name
        self.jpeg = TurboJPEG()
        self.jpeg_quality = jpeg_quality
        self.frames = queue.Queue()
        self.sm_client = boto3.client('sagemaker-runtime', region_name=region_name)
        self.endpoint_name = endpoint_name
        self.output_dir = output_dir
        self.tile_size = tile_size
        self.params = json.dumps({
            "tile_width": tile_size[0],#960,
            "tile_height": tile_size[1],#540,
            "conf_thres": conf_thres, #0.15,
            "iou_thres": iou_thres, #0.45
        })
        self.cache = []
        self.workers = [threading.Thread(target=self.__worker__, args=(i,)) for i in range(max_workers)]
        self.trackers = [tracker.Tracker() for i in range(max_cams_per_batch)] if enable_tracking else None
        self.preds_per_file = preds_per_file
        self.cache_lock = threading.Lock()
        self.cache_counter = 0
        if not os.path.isdir(self.output_dir): os.mkdir(self.output_dir)

    def run(self):
        self.running = True
        # initialize workers
        for w in self.workers: w.start()
        # run gstreamer main loop
        super(SageMakerCVPipeline, self).run()
        # wait for all workers to finalize
        for w in self.workers: w.join()
        # dump to disk the last pending predictions
        self.dump_cache(True)

    def dump_cache(self, flush=False):
        '''Saves predictions to disk as compressed numpy filers'''
        if flush or len(self.cache) >= self.preds_per_file:
            dump = False
            self.cache_lock.acquire()
            # check if there are predictions to be flushed
            if len(self.cache) > 0:
                cache,self.cache,pred_file_id,dump = self.cache,[],self.cache_counter,True
                self.cache_counter += 1
            self.cache_lock.release()
            if dump:
                # ok. there are predictions, save to a file
                print(f'Dumping {len(cache)}... ')
                np.savez(os.path.join(self.output_dir, f'pred_{pred_file_id:05d}.npz'), cache)

    def __worker__(self, worker_id):
        '''A worker will keep listening to a queue for frames to process'''
        while self.running:
            if self.frames.empty():
                time.sleep(0.1)
            else:
                # alrigth, there is a new frame
                frame,timestamp = self.frames.get()
                with io.BytesIO() as resp:
                    # invoke the endpoint and keep the predictions
                    resp.write(self.sm_client.invoke_endpoint(
                        EndpointName=self.endpoint_name,
                        Body=frame,
                        ContentType="image/jpeg",
                        Accept="application/x-npy",
                        CustomAttributes=self.params
                    )['Body'].read())
                    resp.seek(0)
                    # resp format: [cam_id, obj_id, obj_cls, conf, bbox(x1,y1,x2,y2) pose(x1,y1,conf1,...,x17,y17,conf17)]
                    preds = np.load(resp).astype(np.object)
                    data = [timestamp, preds, []]
                    if not self.trackers is None:
                        dets = []
                        for pred in preds:
                            cam_id,conf,bbox = pred[0],pred[3],pred[4:8]
                            dets.append(np.hstack((bbox, [conf])))
                        dets = np.array(dets).astype(np.object)
                        data[2].append(self.trackers[int(cam_id)].step(dets, self.tile_size))

                    self.cache_lock.acquire()
                    self.cache.append(data)
                    self.cache_lock.release()
                    self.dump_cache()

    def process_frame(self, frame, timestamp):
        '''Concrete implementation of frame processing'''
        # a mosaic will be encoded as jpeg outside gstreamer for max performance
        frame = self.jpeg.encode(frame, quality=self.jpeg_quality)
        self.frames.put((frame,timestamp))
