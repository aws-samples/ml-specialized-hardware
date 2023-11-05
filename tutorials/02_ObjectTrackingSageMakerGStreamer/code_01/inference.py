# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
os.environ['NEURON_RT_NUM_CORES'] = '4'
import io
import cv2
import json
import time
import torch
import torch.neuron
import numpy as np

from turbojpeg import TurboJPEG

class Detector(object):
    '''Main class responsible for pre/post processing + model invocation'''
    def __init__(self, model_path):
            
        self.model = torch.jit.load(model_path).eval()
        self.jpeg = TurboJPEG()
        
        print(f'Model loaded')

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] 
        # where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    # non maximum suppression. Inspired by torchvision.nms
    def nms(self, bboxes, scores, iou_threshold=0.45):
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1) 
        order = scores.ravel().argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        bboxes = bboxes[keep]
        scores = scores[keep]
        return bboxes, scores, keep

    def non_max_suppression_kpt(self, prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False,
                            labels=(), kpt_label=False, nc=None, nkpt=None):
        """Runs Non-Maximum Suppression (NMS) on inference results

        Returns:
             list of detections, on (n,6) tensor per image [xyxy, conf, cls, keypoints]
        """
        if nc is None:
            nc = prediction.shape[2] - 5  if not kpt_label else prediction.shape[2] - 56 # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_det = 300  # maximum number of detections per image
        max_nms = 30000  # maximum number of boxes
        time_limit = 10.0  # seconds to quit after

        t = time.time()
        output = [np.zeros((0,57))] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints        
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                l = labels[xi]
                v = np.zeros((len(l), nc + 5))
                v[:, :4] = l[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
                x = np.concatenate((x, v), axis=0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:5+nc] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])

            if not kpt_label:
                conf = x[:, 5:].max(axis=1, keepdims=True)
                j = np.argmax(x[:, 5:], axis=1).reshape(x[:, 5:].shape[0],-1)
                x = np.concatenate((box, conf, j), axis=1)[conf.ravel() > conf_thres]
            else:
                kpts = x[:, 6:]
                conf = x[:, 5:6].max(axis=1, keepdims=True)                
                j = np.argmax(x[:, 5:6], axis=1).reshape(x[:, 5:6].shape[0],-1)
                x = np.concatenate((box, conf, j, kpts), axis=1)[conf.ravel() > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == classes).any(1)]

            # Check shape
            n = x.shape[0]  # number of boxes        
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort()[::-1][:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            
            boxes,scores,i = self.nms(boxes, scores, iou_thres)  # NMS
            
            #if len(i) > max_det:  # limit detections
            #    i = i[:max_det]
            #    boxes = boxes[:max_det]
            #    scores = scores[:max_det]

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                print(f'WARNING: NMS time limit {time_limit}s exceeded')
                break  # time limit exceeded
        return output
 
    def predict(self,x):
        with torch.no_grad():
            return self.model(x).numpy()#torch.from_numpy(x)).numpy()

    def preprocess(self, img, img_size=960):
        '''Make the image squared and prepare the tensor as [B,C,H,W]'''
        h,w,c = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if h!=w:
            max_size=max(h,w)
            img_sqr = np.zeros((max_size, max_size,c), dtype=np.uint8)
            img_sqr[0:h,0:w],img = img[:],img_sqr
        x = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        return x
        #x = np.expand_dims((x.transpose(2,0,1) / 255.0).astype(np.float32), axis=0)
        #return np.ascontiguousarray(x)

    def postprocess(self, output, tensor_shape, img_shape, conf_thres=0.15, iou_thres=0.45, nc=1, nkpt=17):
        '''Run NMS to filter bboxes & return detections with keypoints'''
        detections = self.non_max_suppression_kpt(
            output, conf_thres,iou_thres, nc=nc, nkpt=nkpt, kpt_label=True)
        
        # targets in the format
        # [det_index, int(class_id), [x1,y1,x2,y2], conf, [x0,y0,conf0...x16,y16,conf16]]
        targets = []
        for i,det in enumerate(detections):
            bboxes,scores,classes,keypoints = det[:, :4],det[:, 4], det[:, 5],det[:,6:]
            bboxes = bboxes.clip(0,tensor_shape[0])
            # rescale bboxes and poses
            # fix the distortion provoked by preprocess
            tw,th,ih,iw = *tensor_shape, *img_shape
            bboxes = bboxes / [tw,th,tw,th] * [iw,ih,iw,ih]        
            keypoints = (keypoints / ([tw,th,1]*nkpt)) * ([ih,iw,1]*nkpt)
            dets = []
            for index, (box, conf, cls, pose) in enumerate(zip(bboxes,scores,classes,keypoints)):
                dets.append([index, int(cls), box.astype(np.int32), conf, pose])
            if len(dets)>0: targets.append(dets)
        return targets
    
    def mosaic2batch(self, data, tile_width=960, tile_height=540, img_size=960):
        mosaic = self.jpeg.decode(data)
        h,w,c = mosaic.shape

        max_size=max(tile_width, tile_height)
        min_size=min(tile_width, tile_height)
        num_pixels = max_size*max_size*3
        batch_size = h//tile_height * w//tile_width
        batch = torch.zeros(max_size*max_size*c * batch_size, dtype=torch.float32)
        ttl_pixels=0
        # build a batch out of the tiles
        for row in range(h//tile_height):
            for col in range(w//tile_width):
                pw,ph=col*tile_width,row*tile_height
                tile = mosaic[ph:ph+tile_height, pw:pw+tile_width]
                
                tile = self.preprocess(tile, img_size)
                
                batch[ttl_pixels:ttl_pixels + num_pixels] = torch.from_numpy(tile).ravel()
                ttl_pixels = ttl_pixels + num_pixels

        batch = batch.reshape(-1,max_size,max_size,c)
        batch = (batch / 255.0).float() # to FP32
        batch = batch.permute(0,3,1,2) # NHWC --> NCHW

        return batch
    
## SAGEMAKER FUNCTIONS ##
# The following functions are invoked by SageMaker to load the model, 
# receive the payload, invoke the model and prepare the output
def model_fn(model_dir):
    return Detector(os.path.join(model_dir, 'model.pt'))

def input_fn(data, content_type, context=None):
    if content_type != 'image/jpeg':
        raise Exception(f'Invalid data type. Expected image/jpeg, got {content_type}')
   
    try:
        custom_attributes = context.get_request_header(0,'X-Amzn-SageMaker-Custom-Attributes')
        params = json.loads(custom_attributes)
        return data, params
    except Exception as e:
        raise Exception(f"You need to pass Custom Attributes")

def output_fn(predictions, accept, context=None):
    if accept!='application/x-npy':
        raise Exception(f'Invalid data type. Expected application/x-npy, got {accept}')

    with io.BytesIO() as b:   
        data = []
        for i,objs in enumerate(predictions):
            for obj_id, obj_cls, bbox, conf, pose_kpts in objs:
                data.append(np.hstack([
                    [i, obj_id, obj_cls],
                    bbox.astype(np.float32),
                    pose_kpts
                ]))
        np.save(b, np.vstack(data))
        b.seek(0)
        return b.read()

def predict_fn(data, detector, context=None):
    mosaic,params = data
    # adjust img_size accordinly with the input shape of your model
    img_size=960
    tile_width=params.get('tile_width', 960)
    tile_height=params.get('tile_height', 540)
    conf_thres=params.get('conf_thres', 0.15)
    iou_thres=params.get('iou_thres', 0.45)
    
    x = detector.mosaic2batch(mosaic, tile_width, tile_height, img_size)
    out = detector.predict(x)
    detections = detector.postprocess(out, x.shape[2:], (tile_height, tile_width), conf_thres, iou_thres)
    return detections
