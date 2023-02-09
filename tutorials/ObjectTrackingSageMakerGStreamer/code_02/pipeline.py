# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import os
import sys
import time
import argparse
sys.path.append("/opt/ml/processing/input/libs/bytetrack")
sys.path.append("/opt/ml/processing/input/libs")
from smcvpipeline import SageMakerCVPipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'CV Pipeline for ML',
                    description = 'Video streaming processing with ML')
    
    parser.add_argument('-k','--enable-tracking', type=bool, help="Enable object tracking", default=True)
    parser.add_argument('-j','--jpeg-quality', type=int, help="Quality of the jpeg mosaic sent to SM endpoint", default=90)
    parser.add_argument('-e','--endpoint-name', type=str, help="SageMaker Endpoint Name", required=True)
    parser.add_argument('-r','--region-name', type=str, help="Region Name", default="us-east-1")
    parser.add_argument('-p','--preds-per-output-file', type=int, help="Number of predictions per output file", default=150)
    parser.add_argument('-w','--num-workers', type=int, help="Number of workers that will invoke the model", default=5)
    parser.add_argument('-n','--cams-per-row', type=int, help="Number of cams per row", default=2)
    parser.add_argument('-m','--max-cams-per-batch', type=int, help="Max number of cams per batch", default=4)
    parser.add_argument('-i','--input-shape', type=int, help="Resized resolution of the feeds", nargs=2, default=[1280, 720])
    parser.add_argument('-t','--tile-size', type=int, help="Shape of each tile in the mosaic", nargs=2, default=[960, 540])
    parser.add_argument('-c','--conf-thres', type=float, help="Confidence threshold of the object", default=0.15)
    parser.add_argument('-o','--iou-thres', type=float, help="Confidence threshold of the IoU ", default=0.45)
    
    args = parser.parse_args()
    print(args)
    
    cams_per_row=args.cams_per_row
    max_cams_per_batch=args.max_cams_per_batch
    raw_width,raw_height=args.input_shape
    input_dir = "/opt/ml/processing/input"
    output_dir = "/opt/ml/processing/output"    
    failure_file = output_dir + "/failure"
    pipeline = None
    if not os.path.isdir(output_dir): os.makedirs(output_dir)
    
    # Tracking requires sequential processing, that's why we can have only 1 active worker
    if args.enable_tracking and args.num_workers > 1:
        print(f"Tracking enabled. Setting num_workers to 1. Current: {args.num_workers}")
        args.num_workers = 1
    
    try:        
        # list the pipes in the input dir
        # parse the manifest file and extract all file names
        file_names = [f.strip() for f in open(f'{input_dir}/data/input-1-manifest', 'r').readlines()[1:]] # skip first line
        num_batches = ((len(file_names)-1)//max_cams_per_batch) + 1
        print(f"Num files: {len(file_names)}, Num batches: {num_batches}")
        
        for batch in range(num_batches):
            start = batch * max_cams_per_batch
            end = start + min(max_cams_per_batch,len(file_names[start:]))
            
            mosaic,sources = [],[]
            for i,s3_path in enumerate(file_names[start:end]):
                # convert the s3 path to the expected by awss3src
                s3_path = s3_path.replace('s3://', f's3://{args.region_name}/')
                
                xoff,yoff = raw_width * (i%cams_per_row), raw_height * (i//cams_per_row)
                mosaic.append(f"sink_{i}::xpos={xoff} sink_{i}::ypos={yoff}")
                sources.append(f"\n    awss3src uri={s3_path} ! decodebin ! videoconvert ! video/x-raw,format=(string)BGR ")
                sources.append(f"! videoscale method=0 add-borders=false ! video/x-raw,width={raw_width},height={raw_height} ! queue2 max-size-buffers=1000 ! comp.sink_{i}")
            mosaic,sources = " ".join(mosaic), "".join(sources)
            pre_pipeline = f"""
            compositor name=comp {mosaic}
            ! videoconvert ! video/x-raw,format=BGR ! fakesink name=input {sources}
            """
            print(pre_pipeline)
            params = (
                pre_pipeline, args.endpoint_name, args.region_name, max_cams_per_batch, 
                output_dir, args.tile_size, args.conf_thres, args.iou_thres, 
                args.num_workers, args.preds_per_output_file, args.jpeg_quality,
                args.enable_tracking
            )
            pipeline = SageMakerCVPipeline(*params)
            t = time.time()
            pipeline.start()
            pipeline.join()
            print(f"Total time: {time.time()-t}")
    except Exception as e:                        
        print(f"ERROR: {sys.exc_info()[0]} {e}")
        with open(failure_file, 'w') as f:
            f.write(str(e))
        raise e
    finally:
        if not pipeline is None and pipeline.is_running(): # should not happen
            print('Stopping pipeline...')
            pipeline.stop()
            pipeline.join()
