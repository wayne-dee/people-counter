"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
##running converting model to IR
#python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json

# python main.py -i /resource/Pedestrian_Detect_2_1_1.mp4 -m frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

#python main.py -i /home/workspace/resources/Pedestrian_Detect_2_1_1.mp4 -m ./ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

import os
import sys
import time
import socket
import json
import cv2


import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60
input_stream = "Pedestrian_Detect_2_1_1.mp4"

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    
    

    
    parser = ArgumentParser()
#     parser._action_groups.pop()
#     required = parser.add_argument_group('required arguments')

#     optional = parser.add_argument_group('optional arguments')
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-pc", "--perf_counts", type=str, default=False,
                        help="Print performance counters")
    parser.add_argument("-c", "--color",type=str, default="BLUE", help="The color of the bounding boxes to draw; RED, GREEN,BLUE")
#     parser.add_argument("-c","--color" help="The color of the bounding boxes to draw; RED, GREEN or BLUE", default='BLUE')
                     
    return parser


def performance_counts(perf_count):
    """
    print information about layers of the model.

    :param perf_count: Dictionary consists of status of the layers.
    :return: None
    """
    print("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type',
                                                      'exec_type', 'status',
                                                      'real_time, us'))
    for layer, stats in perf_count.items():
        print("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer,
                                                          stats['layer_type'],
                                                          stats['exec_type'],
                                                          stats['status'],
                                                          stats['real_time']))




def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)


    return client

def convert_color(color_string):
    '''
    Get the BGR value of the desired bounding box color.
    Defaults to Blue if an invalid color is given.
    '''
    color = {"BLUE": (255,0,0), "GREEN": (0,255,0), "RED": (0,0,255)}
    out_color = color.get(color_string)
    if out_color:
        return out_color
    else:
        return color['BLUE']

def draw_boxes(frame, result):
    '''
    Draw bounding boxes onto the frame.
    '''
    current_count = 0
    for obj in result[0][0]:
        # Draw bounding box for object when it's probability is more than
        #  the specified threshold
        if obj[2] > prob_threshold:
            xmin = int(obj[3] * initial_w)
            ymin = int(obj[4] * initial_h)
            xmax = int(obj[5] * initial_w)
            ymax = int(obj[6] * initial_h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,0,255), 1)
            current_count = current_count + 1
    return frame, current_count

    
#     current_count = 0
#     for box in result[0][0]: # Output shape is 1x1x100x7
#         conf = box[2]
#         if conf >= args.prob_threshold:
#             xmin = int(box[3] * width)
#             ymin = int(box[4] * height)
#             xmax = int(box[5] * width)
#             ymax = int(box[6] * height)
#             cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), args.color, 1)
            
#             current_count = current_count + 1
            
#     return frame, current_count

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    #convert args for color and threshold
#     color = convert_color(args.color)
    
    cur_request_id = 0
    last_count = 0
    total_count = 0
    start_time = 0

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device,1,cur_request_id, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()

    ### TODO: Handle the input stream ###
   
    image_flag = False
    ##check the input if is a webcam
    if args.input == 'CAM':
        input_stream = 0
        
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        image_flag = True
        input_stream = args.input
    
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

       
    cap = cv2.VideoCapture(input_stream)
    
    if input_stream:#to be changed
        cap.open(args.input)
    if not cap.isOpened():#to be changed
        log.error("ERROR! Unable to open video source")
    global initial_w, initial_h
    initial_w = cap.get(3)
    initial_h = cap.get(4)
    
            
    ### TODO: Loop until stream is over ###
    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        ##read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        # Pre-process the frame
        frame = cv2.resize(frame, (100,100))
        frame = frame.transpose((2,0,1))
        frame = frame.reshape(1, *frame.shape)
#         frame = cv2.Canny(frame, 100,100)

        ### TODO: Start asynchronous inference for specified request ###
        inf_start = time.time()
        infer_network.async_inference(cur_request_id,frame)

        ### TODO: Wait for the result ###
        # Get the output of inference
        if infer_network.wait(cur_request_id) == 0:
            det_time = time.time() - inf_start
            ### TODO: Get the results of the inference request ###
            result = infernetwork.extract_output(cur_request_id)
#             result = infer_network.get_output(cur_request_id)
            if args.perf_counts:
                perf_count = infer_network.performance_counter(cur_request_id)
                performance_counts(perf_count)

            ### TODO: Extract any desired stats from the results ###
            #out_frame, classes = draw_boxes(result, width, height)
            frame, current_count = draw_boxes(frame, result, args, width, height)
            inf_time_message = "Inference time: {:.3f}ms"\
                               .format(det_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            
            # When new person enters the video
            if current_count > last_count:
                start_time = time.time()
                total_count = total_count + current_count - last_count
                client.publish("person", json.dumps({"total": total_count}))
            
             # Person duration in the video is calculated
            if current_count < last_count:
                duration = int(time.time() - start_time)
                # Publish messages to the MQTT server
                client.publish("person/duration",
                               json.dumps({"duration": duration}))

            client.publish("person", json.dumps({"count": current_count}))
            last_count = current_count
            
           
        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(out_frame)
        sys.stdout.flush()
        
        ### TODO: Write an output image if `single_image_mode` ###
        if image_flag:
            cv2.imwrite('output_image.jpg', frame)
        
        
        #extra info
        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    ### TODO: Disconnect from MQTT
    client.disconnect()



def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args,client)
    


if __name__ == '__main__':
    main()
