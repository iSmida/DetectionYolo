from __future__ import print_function
import sys
import os
import argparse
import numpy as np
if '/data/software/opencv-3.4.0/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/data/software/opencv-3.4.0/lib/python2.7/dist-packages')
if '/data/software/opencv-3.3.1/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/data/software/opencv-3.3.1/lib/python2.7/dist-packages')
import cv2
import torch
import timeit
from lib.ssds import ObjectDetector
from lib.utils.config_parse import cfg_from_file

#change the parse_args.cfg, change VOC_CLASSES
'''
VOC_CLASSES = ( 'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')
'''
VOC_CLASSES = ( 'face')
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Demo a ssds.pytorch network')
    parser.add_argument('--cfg', dest='confg_file',
            help='the address of optional config file', default = './experiments/cfgs/yolo_v3_mobilenetv2_voc-6-26-0.yml', type=str)
    parser.add_argument('--demo', dest='demo_file',
            help='the address of the demo file', default='/data/datasets/101/11.mp4', type=str)
    parser.add_argument('-t', '--type', dest='type',
            help='the type of the demo file, could be "image", "video", "camera" or "time", default is "image"', default='video', type=str)
    parser.add_argument('-d', '--display', dest='display',
            help='whether display the detection result, default is True', default=True, type=bool)
    parser.add_argument('-s', '--save', dest='save',
            help='whether write the detection result, default is False', default=True, type=bool) 
    
    #if len(sys.argv) == 1:
     #  parser.print_help()
     #   sys.exit(1)
    
    args = parser.parse_args()
    
    return args


COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

def demo(args, image_path):
    # 1. load the configure file
    cfg_from_file(args.confg_file)

    # 2. load detector based on the configure file
    object_detector = ObjectDetector()
    args = parse_args()

    # 3. load image
    image = cv2.imread(image_path)

    # 4. detect
    _labels, _scores, _coords = object_detector.predict(image)

    # 5. draw bounding box on the image
    for labels, scores, coords in zip(_labels, _scores, _coords):
        cv2.rectangle(image, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), COLORS[labels % 3], 2)
        cv2.putText(image, '{label}: {score:.3f}'.format(label=VOC_CLASSES[labels], score=scores), (int(coords[0]), int(coords[1])), FONT, 0.5, COLORS[labels % 3], 2)
    
    # 6. visualize result
    if args.display is True:
        cv2.imshow('result', image)
        

    # 7. write result
    if args.save is True:
        path, _ = os.path.splitext(image_path)
        cv2.imwrite(path + '_result.jpg', image)
    

def demo_file(args, imagelist,imagefile):
    # 1. load the configure file
    cfg_from_file(args.confg_file)
    # 2. load detector based on the configure file
    object_detector = ObjectDetector()
    args = parse_args()
    for image_name in imagelist:
    # 3. load image
        image_path = os.path.join(imagefile,image_name)
        image = cv2.imread(image_path)

    # 4. detect
        _labels, _scores, _coords = object_detector.predict(image)

    # 5. draw bounding box on the image
        for labels, scores, coords in zip(_labels, _scores, _coords):
            cv2.rectangle(image, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), COLORS[labels % 3], 2)
            cv2.putText(image, '{label}: {score:.3f}'.format(label=VOC_CLASSES[labels], score=scores), (int(coords[0]), int(coords[1])), FONT, 0.5, COLORS[labels % 3], 2)
    
    # 6. visualize result
        if args.display is True:
            cv2.imshow('result', image)
            cv2.waitKey(800)
        '''
    # 7. write result
       # if args.save is True:
        path = './'
        path = os.path.join(path,image_name)
        print(path)
        #path, _ = os.path.splitext(image_path)
        #print(path)
        cv2.imwrite(path, image)
        '''
        '''
        if args.save is True:
            path = os.path.join(imagefile,'result')
            path = os.path.join(path,image_name)
            #path, _ = os.path.splitext(image_path)            
            cv2.imwrite(path, image)
        '''   
        
        
        
def demo_live(args, video_path):
    # 1. load the configure file
    cfg_from_file(args.confg_file)

    # 2. load detector based on the configure file
    object_detector = ObjectDetector()

    # 3. load video
    cam = cv2.VideoCapture(0)
    #cam.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
    #cam.set(cv2.CAP_PROP_FRAME_WIDTH,1920)frametime10

    index = -1
    frametime20 = np.zeros((100,1))
    index = 0
    while True:
        #index = index + 1
        #sys.stdout.write('Process image: {} \r'.format(index))
        #sys.stdout.flush()

        # 4. read image
        #flag, image = video.read()
        retval,image = cam.read()
        #
        #image = cv2.resize(image,(192,108))
        #image = image[270:270+540,480:480+540,:]
        print(image.shape)
        if retval == False:
            print("Can not read image in Frame : {}".format(index))
            break
        t0 = timeit.default_timer()
        # 5. detect
        _labels, _scores, _coords = object_detector.predict(image)

        # 6. draw bounding box on the image
        for labels, scores, coords in zip(_labels, _scores, _coords):
            cv2.rectangle(image, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), COLORS[labels % 3], 2)
            cv2.putText(image, '{label}: {score:.3f}'.format(label=VOC_frametime10CLASSES[labels], score=scores), (int(coords[0]), int(coords[1])), FONT, 0.5, COLORS[labels % 3], 2)
        elapsed = timeit.default_timer() - t0
        frametime20[index] = elapsed
        index = index+1
        index = index%100
        print(elapsed)
        # 7. visualize result
        cv2.imshow('result', image)
        cv2.waitKey(1)
        '''
        # 8. write result
        if args.save is True:
            path, _ = os.path.splitext(video_path)
            path = path + '_result'
            if not os.path.exists(path):
                os.mkdir(path)
            cv2.imwrite(path + '/{}.jpg'.format(index), image)        
            '''

def time_benchmark(args, image_path):
    # 1. load the configure file
    cfg_from_file(args.confg_file)

    # 2. load detector based on the configure file
    object_detector = ObjectDetector()

    # 3. load imagedemo
    image = cv2.imread(image_path)

    # 4. time test
    warmup = 20
    time_iter = 100
    print('Warmup the detector...')
    _t = list()
    for i in range(warmup+time_iter):
        _, _, _, (total_time, preprocess_time, net_forward_time, detect_time, output_time) \
            = object_detector.predict(image, check_time=True)
        if i > warmup:
            _t.append([total_time, preprocess_time, net_forward_time, detect_time, output_time])
            if i % 20 == 0: 
                print('In {}\{}, total time: {} \n preprocess: {} \n net_forward: {} \n detect: {} \n output: {}'.format(
                    i-warmup, time_iter, total_time, preprocess_time, net_forward_time, detect_time, output_time
                ))
    total_time, preprocess_time, net_forward_time, detect_time, output_time = np.sum(_t, axis=0)/time_iter
    print('In average, total time: {} \n preprocess: {} \n net_forward: {} \n detect: {} \n output: {}'.format(
        total_time, preprocess_time, net_forward_time, detect_time, output_time
    ))
def demo_video(args, video_path):
    # 1. load the configure file
    cfg_from_file(args.confg_file)

    # 2. load detector based on the configure file
    object_detector = ObjectDetector()

    # 3. load video
    video = cv2.VideoCapture(video_path)
    index = -1
    frametime10 = np.zeros((10,1))
    while(video.isOpened()):
        #sys.stdout.write('Process image: {} \r'.format(index))
        #sys.stdout.flush()
        # 4. read image
        flag, image = video.read()
        t0 = timeit.default_timer()
        if flag == False:
            #print("Can not read image in Frame : {}".format(index))
            break

        # 5. detect
        _labels, _scores, _coords,elapsed = object_detector.predict(image)

        # 6. draw bounding box on the image
        for labels, scores, coords in zip(_labels, _scores, _coords):
            cv2.rectangle(image, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), COLORS[labels % 3], 5)
            cv2.putText(image, '{label}: {score:.3f}'.format(label=VOC_CLASSES[labels], score=scores), (int(coords[0]), int(coords[1])), FONT, 0.5, COLORS[labels % 3], 2)
        
        #print(elapsed)
        index = index + 1
        index = index%10
        #frametime10[index] = elapsed
        #fps = 1/np.average(frametime10)
        fps = 1/elapsed
        cv2.putText(image, '{fps:%.5s}'%(fps), (int(50), int(50)), FONT, 2,COLORS[1])
        #print(fps)
        # 7. visualize result
        if args.display is True:
            image = cv2.resize(image,(540,960))
            cv2.imshow('result', image)
            key = cv2.waitKey(1) & 0xFF
            if key==ord("q"):
                break
                video.close()
            if key==ord("s"):
                key2 = cv2.waitKey(0)


if __name__ == '__main__':
    #imagefile='/data/datasets/widerface-all/widerface-40-pixels/VOC2012/JPEGImages'
    imagefile = '/data/datasets/FDDB/VOC2007/JPEGImages'  
    imagelist=os.listdir(imagefile)
    args = parse_args()
    torch.set_num_threads(3)
    '''
    for image in imagelist:
        imagedir = os.path.join(imagefile,image)
        args.demo = imagedir
        demo(args, args.demo_file)
        print(args.demo)
    '''
    if args.type == 'image':
        demo_file(args, imagelist,imagefile)
    elif args.type == 'video':
        demo_video(args, args.demo_file)
    elif args.type == 'camera':
        demo_live(args, 0)
    elif args.type == 'time':
        time_benchmark(args, args.demo_file)
    else:
        AssertionError('type is not correct')
