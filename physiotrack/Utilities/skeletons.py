#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
###########################################################################
## SKELETONS DEFINITIONS                                                 ##
###########################################################################

The definition and hierarchy of the following skeletons are available: 
- RTMPose HALPE_26, COCO_133, COCO_133_WRIST, COCO_17, HAND, FACE, ANIMAL
- OpenPose BODY_25B, BODY_25, BODY_135, COCO, MPII
- Mediapipe BLAZEPOSE
- AlphaPose HALPE_26, HALPE_68, HALPE_136, COCO_133, COCO, MPII 
(for COCO and MPII, AlphaPose must be run with the flag "--format cmu")
- DeepLabCut CUSTOM: the skeleton will be defined in Config.toml

N.B.: Not all face and hand keypoints are reported in the skeleton architecture, 
since some are redundant for the orientation of some bodies.

Check the skeleton structure with:
from anytree import Node, RenderTree
for pre, _, node in RenderTree(model): 
    print(f'{pre}{node.name} id={node.id}')
'''

## INIT
from anytree import Node


## AUTHORSHIP INFORMATION
__author__ = "PhysioTrack Team"
__copyright__ = "Copyright 2023, PhysioTrack"
__license__ = "BSD 3-Clause License"
__version__ = "0.1.0"


'''HALPE_26 (full-body without hands, from AlphaPose, MMPose, etc.)
https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md
https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose'''
HALPE_26 = Node("Hip", id=19, children=[
    Node("RHip", id=12, children=[
        Node("RKnee", id=14, children=[
            Node("RAnkle", id=16, children=[
                Node("RBigToe", id=21, children=[
                    Node("RSmallToe", id=23),
                ]),
                Node("RHeel", id=25),
            ]),
        ]),
    ]),
    Node("LHip", id=11, children=[
        Node("LKnee", id=13, children=[
            Node("LAnkle", id=15, children=[
                Node("LBigToe", id=20, children=[
                    Node("LSmallToe", id=22),
                ]),
                Node("LHeel", id=24),
            ]),
        ]),
    ]),
    Node("Neck", id=18, children=[
        Node("Head", id=17, children=[
            Node("Nose", id=0),
        ]),
        Node("RShoulder", id=6, children=[
            Node("RElbow", id=8, children=[
                Node("RWrist", id=10),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=7, children=[
                Node("LWrist", id=9),
            ]),
        ]),
    ]),
])


'''COCO_133_WRIST (full-body with hands and face, from AlphaPose, MMPose, etc.)
https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md
https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose'''
COCO_133_WRIST = Node("Hip", id=None, children=[
    Node("RHip", id=12, children=[
        Node("RKnee", id=14, children=[
            Node("RAnkle", id=16, children=[
                Node("RBigToe", id=20, children=[
                    Node("RSmallToe", id=21),
                ]),
                Node("RHeel", id=22),
            ]),
        ]),
    ]),
    Node("LHip", id=11, children=[
        Node("LKnee", id=13, children=[
            Node("LAnkle", id=15, children=[
                Node("LBigToe", id=17, children=[
                    Node("LSmallToe", id=18),
                ]),
                Node("LHeel", id=19),
            ]),
        ]),
    ]),
    Node("Neck", id=None, children=[
        Node("Nose", id=0, children=[
            Node("REye", id=2),
            Node("LEye", id=1),
        ]),
        Node("RShoulder", id=6, children=[
            Node("RElbow", id=8, children=[
                Node("RWrist", id=10, children=[
                    Node("RThumb", id=114),
                    Node("RIndex", id=117),
                    Node("RPinky", id=129),
                ]),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=7, children=[
                Node("LWrist", id=9, children=[
                    Node("LThumb", id=93),
                    Node("LIndex", id=96),
                    Node("LPinky", id=108),
                ])
            ]),
        ]),
    ]),
])


'''COCO_17 (full-body without hands and feet, from OpenPose, AlphaPose, OpenPifPaf, YOLO-pose, MMPose, etc.)
https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose'''
COCO_17 = Node("Hip", id=None, children=[
    Node("RHip", id=12, children=[
        Node("RKnee", id=14, children=[
            Node("RAnkle", id=16),
        ]),
    ]),
    Node("LHip", id=11, children=[
        Node("LKnee", id=13, children=[
            Node("LAnkle", id=15),
        ]),
    ]),
    Node("Neck", id=None, children=[
        Node("Nose", id=0),
        Node("RShoulder", id=6, children=[
            Node("RElbow", id=8, children=[
                Node("RWrist", id=10),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=7, children=[
                Node("LWrist", id=9),
            ]),
        ]),
    ]),
])

# More skeletons definitions follow the same pattern but are omitted for brevity