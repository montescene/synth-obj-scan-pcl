from copy import deepcopy
import pickle
# from paths import *
import numpy as np
from os.path import join
import os
import open3d as o3d

from creatSynthUtils import export, getPointsWithin3DBB, getObj3DBB
from creatSynthUtils import nyu2palette
from creatSynthUtils import class_name_to_id as nyuName2Label
import random

SHAPENETCOREV2_DIR = '/media/shreyas/4aa82be1-14a8-47f7-93a7-171e3ebac2b0/Datasets/ShapeNetCore.v2'
TRAIN_SCANS_DIR = '/media/shreyas/ssd2/Dataset/scanNet/scans'
SYNTH_DATASET_DIR = '/media/shreyas/ssd2/Dataset/scan2cad_synth_forScenes'

from absl import flags
from absl import app
FLAGS = flags.FLAGS

flags.DEFINE_integer('start', 0, 'Start')
flags.DEFINE_integer('end', 1201, 'End')
flags.DEFINE_string('out_dir', '/media/shreyas/4aa82be1-14a8-47f7-93a7-171e3ebac2b0/Datasets/scannet_synth_testing', 'out dir')
flags.DEFINE_boolean('visualize', False, 'Visualize synth scene')
flags.DEFINE_boolean('dump_dataset', True, 'Dump the dataset to out_dir')


model2scanCoordinateChangeMatrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)

maxModelPerCat = {'chair': 8,
                  'table': 4,
                  'cabinet': 5,
                  'sofa': 5,
                  'bed': 4,
                  'bathtub': 2}

OBJ_CLASS_IDS = np.array([3,4,5,6,7,10,14,15,17,24,33,34,36])

NUM_DUPLICATES_PER_SCENE = 1   # number of duplicates per scene, creates a different combination of objects each time
MAX_NUM_INST_REPLICATIONS = 0  # always set to 0

sceneCounterDict = {}
def initSceneCounters(sceneList):
    for scene in sceneList:
        if scene[:-3] in sceneCounterDict:
            sceneCounterDict[scene[:-3]] += 1
        else:
            sceneCounterDict[scene[:-3]] = 1

def getOpen3dVis():
    # pcl and BB visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Open3D', width=640, height=480, left=0, top=0,
                      visible=True)  # use visible=True to visualize the point cloud
    vis.get_render_option().light_on = False

    return vis

def getScalePoseMat(pickData, SHAPENETCOREV2_DIR):
    '''
    Retrieve the scale and pose from the SOSPC dataset pickle file
    :param pickData:
    :param SHAPENETCOREV2_DIR:
    :return:
    '''
    if len(pickData['catID'].split('_'))>1:
        assert False
    objMesh = o3d.io.read_triangle_mesh(
        join(SHAPENETCOREV2_DIR, pickData['catID'], pickData['modelID'], 'models', 'model_normalized.obj'))
    objBBRest = getObj3DBB(np.asarray(objMesh.vertices))
    objCenter = np.mean(objBBRest, axis=0)

    scale = pickData['scale'][0]
    scale = scale[[0, 2, 1]]

    rotMat = pickData['poseMat'][0].T[:3, :3].dot(model2scanCoordinateChangeMatrix[:3, :3].T)
    trans = pickData['poseMat'][0].T[:3, 3] - rotMat.dot(np.diag(scale).dot(objCenter))

    poseMat = np.eye(4)
    poseMat[:3, :3] = rotMat
    poseMat[:3, 3] = trans

    return scale, poseMat, objMesh

def get3DBBForVotenet(obj_pc, label_id):
    xmin = np.min(obj_pc[:, 0])
    ymin = np.min(obj_pc[:, 1])
    zmin = np.min(obj_pc[:, 2])
    xmax = np.max(obj_pc[:, 0])
    ymax = np.max(obj_pc[:, 1])
    zmax = np.max(obj_pc[:, 2])
    bbox = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2,
                     xmax - xmin, ymax - ymin, zmax - zmin, label_id])

    return bbox

def dumpForVotenet(vert, labels, insts, bbox, sceneID, sceneCnt, SYNTH_SCENE_DATASET_DIR):
    assert sceneCnt > 0
    votenetDir = join(SYNTH_SCENE_DATASET_DIR, 'votenet')
    if not os.path.exists(votenetDir):
        os.mkdir(votenetDir)

    synthSceneName = sceneID[:-2] + '%02d'%(sceneCounterDict[sceneID[:-3]])

    vert = np.concatenate([vert, np.zeros_like(vert)], axis=1)

    N = vert.shape[0]
    MAX_NUM_POINT = 50000
    if N > MAX_NUM_POINT:
        choices = np.random.choice(N, MAX_NUM_POINT, replace=False)
        vert = vert[choices, :]
        labels = labels[choices]
        insts = insts[choices]

    np.save(join(votenetDir, synthSceneName + '_vert.npy'), vert)
    np.save(join(votenetDir, synthSceneName + '_sem_label.npy'), labels)
    np.save(join(votenetDir, synthSceneName + '_ins_label.npy'), insts)
    np.save(join(votenetDir, synthSceneName + '_bbox.npy'), bbox)


def dumpForMinkowski(pcl, labels, sceneID, sceneCnt, SYNTH_SCENE_DATASET_DIR):
    minkDir = join(SYNTH_SCENE_DATASET_DIR, 'minkowski')
    if not os.path.exists(minkDir):
        os.mkdir(minkDir)

    synthSceneName = sceneID[:-2] + '%02d' % (sceneCounterDict[sceneID[:-3]])

    scenePcl = o3d.geometry.PointCloud()
    scenePcl.points = o3d.utility.Vector3dVector(pcl)
    scenePcl.colors = o3d.utility.Vector3dVector(np.tile(np.expand_dims(labels,1), [1,3])/255.)

    o3d.io.write_point_cloud(join(minkDir, synthSceneName+'.ply'), scenePcl)

def dumpPclToCAD(pcl, semLabels, instLabels, annoDict, sceneID, sceneCnt, SYNTH_SCENE_DATASET_DIR):
    pcl2cadDir = join(SYNTH_SCENE_DATASET_DIR, 'PCL2CAD')
    if not os.path.exists(pcl2cadDir):
        os.mkdir(pcl2cadDir)

    synthSceneName = sceneID[:-2] + '%02d' % (sceneCounterDict[sceneID[:-3]])

    segPcl = o3d.geometry.PointCloud()
    segPcl.points = o3d.utility.Vector3dVector(pcl)
    segPcl.colors = o3d.utility.Vector3dVector(np.tile(np.expand_dims(semLabels, 1), [1, 3]) / 255.)

    instPcl = o3d.geometry.PointCloud()
    instPcl.points = o3d.utility.Vector3dVector(pcl)
    instPcl.colors = o3d.utility.Vector3dVector(np.tile(np.expand_dims(instLabels, 1), [1, 3]) / 255.)

    o3d.io.write_point_cloud(join(pcl2cadDir, synthSceneName + '_seg.ply'), segPcl)
    o3d.io.write_point_cloud(join(pcl2cadDir, synthSceneName + '_inst.ply'), instPcl)
    with open(join(pcl2cadDir, synthSceneName+'_cad.pickle'), 'wb') as f:
        pickle.dump(annoDict, f)


def findPlaceToAddPclInScene(origVerts, semLabels, newVertList, currVert, pickData, origSegPcl):
    bb3d = np.array([[np.min(currVert[:, 0]), np.min(currVert[:, 1]), np.min(currVert[:, 2])],
                     [np.min(currVert[:, 0]), np.min(currVert[:, 1]), np.max(currVert[:, 2])],
                     [np.min(currVert[:, 0]), np.max(currVert[:, 1]), np.min(currVert[:, 2])],
                     [np.min(currVert[:, 0]), np.max(currVert[:, 1]), np.max(currVert[:, 2])],
                     [np.max(currVert[:, 0]), np.min(currVert[:, 1]), np.min(currVert[:, 2])],
                     [np.max(currVert[:, 0]), np.min(currVert[:, 1]), np.max(currVert[:, 2])],
                     [np.max(currVert[:, 0]), np.max(currVert[:, 1]), np.min(currVert[:, 2])],
                     [np.max(currVert[:, 0]), np.max(currVert[:, 1]), np.max(currVert[:, 2])],
                     ]
                    )
    bb3d[::2,2] = 0
    bbCenterXY =  bb3d[0].copy()
    bbCenterXY[:2] = np.mean(bb3d[::2])
    xRange = np.arange(np.min(origVerts[:, 0])+0.5, np.max(origVerts[:, 0])-0.5, 0.2)
    yRange = np.arange(np.min(origVerts[:, 1])+0.5, np.max(origVerts[:, 1])-0.5, 0.2)
    np.random.shuffle(xRange)
    np.random.shuffle(yRange)
    tmpList = []
    for x in xRange:
        for y in yRange:
            newBB = bb3d - bbCenterXY + np.array([x,y,0])

            newBBForFloor = newBB.copy()
            newBBForFloor[::2,2] = -1.
            maskInsideBB, aa = getPointsWithin3DBB(np.array(origSegPcl.points), newBBForFloor)
            if np.sum(maskInsideBB) == 0:
                continue

            cols = np.round(np.array(origSegPcl.colors)[maskInsideBB]*255)
            floorMaskInsideBB =  cols[:,0]== 152
            numFloorPts = np.sum(floorMaskInsideBB)
            if numFloorPts < 300:
                continue

            # print('Found floor space')
            mask, pointsInsideBB = getPointsWithin3DBB(origVerts, newBB)
            floorMask = semLabels[mask] == nyuName2Label['floor']
            pointsInsideBB = pointsInsideBB[np.logical_not(floorMask)]
            if pointsInsideBB.shape[0] == 0:
                pointsInsideBB = np.zeros((1,3))*1.0
            for vert in newVertList:
                _, pointsInsideBB1 = getPointsWithin3DBB(vert, newBB)
                if pointsInsideBB1.shape[0]>0:
                    pointsInsideBB = np.concatenate([pointsInsideBB, pointsInsideBB1], axis=0)
            tmpList.append(pointsInsideBB.shape[0])
            if pointsInsideBB.shape[0] < 250:
                # print('Found object space')
                currVert = currVert - bbCenterXY + np.array([x,y,0])
                return currVert

    tmpList = np.array(tmpList)
    return None


def createSynthScene(sceneID, SYNTH_SCENE_DATASET_DIR, bookshelfModels, visualize, dump_dataset):
    if visualize:
        print('Visualizing Synthetic scene for ', sceneID)
    if dump_dataset:
        print('Creating Synthetic scene for ', sceneID)
    LABEL_MAP_FILE = 'scannetv2-labels.combined.tsv'
    mesh_file = os.path.join(TRAIN_SCANS_DIR, sceneID, sceneID + '_vh_clean_2.ply')
    agg_file = os.path.join(TRAIN_SCANS_DIR, sceneID, sceneID + '.aggregation.json')
    seg_file = os.path.join(TRAIN_SCANS_DIR, sceneID, sceneID + '_vh_clean_2.0.010000.segs.json')
    meta_file = os.path.join(TRAIN_SCANS_DIR, sceneID, sceneID + '.txt') # includes axisAlignment info for the train set scans.
    mesh_vertices, semantic_labels, instance_labels, instance_bboxes, instance2semantic = \
        export(mesh_file, agg_file, seg_file, meta_file, LABEL_MAP_FILE, None)

    bbox_mask = np.in1d(instance_bboxes[:, -1], OBJ_CLASS_IDS)
    instance_bboxes = instance_bboxes[bbox_mask, :]



    origSegPcl = o3d.io.read_point_cloud(os.path.join(TRAIN_SCANS_DIR, sceneID, sceneID + '_vh_clean_2.labels.ply'))
    origMesh = o3d.io.read_triangle_mesh(os.path.join(TRAIN_SCANS_DIR, sceneID, sceneID + '_vh_clean_2.ply'))
    # Load scene axis alignment matrix
    lines = open(meta_file).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) \
                for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
    origPts = np.array(origSegPcl.points)
    origVert = np.array(origMesh.vertices)
    pts = np.ones((origPts.shape[0], 4))
    verts = np.ones((origVert.shape[0], 4))
    pts[:,0:3] = origPts[:,0:3]
    verts[:,0:3] = origVert[:,0:3]
    pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
    verts = np.dot(verts, axis_align_matrix.transpose()) # Nx4
    origPts[:,0:3] = pts[:,0:3]
    origVert[:,0:3] = verts[:,0:3]
    xoffset = np.array([np.max(origPts[:,0]) - np.min(origPts[:,0]) + 1., 0., 0.])

    origSegPcl.points = o3d.utility.Vector3dVector(origPts)
    origMesh.vertices = o3d.utility.Vector3dVector(origVert)


    catList = ['chair', 'table', 'cabinet', 'bed', 'sofa', 'bathtub']

    for sceneCnt in range(NUM_DUPLICATES_PER_SCENE):
        if visualize:
            visOrig = getOpen3dVis()
            origSegPclForVis = deepcopy(origSegPcl)
            origSegPclForVis.points = o3d.utility.Vector3dVector(origPts+xoffset)
            visOrig.add_geometry(origSegPclForVis)
            visOrig.add_geometry(origMesh)

        synthObjDict = {}
        meshVert = mesh_vertices.copy()
        semLabels = semantic_labels.copy()
        instLabels = instance_labels.copy()
        removeMask = np.zeros((instLabels.shape[0],), dtype=np.bool)
        synthMeshVertList = []
        synthMeshLabelList = []
        synthMeshInstanceList = []
        bbListOrig = []
        instance_bboxes_new = instance_bboxes[:,[0,1,2,3,4,5,7]].copy()
        objMeshList = []
        if instance_bboxes.shape[0] == 0:
            return
        instanceIDCounter = np.max(instance_bboxes[:,6]) + 1
        random.shuffle(catList)
        annoDict = {}
        for cat in catList:
            synthCatDir = join(SYNTH_DATASET_DIR, cat)
            files = os.listdir(synthCatDir)
            files = [f[:-7] for f in files if 'pickle' in f and sceneID in f]
            files = sorted(files)
            instanceID = np.array([int(f.split('_')[2]) for f in files])
            sampleID = [f.split('_')[3] for f in files]
            if len(files) == 0:
                continue

            uniqueInstID = np.unique(instanceID)
            randomInds = []
            for id in uniqueInstID:
                currInstIDInd = np.arange(0, instanceID.shape[0])[instanceID == id]
                randomInds.append(np.random.choice(currInstIDInd[:min(currInstIDInd.shape[0],maxModelPerCat[cat])]))
                # randomInds.append(np.arange(0, instanceID.shape[0])[instanceID == id][0])

            synthObjDict[cat] = {'files': np.array(files)[randomInds],
                                 'instanceID':np.array(instanceID)[randomInds],
                                 'sampleID':np.array(sampleID)[randomInds]}


            if True:
                objInstVertCntList = []
                for i in range(uniqueInstID.shape[0]):
                    # remove
                    instIDCurr = synthObjDict[cat]['instanceID'][i]
                    removeMask = np.logical_or(removeMask, instLabels==instIDCurr)
                    objInstVertCntList.append(np.sum(instLabels==instIDCurr))

                    bbox6 = instance_bboxes[instance_bboxes[:,-2] == instIDCurr-1][0,:6]
                    bbox8x3 = np.zeros((8, 3), dtype=np.float32)
                    bbox8x3[0] = bbox6[:3] + np.array([-bbox6[3], -bbox6[4], -bbox6[5]]) / 2.
                    bbox8x3[1] = bbox6[:3] + np.array([-bbox6[3], -bbox6[4], bbox6[5]]) / 2.
                    bbox8x3[2] = bbox6[:3] + np.array([-bbox6[3], bbox6[4], -bbox6[5]]) / 2.
                    bbox8x3[3] = bbox6[:3] + np.array([-bbox6[3], bbox6[4], bbox6[5]]) / 2.
                    bbox8x3[4] = bbox6[:3] + np.array([bbox6[3], -bbox6[4], -bbox6[5]]) / 2.
                    bbox8x3[5] = bbox6[:3] + np.array([bbox6[3], -bbox6[4], bbox6[5]]) / 2.
                    bbox8x3[6] = bbox6[:3] + np.array([bbox6[3], bbox6[4], -bbox6[5]]) / 2.
                    bbox8x3[7] = bbox6[:3] + np.array([bbox6[3], bbox6[4], bbox6[5]]) / 2.
                    bbListOrig.append(bbox8x3)

            for i in range(uniqueInstID.shape[0]):
                #add
                instIDCurr = synthObjDict[cat]['instanceID'][i]

                fileCurr = synthObjDict[cat]['files'][i]
                with open(join(synthCatDir, fileCurr+'.pickle'), 'rb') as f:
                    pickData = pickle.load(f)
                samplePcl = o3d.io.read_point_cloud(join(synthCatDir, fileCurr + '.ply'))

                samplePts = np.array(samplePcl.points)

                synthMeshVertList.append(samplePts.copy()) #for synth scene pcl

                newLabel = nyuName2Label[cat]
                if cat == 'cabinet':
                    # differentiate between bookshelf and cabinet+regfrigerator
                    if pickData['modelID'] in bookshelfModels:
                        newLabel = nyuName2Label['bookshelf']
                synthMeshLabelList.append(np.ones((samplePts.shape[0]))*newLabel) # for synth scene segmentation

                synthMeshInstanceList.append(np.ones((samplePts.shape[0]))*instIDCurr) # for synth scene instance seg

                instance_bboxes_new[np.where(instance_bboxes[:,-2] == instIDCurr-1)[0][0]] = get3DBBForVotenet(samplePts, newLabel) # for synth scene Bbox

                numInstancesDict = {}
                numInstancesDict['chair'] = np.sum(instance_bboxes[:,-1]==nyuName2Label['chair'])
                numInstancesDict['sofa'] = np.sum(instance_bboxes[:, -1] == nyuName2Label['sofa'])
                numInstancesDict['bed'] = np.sum(instance_bboxes[:, -1] == nyuName2Label['bed'])
                numInstancesDict['table'] = np.sum(np.logical_or(instance_bboxes[:,-1]==nyuName2Label['table'], instance_bboxes[:,-1]==nyuName2Label['desk']))
                numInstancesDict['cabinet'] = 0
                numInstancesDict['bathtub'] = 0
                numInstances = numInstancesDict[cat]
                if numInstances> 5:
                    numReplications = 1
                elif numInstances > 3:
                    numReplications = 2
                else:
                    numReplications = MAX_NUM_INST_REPLICATIONS

                for instReplCnt in range(min(numReplications,MAX_NUM_INST_REPLICATIONS)):
                    if cat in ['chair', 'table', 'sofa', 'bed']:
                        currInstIDInd = np.arange(0, instanceID.shape[0])[instanceID == instIDCurr]
                        newRandomInstInd = np.random.choice(currInstIDInd)
                        newFile = np.array(files)[newRandomInstInd]
                        with open(join(synthCatDir, newFile + '.pickle'), 'rb') as f:
                            pickData = pickle.load(f)
                        samplePcl = o3d.io.read_point_cloud(join(synthCatDir, fileCurr + '.ply'))
                        samplePts = np.array(samplePcl.points)
                        samplePtsRandomInds = np.random.choice(np.arange(0,samplePts.shape[0]), min(objInstVertCntList[i],samplePts.shape[0]), replace=False)
                        samplePts = samplePts[samplePtsRandomInds]
                        samplePts = findPlaceToAddPclInScene(meshVert[:, :3], semLabels, synthMeshVertList, samplePts, pickData, origSegPcl)

                        if samplePts is None:
                            continue

                        synthMeshVertList.append(samplePts.copy())  # for synth scene pcl
                        synthMeshLabelList.append(
                            np.ones((samplePts.shape[0])) * newLabel)  # for synth scene segmentation
                        synthMeshInstanceList.append(
                            np.ones((samplePts.shape[0])) * (instanceIDCounter))  # for synth scene instance seg
                        instance_bboxes_new = np.concatenate([instance_bboxes_new,
                                                              np.expand_dims(get3DBBForVotenet(samplePts, newLabel),0)], axis=0)  # for synth scene Bbox

                        instanceIDCounter = instanceIDCounter + 1



                assert instIDCurr not in annoDict.keys()
                annoDict[instIDCurr] = {'catID':pickData['catID'],
                                        'modelID': pickData['modelID'],
                                        'scale': pickData['scale'].copy(),
                                        'poseMat': pickData['poseMat'].copy()}
                if len(annoDict[instIDCurr]['catID'].split('_'))>1:
                    catIDsOrig = annoDict[instIDCurr]['catID'].split('_')
                    if 'toilet' in catIDsOrig:
                        catIDsOrig.append('03001627')
                    found = False
                    for kk in range(len(catIDsOrig)):
                        if os.path.exists(join(SHAPENETCOREV2_DIR, catIDsOrig[kk], annoDict[instIDCurr]['modelID'], 'models', 'model_normalized.obj')):
                            found = True
                            break

                    if not found:
                        print(catIDsOrig, annoDict[instIDCurr]['modelID'])
                        assert False
                    annoDict[instIDCurr]['catID'] = catIDsOrig[kk]

                # get the scale and pose matrix from the pickle file
                annoDict[instIDCurr]['scale'], annoDict[instIDCurr]['poseMat'],\
                objMesh = getScalePoseMat(annoDict[instIDCurr], SHAPENETCOREV2_DIR)
                scale = annoDict[instIDCurr]['scale']
                poseMat = annoDict[instIDCurr]['poseMat']

                # for visualization
                if visualize:
                    objVert = np.asarray(objMesh.vertices).dot(np.diag(scale)).dot(poseMat[:3, :3].T) + poseMat[:3,
                                                                                                        3]

                    objMesh.vertices = o3d.utility.Vector3dVector(objVert + xoffset * 3)
                    objMesh.vertex_colors = o3d.utility.Vector3dVector(
                        np.reshape(np.random.uniform(0., 1., objVert.shape[0] * 3), (objVert.shape[0], 3)))
                    visOrig.add_geometry(objMesh)






        meshVert = meshVert[np.logical_not(removeMask)]
        semLabels = semLabels[np.logical_not(removeMask)]
        instLabels = instLabels[np.logical_not(removeMask)]

        meshVert = np.concatenate([meshVert[:, :3]] + synthMeshVertList, axis=0)
        semLabels = np.concatenate([semLabels] + synthMeshLabelList, axis=0)
        instLabels = np.concatenate([instLabels] + synthMeshInstanceList, axis=0)

        if dump_dataset:
            ## uncomment below to lines to dump the dataset in the format required to train Votenet and Minkowski net
            # dumpForVotenet(meshVert, semLabels, instLabels, instance_bboxes_new, sceneID, sceneCnt+1, SYNTH_SCENE_DATASET_DIR)
            # dumpForMinkowski(meshVert, semLabels, sceneID, sceneCnt+1, SYNTH_SCENE_DATASET_DIR)

            dumpPclToCAD(meshVert, semLabels, instLabels, annoDict, sceneID, sceneCnt+1, SYNTH_SCENE_DATASET_DIR)
            sceneCounterDict[sceneID[:-3]] += 1

        # visualize
        if visualize:
            semLabelsVis = nyu2palette(np.tile(np.expand_dims(semLabels,1), [1,3]))
            newPcl = o3d.geometry.PointCloud()
            newPcl.points = o3d.utility.Vector3dVector(meshVert+ 2*xoffset)
            newPcl.colors = o3d.utility.Vector3dVector(semLabelsVis/255)
            visOrig.add_geometry(newPcl)

            # draw the BBs
            for k in range(len(bbListOrig)):
                lines = [[0, 1], [0, 2], [1, 3], [2, 3],
                         [4, 5], [4, 6], [5, 7], [6, 7],
                         [0, 4], [1, 5], [2, 6], [3, 7]
                         ]
                colors = [[1, 0, 0] for i in range(len(lines))]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(bbListOrig[k])
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors)
                visOrig.add_geometry(line_set)

            # draw the BBs
            for k in range(instance_bboxes_new.shape[0]):
                bbox6 = instance_bboxes_new[k]
                bbox8x3 = np.zeros((8, 3), dtype=np.float32)
                bbox8x3[0] = bbox6[:3] + np.array([-bbox6[3], -bbox6[4], -bbox6[5]]) / 2.
                bbox8x3[1] = bbox6[:3] + np.array([-bbox6[3], -bbox6[4], bbox6[5]]) / 2.
                bbox8x3[2] = bbox6[:3] + np.array([-bbox6[3], bbox6[4], -bbox6[5]]) / 2.
                bbox8x3[3] = bbox6[:3] + np.array([-bbox6[3], bbox6[4], bbox6[5]]) / 2.
                bbox8x3[4] = bbox6[:3] + np.array([bbox6[3], -bbox6[4], -bbox6[5]]) / 2.
                bbox8x3[5] = bbox6[:3] + np.array([bbox6[3], -bbox6[4], bbox6[5]]) / 2.
                bbox8x3[6] = bbox6[:3] + np.array([bbox6[3], bbox6[4], -bbox6[5]]) / 2.
                bbox8x3[7] = bbox6[:3] + np.array([bbox6[3], bbox6[4], bbox6[5]]) / 2.

                lines = [[0, 1], [0, 2], [1, 3], [2, 3],
                         [4, 5], [4, 6], [5, 7], [6, 7],
                         [0, 4], [1, 5], [2, 6], [3, 7]
                         ]
                colors = [[1, 0, 0] for i in range(len(lines))]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(bbox8x3+2*xoffset)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors)
                visOrig.add_geometry(line_set)


            visOrig.run()
            visOrig.close()
            visOrig.destroy_window()
            del visOrig


def main(argv):
    SYNTH_SCENE_DATASET_DIR = FLAGS.out_dir

    if not os.path.exists(SYNTH_SCENE_DATASET_DIR) and FLAGS.dump_dataset:
        os.mkdir(SYNTH_SCENE_DATASET_DIR)

    with open('scannetv2_%s.txt' % ('train'), 'r') as f:
        lines = f.readlines()
    sceneList = [l.strip() for l in lines]
    # sceneList = ['scene0519_00']
    initSceneCounters(sceneList)
    bookshelfModels = os.listdir(join(SHAPENETCOREV2_DIR, '02871439'))



    for scene in sceneList[FLAGS.start:FLAGS.end:1]:
        createSynthScene(scene, SYNTH_SCENE_DATASET_DIR, bookshelfModels, FLAGS.visualize, FLAGS.dump_dataset)


if __name__ == '__main__':
    app.run(main)


