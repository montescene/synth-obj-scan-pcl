import numpy as np
import os
import csv
from plyfile import PlyData, PlyElement
import json

# color palette for nyu40 labels
def create_color_palette():
    return [
       (0, 0, 0),
       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair
       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf
       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),
       (247, 182, 210),		# desk
       (66, 188, 102),
       (219, 219, 141),		# curtain
       (140, 57, 197),
       (202, 185, 52),
       (51, 176, 203),
       (200, 54, 131),
       (92, 193, 61),
       (78, 71, 183),
       (172, 114, 82),
       (255, 127, 14), 		# refrigerator
       (91, 163, 138),
       (153, 98, 156),
       (140, 153, 101),
       (158, 218, 229),		# shower curtain
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209),
       (227, 119, 194),		# bathtub
       (213, 92, 176),
       (94, 106, 211),
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)
    ]

def nyu2palette(arr):
    isNormalized = False
    if np.max(arr) <= 1.:
        isNormalized = True
        arr = np.round(arr * 255).astype(np.uint8)
    palette = create_color_palette()
    nyu2palette = {}
    palette2nyu = {}
    for i in range(len(palette)):
        nyu2palette[i] = palette[i]
        palette2nyu[palette[i]] = i

    for i in range(arr.shape[0]):
        arr[i] = np.array(nyu2palette[arr[i,0]])
        if isNormalized:
            arr[i] = arr[i]/255.

    return arr

def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + 1 # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs


def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


def export(mesh_file, agg_file, seg_file, meta_file, label_map_file, output_file=None):
    """ points are XYZ RGB (RGB in 0-255),
    semantic label as nyu40 ids,
    instance label as 1-#instance,
    box as (cx,cy,cz,dx,dy,dz,semantic_label)
    """
    label_map = read_label_mapping(label_map_file,
        label_from='raw_category', label_to='nyu40id')    
    mesh_vertices = read_mesh_vertices_rgb(mesh_file)

    # Load scene axis alignment matrix
    lines = open(meta_file).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) \
                for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:,0:3] = mesh_vertices[:,0:3]
    pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
    mesh_vertices[:,0:3] = pts[:,0:3]

    # Load semantic and instance labels
    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)
    label_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
    object_id_to_label_id = {}
    for label, segs in label_to_segs.items():
        label_id = label_map[label]
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = label_id
    instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
    num_instances = len(np.unique(list(object_id_to_segs.keys())))
    for object_id, segs in object_id_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            instance_ids[verts] = object_id
            if object_id not in object_id_to_label_id:
                object_id_to_label_id[object_id] = label_ids[verts][0]
    instance_bboxes = np.zeros((num_instances,8))
    for obj_id in object_id_to_segs:
        label_id = object_id_to_label_id[obj_id]
        obj_pc = mesh_vertices[instance_ids==obj_id, 0:3]
        if len(obj_pc) == 0: continue
        # Compute axis aligned box
        # An axis aligned bounding box is parameterized by
        # (cx,cy,cz) and (dx,dy,dz) and label id
        # where (cx,cy,cz) is the center point of the box,
        # dx is the x-axis length of the box.
        xmin = np.min(obj_pc[:,0])
        ymin = np.min(obj_pc[:,1])
        zmin = np.min(obj_pc[:,2])
        xmax = np.max(obj_pc[:,0])
        ymax = np.max(obj_pc[:,1])
        zmax = np.max(obj_pc[:,2])
        bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2,
            xmax-xmin, ymax-ymin, zmax-zmin, obj_id-1, label_id])
        # NOTE: this assumes obj_id is in 1,2,3,.,,,.NUM_INSTANCES
        instance_bboxes[obj_id-1,:] = bbox 

    if output_file is not None:
        np.save(output_file+'_vert.npy', mesh_vertices)
        np.save(output_file+'_sem_label.npy', label_ids)
        np.save(output_file+'_ins_label.npy', instance_ids)
        np.save(output_file+'_bbox.npy', instance_bboxes)

    return mesh_vertices, label_ids, instance_ids,\
        instance_bboxes, object_id_to_label_id

def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k):v for k,v in mapping.items()}
    return mapping

def read_mesh_vertices(filename):
    """ read XYZ for each vertex.
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
    return vertices

def read_mesh_vertices_rgb(filename):
    """ read XYZ RGB for each vertex.
    Note: RGB values are in 0-255
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['green']
        vertices[:,5] = plydata['vertex'].data['blue']
    return vertices

def represents_int(s):
    ''' if string s represents an int. '''
    try: 
        int(s)
        return True
    except ValueError:
        return False
    
def getPointsWithin3DBB(pcl, bb, th=0.001):
    assert bb.shape == (8,3)
    d1 = bb[1] - bb[0]
    d2 = bb[2] - bb[0]
    d3 = bb[4] - bb[0]

    d = np.stack([d1, d2, d3], axis=0)

    t1 = np.matmul(d, pcl.T)

    c = np.matmul(d, bb[0:1].T)
    c11 = d1.dot(bb[0])
    c21 = d2.dot(bb[0])
    c31 = d3.dot(bb[0])

    c12 = d1.dot(bb[1])
    c22 = d2.dot(bb[2])
    c32 = d3.dot(bb[4])

    tt1 = np.logical_and(t1[0] < c12+th, t1[0] > c11-th)
    tt2 = np.logical_and(t1[1] < c22+th, t1[1] > c21-th)
    tt3 = np.logical_and(t1[2] < c32+th, t1[2] > c31-th)

    mask = np.squeeze(np.logical_and(np.logical_and(tt1, tt2), tt3))

    a = np.sum(mask.astype(np.uint32))

    pclMasked = pcl[mask]

    return mask, pclMasked

def getObj3DBB(verts):
    '''
    The order of the corners returned are:
    [Front-Left-Bottom,
     Front-Left-Top,
     Front-Right-Bottom,
     Front-Right-Top,
     Back-Left-Bottom,
     Back-Left-Top,
     Back-Right-Bottom,
     Back-Right-Top,
    ]
    :param verts:
    :return:
    '''
    assert len(verts.shape) == 2
    assert verts.shape[1] <= 4

    bb3d = np.array([[np.max(verts[:, 0]), np.min(verts[:, 1]), np.max(verts[:, 2])],
                     [np.max(verts[:, 0]), np.max(verts[:, 1]), np.max(verts[:, 2])],
                     [np.max(verts[:, 0]), np.min(verts[:, 1]), np.min(verts[:, 2])],
                     [np.max(verts[:, 0]), np.max(verts[:, 1]), np.min(verts[:, 2])],
                     [np.min(verts[:, 0]), np.min(verts[:, 1]), np.max(verts[:, 2])],
                     [np.min(verts[:, 0]), np.max(verts[:, 1]), np.max(verts[:, 2])],
                     [np.min(verts[:, 0]), np.min(verts[:, 1]), np.min(verts[:, 2])],
                     [np.min(verts[:, 0]), np.max(verts[:, 1]), np.min(verts[:, 2])],
                     ]
                    )

    bb3d = np.array([[np.min(verts[:, 0]), np.min(verts[:, 1]), np.max(verts[:, 2])],
                     [np.min(verts[:, 0]), np.max(verts[:, 1]), np.max(verts[:, 2])],
                     [np.max(verts[:, 0]), np.min(verts[:, 1]), np.max(verts[:, 2])],
                     [np.max(verts[:, 0]), np.max(verts[:, 1]), np.max(verts[:, 2])],
                     [np.min(verts[:, 0]), np.min(verts[:, 1]), np.min(verts[:, 2])],
                     [np.min(verts[:, 0]), np.max(verts[:, 1]), np.min(verts[:, 2])],
                     [np.max(verts[:, 0]), np.min(verts[:, 1]), np.min(verts[:, 2])],
                     [np.max(verts[:, 0]), np.max(verts[:, 1]), np.min(verts[:, 2])],
                     ]
                    )

    return bb3d
    

SCANNET_COLORMAP = [
    [0., 0., 0.],
    [174., 199., 232.],
    [152., 223., 138.],
    [31., 119., 180.],
    [255., 187., 120.],
    [188., 189., 34.],
    [140., 86., 75.],
    [255., 152., 150.],
    [214., 39., 40.],
    [197., 176., 213.],
    [148., 103., 189.],
    [196., 156., 148.],
    [23., 190., 207.],
    [247., 182., 210.],
    [66., 188., 102.],
    [219., 219., 141.],
    [140., 57., 197.],
    [202., 185., 52.],
    [51., 176., 203.],
    [200., 54., 131.],
    [92., 193., 61.],
    [78., 71., 183.],
    [172., 114., 82.],
    [255., 127., 14.],
    [91., 163., 138.],
    [153., 98., 156.],
    [140., 153., 101.],
    [158., 218., 229.],
    [100., 125., 154.],
    [178., 127., 135.],
    [146., 111., 194.],
    [44., 160., 44.],
    [112., 128., 144.],
    [96., 207., 209.],
    [227., 119., 194.],
    [213., 92., 176.],
    [94., 106., 211.],
    [82., 84., 163.],
    [100., 85., 144.]]

SCANNET_COLORMAP = np.asarray(SCANNET_COLORMAP) / 255.

class_names = []
class_name_to_id = {}
for i, line in enumerate(open("label_names.txt").readlines()):
    class_id = i  # starts with -1
    class_name = line.strip()
    class_name_to_id[class_name] = class_id
    class_names.append(class_name)
class_names = tuple(class_names)