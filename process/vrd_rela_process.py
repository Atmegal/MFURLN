import numpy as np
import json
import cv2
import pdb

train_file_path = '/home/hp/dataset/VRD/json_dataset/annotations_train.json'
test_file_path = '/home/hp/dataset/VRD/json_dataset/annotations_test.json'
file_path = [train_file_path, test_file_path]
train_image_path = '/home/hp/dataset/VRD/sg_dataset/sg_train_images/'
test_image_path = '/home/hp/dataset/VRD/sg_dataset/sg_test_images/'
image_path = [train_image_path, test_image_path]


for r in range(2):
	file_path_use = file_path[r]
	image_path_use = image_path[r]
	roidb = []
	with open(file_path_use,'r') as f:
		data=json.load(f)
		image_name_key = data.keys()
		len_img = len(image_name_key)
		t = 0
		for image_name in image_name_key:
			roidb_temp = {}
			print(image_name)
			image_full_path = image_path_use + image_name
			im = cv2.imread(image_full_path)
			if type(im) == type(None):
				continue

			im_shape = np.shape(im)
			im_h = im_shape[0]
			im_w = im_shape[1]

			roidb_temp['image'] = image_full_path
			roidb_temp['width'] = im_w
			roidb_temp['height'] = im_h

			d = data[image_name]
			relation_length = len(d)
			if relation_length == 0:
				continue
			sb_new = np.zeros(shape=[relation_length,4])
			ob_new = np.zeros(shape=[relation_length,4])
			rela = np.zeros(shape=[relation_length,])
			obj = np.zeros(shape=[relation_length,])
			subj = np.zeros(shape=[relation_length,])
			t = t+1
			for relation_id in range(relation_length):
				relation = d[relation_id]
				obj[relation_id] = relation['object']['category'] +1
				subj[relation_id] = relation['subject']['category'] +1
				rela[relation_id] = relation['predicate']

				ob_temp = relation['object']['bbox']
				sb_temp = relation['subject']['bbox']
				ob = [ob_temp[0],ob_temp[1],ob_temp[2],ob_temp[3]]
				sb = [sb_temp[0],sb_temp[1],sb_temp[2],sb_temp[3]]
				
				ob_new[relation_id][0:4] = [ob[2],ob[0],ob[3],ob[1]]
				sb_new[relation_id][0:4] = [sb[2],sb[0],sb[3],sb[1]]

			sub_box = sb_new
			obj_box = ob_new
			sub_box_1 = np.concatenate( (sub_box, np.reshape(subj,[-1,1])), axis=1 )
			obj_box_1 = np.concatenate( (obj_box, np.reshape(obj,[-1,1])), axis=1 )

			boxes = np.concatenate( (sub_box_1, obj_box_1), axis=0 )
			unique_boxes, unique_inds = np.unique(boxes, axis=0, return_index = True)
			#pdb.set_trace()
			roidb_temp['boxes'] = unique_boxes[:,:4]
			roidb_temp['gt_classes'] = unique_boxes[:,4]
			roidb_temp['max_overlaps'] = np.ones(np.shape(roidb_temp['gt_classes']))
			roidb_temp['flipped'] = False
			roidb_temp['id'] = t
			roidb_temp['sub_boxes1'] = sub_box
			roidb_temp['obj_boxes1'] = obj_box
			roidb_temp['sub_gt1'] = subj
			roidb_temp['obj_gt1'] = obj
			roidb_temp['rela_gt1'] = rela

			boxes_1 = np.concatenate( (sub_box_1, obj_box_1), axis=1 )
			unique_boxes_1, unique_inds_1 = np.unique(boxes_1, axis=0, return_index = True)

			roidb_temp['sub_boxes'] = sub_box[unique_inds_1]
			roidb_temp['obj_boxes'] = obj_box[unique_inds_1]
			roidb_temp['sub_gt'] = subj[unique_inds_1]
			roidb_temp['obj_gt'] = obj[unique_inds_1]
			roidb_temp['rela_gt'] = np.zeros([len(roidb_temp['sub_boxes']), 71])

			for i in range(len(roidb_temp['rela_gt'])):
				for j in range(len(sub_box)):
					if np.sum(np.abs(roidb_temp['sub_gt'][i]-subj[j]) +
							  np.abs(roidb_temp['obj_gt'][i]-obj[j]) +
							  np.abs(roidb_temp['sub_boxes'][i]-sub_box[j]) +
							  np.abs(roidb_temp['obj_boxes'][i]-obj_box[j]) ) == 0:
						roidb_temp['rela_gt'][i,  np.int(rela[j]) + 1] = 1
			roidb.append(roidb_temp)

	if r == 0:
		val_roidb = roidb[3500:]		
		print(len(val_roidb))
		roidb = roidb[:3500]	
		N_roidb = len(roidb)
		print(len(roidb))
		for image_id in range(N_roidb):
			roidb_temp = {}
			roidb_temp['image'] = roidb[image_id]['image']
			roidb_temp['width'] = roidb[image_id]['width']
			roidb_temp['height'] = roidb[image_id]['height']
			roidb_temp['gt_classes'] = roidb[image_id]['gt_classes']
			roidb_temp['max_overlaps'] = roidb[image_id]['max_overlaps']
			roidb_temp['flipped'] = True
			roidb_temp['id'] = roidb[image_id]['id']
			roidb_temp['rela_gt'] = roidb[image_id]['rela_gt']
			roidb_temp['sub_gt'] = roidb[image_id]['sub_gt']
			roidb_temp['obj_gt'] = roidb[image_id]['obj_gt']

			boxes_old = roidb[image_id]['boxes']
			width = roidb[image_id]['width']
			boxes_new = np.zeros(np.shape(boxes_old))
			boxes_new[:,0] = width - boxes_old[:,2] - 1
			boxes_new[:,1] = boxes_old[:,1]
			boxes_new[:,2] = width - boxes_old[:,0] - 1
			boxes_new[:,3] = boxes_old[:,3]
			roidb_temp['boxes'] = boxes_new

			boxes_old = roidb[image_id]['sub_boxes']
			boxes_new = np.zeros(np.shape(boxes_old))
			boxes_new[:,0] = width - boxes_old[:,2] - 1
			boxes_new[:,1] = boxes_old[:,1]
			boxes_new[:,2] = width - boxes_old[:,0] - 1
			boxes_new[:,3] = boxes_old[:,3]
			roidb_temp['sub_boxes'] = boxes_new

			boxes_old = roidb[image_id]['obj_boxes']
			boxes_new = np.zeros(np.shape(boxes_old))
			boxes_new[:,0] = width - boxes_old[:,2] - 1
			boxes_new[:,1] = boxes_old[:,1]
			boxes_new[:,2] = width - boxes_old[:,0] - 1
			boxes_new[:,3] = boxes_old[:,3]
			roidb_temp['obj_boxes'] = boxes_new

			roidb.append(roidb_temp)
		train_roidb = roidb
		print(len(train_roidb))
	elif r == 1:
		test_roidb = roidb

roidb = {}
roidb['train_roidb'] = train_roidb
roidb['val_roidb'] = val_roidb
roidb['test_roidb'] = test_roidb

np.savez('vrd_rela_roidb.npz',roidb=roidb)

