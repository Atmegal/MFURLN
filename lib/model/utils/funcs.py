import pdb
import numpy as np

def generate_phrase_box(sbox, obox):

    phrase = np.zeros([4])
    phrase[0] = min(sbox[0], obox[0])
    phrase[1] = min(sbox[1], obox[1])
    phrase[2] = max(sbox[2], obox[2])
    phrase[3] = max(sbox[3], obox[3])
    return phrase

def compute_iou_each(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    if xB < xA or yB < yA:
        IoU = 0
    else:
        area_I = (xB - xA + 1.0) * (yB - yA + 1.0)
        area1 = (box1[2] - box1[0] + 1.0) * (box1[3] - box1[1] + 1.0)
        area2 = (box2[2] - box2[0] + 1.0) * (box2[3] - box2[1] + 1.0)
        IoU = area_I / float(area1 + area2 - area_I)
    return IoU

def all_recall(test_roidb, pred_roidb, N_recall, class_num):

    N_right_r = 0.0
    N_right_p = 0.0
    N_total = 0.0
    N_data = len(test_roidb)

    for i in range(N_data):
        rela_gt = test_roidb[i]['rela_gt']
        if len(rela_gt) == 0:
            continue
        sub_gt = test_roidb[i]['sub_gt']
        obj_gt = test_roidb[i]['obj_gt']
        sub_box_gt = test_roidb[i]['sub_box_gt']
        obj_box_gt = test_roidb[i]['obj_box_gt']

        pred_rela = pred_roidb[i]['pred_rela']
        pred_rela_score = pred_roidb[i]['pred_rela_score']
        sub_dete = pred_roidb[i]['sub_dete']
        obj_dete = pred_roidb[i]['obj_dete']
        sub_box_dete = pred_roidb[i]['sub_box_dete']
        obj_box_dete = pred_roidb[i]['obj_box_dete']

        N_rela = len(rela_gt)
        N_total = N_total + N_rela

        sort_id = np.argsort(-np.reshape(pred_rela_score, [1, -1]))
        detected_gt_p = np.zeros([N_rela, ])
        detected_gt_r = np.zeros([N_rela, ])
        N_recall = np.minimum(N_recall, np.shape(pred_rela_score)[0])

        for f in range(N_recall):
            j = sort_id[0][f]
            positionk_r = -1
            maxk_r = 0
            positionk_p = -1
            maxk_p = 0
            for k in range(N_rela):
                if detected_gt_r[k] > 0 and detected_gt_p[k] > 0 :
                    continue
                if (sub_gt[k] == sub_dete[j]) and (obj_gt[k] == obj_dete[j]) and (rela_gt[k] == pred_rela[j]):
                    if detected_gt_r[k] == 0:
                        s_iou = compute_iou_each(sub_box_dete[j], sub_box_gt[k])
                        o_iou = compute_iou_each(obj_box_dete[j], obj_box_gt[k])
                        iou = np.max([s_iou, o_iou])
                        if (s_iou >= 0.5) and (o_iou >= 0.5) and iou > maxk_r:
                            maxk_r = iou
                            positionk_r = k
                    if detected_gt_p[k] == 0:
                        phrase_gt = generate_phrase_box(sub_box_gt[k], obj_box_gt[k])
                        phrase_dete = generate_phrase_box(sub_box_dete[j], obj_box_dete[j])
                        p_iou = compute_iou_each(phrase_dete, phrase_gt)
                        if (p_iou >= 0.5) and p_iou > maxk_p:
                            maxk_p = p_iou
                            positionk_p = k

            if positionk_r > -1:
                detected_gt_r[positionk_r] = 1
            if positionk_p > -1:
                detected_gt_p[positionk_p] = 1
        # r relation  p phrase
        N_right_r = N_right_r + np.sum(detected_gt_r)
        N_right_p = N_right_p + np.sum(detected_gt_p)

    return N_right_r, N_right_p, N_total