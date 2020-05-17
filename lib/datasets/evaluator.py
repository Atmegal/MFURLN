import numpy as np

def compute_iou(box, proposal):
    len_proposal = np.shape(proposal)[0]
    IoU = np.empty([len_proposal, 1])
    for i in range(len_proposal):
        xA = max(box[0], proposal[i, 0])
        yA = max(box[1], proposal[i, 1])
        xB = min(box[2], proposal[i, 2])
        yB = min(box[3], proposal[i, 3])

        if xB < xA or yB < yA:
            IoU[i, 0] = 0
        else:
            area_I = (xB - xA + 1.0) * (yB - yA + 1.0)
            area1 = (box[2] - box[0] + 1.0) * (box[3] - box[1] + 1.0)
            area2 = (proposal[i, 2] - proposal[i, 0] + 1.0) * (proposal[i, 3] - proposal[i, 1] + 1.0)
            IoU[i, 0] = area_I / float(area1 + area2 - area_I)
    return IoU


def voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluation(test, pred, class_num, iou_thresh=0.5, use_07_metric=False):
    gt_total = 0
    sorted_pred = sorted(pred, key=lambda x: (-x['scores']))
    det = []
    for i in range(len(test)):
        det_pd = np.zeros(len(test[i]))
        det.append(det_pd)
        if test[i] != []:
            gt_total = gt_total + len(test[i])
    # total is the number of pred objects for per image
    # gt_total is the number of gt objects for per image
    total = len(sorted_pred)
    if total == 0 or gt_total == 0:
        return 0.0

    else:
        tp = np.zeros(total)
        fp = np.zeros(total)
        for i in range(total):
            img_id = sorted_pred[i]['image_id']
            bb = np.reshape(sorted_pred[i]['bbox'], [-1]).astype(np.float32)
            BBGT = np.array(test[img_id])
            ovmax = 0
            max_id = 0
            if len(BBGT) > 0:
                iou = compute_iou(bb, BBGT)
                ovmax = np.max(iou)
                max_id = np.argmax(iou)
            if ovmax >= iou_thresh:
                if det[img_id][max_id] == 0:
                    tp[i] = 1
                    det[img_id][max_id] = 1
                else:
                    fp[i] = 1
            else:
                fp[i] = 1

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(gt_total)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)

        return ap
