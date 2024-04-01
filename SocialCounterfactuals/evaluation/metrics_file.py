import math

def skew_metric(topk_each_label_cnt, each_label_cnt, topk, total_label_cnt):
    numerator = (topk_each_label_cnt / each_label_cnt)
    denominator = (topk / total_label_cnt)

    #print((numerator, denominator))

    skew_value = math.log((numerator / denominator))
    return skew_value

def NDKL(topk, skew_scores, label_cnts):
    z_val = 0.0
    for i in range(1, topk + 1):
        z_val += (1. / (math.log(i + 1)))
    
    ndkl_val = 0.0
    return ndkl_val