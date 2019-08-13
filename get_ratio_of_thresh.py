import sys

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("USAGE:python %s ratiofile thresh FA_per_hour negative_dur"%(sys.argv[0]))
        exit(1)
    thresh = float(sys.argv[2])
    num_FA = float(sys.argv[3]) * float(sys.argv[4])
    anchor_ratio_sum = 0.0
    region_ratio_sum = 0.0
    true_positive = 0.0
    num_positive  = 0.0
    for line in open(sys.argv[1]).readlines():
        score, anchor_ratio, region_ratio = map(float, line.strip().split())
        num_positive += 1.0
        if score >= thresh:
            anchor_ratio_sum += anchor_ratio
            region_ratio_sum += region_ratio
            true_positive += 1.0
    print("Average anchor ratio:%f/%f, Average region ratio:%f/%f\n"%(anchor_ratio_sum/true_positive, anchor_ratio_sum/(true_positive+num_FA), region_ratio_sum/true_positive, region_ratio_sum/(true_positive+num_FA)))
    print("Miss rate: %f\n"%((num_positive-true_positive)/num_positive*100))
    print(str(num_positive))
