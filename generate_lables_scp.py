import sys
LableDict = {'freetext': 0,
             'hixiaowen': 1,
             'nihaowenwen': 2,
             '1':1
            };
SIL=1

def get_kws_lables(utt_id_labels):
    label = []
    keyword_type = utt_id_labels.strip().split()[0].strip().split("_")[-1]
    phones = utt_id_labels.strip().split()[1:]
    # get class
    label.append(LableDict[keyword_type])
    if(LableDict[keyword_type] == 0):
        label.append(-1)
        label.append(-1)
    else:
        # get position
        phones_int = [int(x) for x in phones]
        start_position = 0
        while(phones_int[start_position] == SIL):
            start_position += 1
        end_position = len(phones_int)-1
        while(phones_int[end_position] == SIL):
            end_position -= 1
        label.append(start_position)
        label.append(end_position)

    return label

def write_kws_lables(fid, utt_id, label, factor):
    fid.writelines(utt_id)
    fid.writelines(" %d"%(label[0]))
    fid.writelines(",%d"%(label[1]*factor))
    fid.writelines(",%d"%(label[2]*factor+factor-1))
    fid.writelines("\n")

if __name__=="__main__":
    if(len(sys.argv) < 3):
        print("USAGE:python %s phone_ali.ark kws_lable.scp [sub_sampling_factor]"%(sys.argv[0]))
    fid = open(sys.argv[2],"w")
    if (len(sys.argv) == 4):
        factor = int(sys.argv[3])
    for x in open(sys.argv[1]).readlines():
        utt_id = x.strip().split()[0]
        label = get_kws_lables(x)
        write_kws_lables(fid, utt_id, label, factor)
    fid.close()


