#!/bin/bash

# Copyrigh 2018 houjingyong@gmail.com

# MIT Licence

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh
# Above file is from Kaldi, there are some commands from Kaldi, too.

set -e 

stage=0
path=./
data=$path/fbank
data_type=all

ali_dir=$path/align-fbank
if [ $stage -le 0 ]; then
    # generate lables.scp for each data set according to the alignment file
    gunzip -c $ali_dir/ali.*.gz | ali-to-phones --per-frame=true \
              $ali_dir/final.mdl ark:- ark,t:$data/phone_alignment.ark
    python generate_lables_scp.py $data/phone_alignment.ark $data/labels.scp 1
    # here 1 is the subsampling factor
    for x in train dev test; do
        x=${data_type}_${x}_cmvn
        # prepare torch scp file
        feat-to-len scp:$data/$x/feats.scp ark,t:$data/$x/lens.scp
        python prepare_torch_scp.py $data/$x/feats.scp $data/labels.scp $data/$x/lens.scp 1000 $data/$x/torch.scp
    done
fi
gpu_num=0 # which GPU do you use
batch_size=400 # how many utterances used per mini-batch
learning_rate=0.002 # learning rate
halving_factor=0.5 # annealing penalty factor
num_anchor=20 # how many anchor per frame
left_context=0 
right_context=0
filler=0
output_dim=3 
hidden_dim=128
num_layers=2
dropout=0.5
weight_decay=0
config_file=ticmini2.yml

train_scp=$data/${data_type}_train_cmvn/torch.scp
dev_scp=$data/${data_type}_dev_cmvn/torch.scp
test_scp=$data/${data_type}_test_cmvn/torch.scp
feat_dim=`feat-to-dim scp:$data/all_train_cmvn/feats.scp -`
input_dim=$((($left_context+$right_context+1)*$feat_dim))
proto=Anchor_KWS
layer_type=GRU
echo "Input dim: $input_dim"
echo "Hidden feature dim: $hidden_dim"
echo "Output dim: $output_dim"


previous_model=
for learning_rate in 0.002; do
for lambda in 3; do
save_dir=$path/exp/${data_type}_${proto}_lambda${lambda}_${layer_type}_nl${num_layers}_hd${hidden_dim}_anchor${num_anchor}_bs${batch_size}_lr${learning_rate}_wd${weight_decay}_dp${dropout}_lc${left_context}_rc${right_context}

mkdir -p $save_dir
if [ $stage -le 1 ]; then
    CUDA_VISIBLE_DEVICES=$gpu_num python train_rpn_kws.py \
            --seed=10 \
            --train=1 --test=0 \
            --config-file=$config_file \
            --lambda-factor=$lambda \
            --num-anchor=$num_anchor \
            --input-dim=$input_dim \
            --hidden-dim=$hidden_dim \
            --num-layers=$num_layers \
            --output-dim=$output_dim \
            --dropout=$dropout \
            --left-context=$left_context \
            --right-context=$right_context \
            --max-epochs=20 \
            --min-epochs=10 \
            --batch-size=$batch_size \
            --learning-rate=$learning_rate \
            --init-weight-decay=$weight_decay \
            --halving-factor=$halving_factor \
            --load-model=$previous_model \
            --start-halving-impr=0.01 \
            --end-halving-impr=0.001 \
            --use-cuda=1 \
            --multi-gpu=0 \
            --train-scp=$train_scp \
            --dev-scp=$dev_scp \
            --save-dir=$save_dir \
            --log-interval=10
fi

best_model=$save_dir/final.mdl #log_train_attention_random_window_0.0001_200_2gru_0_0.log
decode_output=ark:$save_dir/test_post.ark
region_output=${save_dir}/test_region.txt
if [ $stage -le 2 ]; then
    # test 
    CUDA_VISIBLE_DEVICES=$gpu_num python train_rpn_kws.py --seed=10 \
            --train=0 --test=1 \
            --config-file=$config_file \
            --lambda-factor=$lambda \
            --num-anchor=$num_anchor \
            --input-dim=$input_dim \
            --hidden-dim=$hidden_dim \
            --num-layers=$num_layers \
            --output-dim=$output_dim \
            --dropout=$dropout \
            --left-context=$left_context \
            --right-context=$right_context \
            --batch-size=$batch_size \
            --load-model=$best_model \
            --use-cuda=1 \
            --multi-gpu=0 \
            --test-scp=$test_scp \
            --output-file="$decode_output" \
            --region-output-file="$region_output" \
            --log-interval=10
fi

#ignore_keyword=1; negative_duration=52.8; num_positive=10641; tag="_1"
ignore_keyword=0; negative_duration=58.3; num_positive=10641; tag=""

if [ $stage -le 3 ]; then
    # get score
    for keyword in hixiaowen nihaowenwen;
    do
    {
        python get_score.py "$decode_output" $keyword \
                "$save_dir/test_${keyword}_score${tag}.txt" $ignore_keyword
    } &
    done
    wait 
fi

if [ $stage -le 4 ]; then
    for keyword in hixiaowen nihaowenwen;
    do
    {
        python compute_roc.py --sliding-window=100 \
                              --start-threshold=0.1 \
                              --end-threshold=1.0 \
                              --threshold-step=0.001 \
                            $save_dir/test_${keyword}_score${tag}.txt \
                            $save_dir/test_${keyword}_roc${tag}.txt \
                            $negative_duration $num_positive
    } &
    done
    wait 
fi

done
done
