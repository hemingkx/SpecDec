data_dir=./data # the dir that contains dict files
checkpoint_path=./checkpoints/cnn-dm-base-nat-drafter-checkpoint.avg10.pt # the dir that contains NAT drafter checkpoint
AR_checkpoint_path=./checkpoints/cnn-dm-base-at-verifier.pt # the dir that contains AT verifier checkpoint
input_path=./data/test.source # the dir that contains raw test files
output_path=./output/result.hypo # the dir for outputs

strategy='gad' # 'fairseq', 'AR', 'gad'
batch=1
beam=1
max_len=200

beta=5
tau=3.0
block_size=25


python inference_paper.py $data_dir --path $checkpoint_path --user-dir block_plugins \
      --task translation_lev_modified --remove-bpe --max-sentences 20 --source-lang source \
      --target-lang target --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --iter-decode-with-beam 1 \
      --gen-subset test --strategy ${strategy} --AR-path $AR_checkpoint_path --input-path $input_path \
      --output-path $output_path --batch ${batch} --block-size ${block_size} --beam ${beam} --max-len ${max_len} \
      --beta ${beta} --tau ${tau}