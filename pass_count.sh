data_dir=./dict_dir # the dir that contains dict files
checkpoint_path=./nat.pt # the dir that contains NAT drafter checkpoint
AR_checkpoint_path=./at.pt # the dir that contains AT verifier checkpoint
input_path=./test.en # the dir that contains bpe test files
output_path=./output/block.out # the dir for outputs

beta=5
tau=3.0
block_size=30

src=en
tgt=de


python pass_count.py $data_dir --path $checkpoint_path \
      --user-dir block_plugins --task translation_lev_modified --remove-bpe --max-sentences 20 --source-lang ${src} \
      --target-lang ${tgt} --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --iter-decode-with-beam 1 \
      --gen-subset test --AR-path $AR_checkpoint_path --input-path $input_path --output-path $output_path \
      --block-size ${block_size} --beta ${beta} --tau ${tau}
