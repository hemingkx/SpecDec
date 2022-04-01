# Generalized Aggressive Decoding

Implementation for the paper "Lossless Speedup of Autoregressive Translation with
Generalized Aggressive Decoding"

### Requirements

- Python >= 3.7
- Pytorch >= 1.5.0

### Installation

```
conda create -n gad python=3.7
cd GAD
pip install --editable .
```

### Train

For training the NAT drafter of GAD (check `train.sh`)

```
python train.py ${bin_path} --arch block --noise block_mask --share-all-embeddings \
    --criterion glat_loss --label-smoothing 0.1 --lr ${lr} --warmup-init-lr 1e-7 \
    --stop-min-lr 1e-9 --lr-scheduler inverse_sqrt --warmup-updates ${warmup} \
    --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 \
    --task translation_lev_modified --max-tokens ${max_tokens} --weight-decay 0.01 \
    --dropout ${dropout} --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 \
    --decoder-embed-dim 512 --fp16 --max-source-positions 1000 \
    --max-target-positions 1000 --max-update ${update} --seed ${seed} --clip-norm 5 \
    --save-dir ./checkpoints --src-embedding-copy --log-interval 1000 \
    --user-dir block_plugins --block-size ${size} --total-up ${update} \
    --update-freq ${update_freq} --decoder-learned-pos --encoder-learned-pos \
    --apply-bert-init --activation-fn gelu
```

### Inference

For vanilla GAD  (check `inference.sh`):

```
python inference.py $data_dir --path $checkpoint_path --user-dir block_plugins \
    --task translation_lev_modified --remove-bpe --max-sentences 20 \
    --source-lang ${src} --target-lang ${tgt} --iter-decode-max-iter 0 \
    --iter-decode-eos-penalty 0 --iter-decode-with-beam 1 --gen-subset test \
    --strategy ${strategy} --AR-path $AR_checkpoint_path --beam $BEAM \
    --input-path $input_path --output-path $output_path --batch $BATCH \
    --block-size ${block_size}
```

For GAD++ (with calculating the average decoding iteration and mean accepted tokens, check `pass_count.sh`)

```
python pass_count.py $data_dir --path $checkpoint_path --user-dir block_plugins \
    --task translation_lev_modified --remove-bpe --max-sentences 20 \
    --source-lang ${src} --target-lang ${tgt} --iter-decode-max-iter 0 \
    --iter-decode-eos-penalty 0 --iter-decode-with-beam 1 --gen-subset test \
    --AR-path $AR_checkpoint_path --input-path $input_path \
    --output-path $output_path --block-size ${block_size} --beta ${beta} --tau ${tau}
```

Calculating compound split bleu:

```
./ref.sh
```

### Note

This code is based on GLAT [(https://github.com/FLC777/GLAT)](https://github.com/FLC777/GLAT). 

