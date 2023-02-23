# Summarization

## Introduction

This folder contains the codes and checkpoints of GAD on the summarization task (CNN-Daily Mail).

## Download

| Description | Model                                                        |
| ----------- | ------------------------------------------------------------ |
| Model       | [at-verifier-base](https://drive.google.com/file/d/1Kp8W89QjjSC7JbxgxQLkPW6jaczw18Ct/view?usp=sharing)， [nat-drafter-base (k=25)](https://drive.google.com/file/d/1JvRNV4QsoWpVs1bHiozeJb8kRnln4x1K/view?usp=sharing) |
| Test Data   | [cnn-dm-test](https://drive.google.com/drive/folders/1eZON9kb5Ga2bHN0_v24Q1BsopWVTZDFL?usp=sharing) |

## Installation

```
conda activate gad
pip install requests # to download BART encoder.json
```

## Preprocess

The raw datasets we used can be obtained following [Fairseq-Summarization](https://github.com/facebookresearch/fairseq/blob/main/examples/bart/README.summarization.md). We release our dicts in `./data` and our raw test data above. Same as the translation task, you need to prepare the distilled data with a trained autoregressive Transformer before training. As for inference, you can directly test the model's performance with our provided raw test data. 

## Finetune

We use BART to initialize **our AT verifier** and finetune it on the distilled data.

```
fairseq-train cnn_dm-distilled-bin \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_base \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters;
```

For training **the NAT drafter** of GAD:

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
    --apply-bert-init --activation-fn gelu \
    --restore-file ./checkpoints/initial_checkpoint.pt \
    --reset-optimizer --reset-meters --reset-lr-scheduler --reset-dataloader
```

## Inference

For GAD++   (check `inference_paper.sh`, set `beta=1` for vanilla GAD):

```
python inference_paper.py ${data_dir} --path ${checkpoint_path} --user-dir block_plugins \
      --task translation_lev_modified --remove-bpe --max-sentences 20 \
      --source-lang source --target-lang target --iter-decode-max-iter 0 \
      --iter-decode-eos-penalty 0 --iter-decode-with-beam 1 --gen-subset test \
      --AR-path ${AR_checkpoint_path} --input-path ${input_path} --strategy ${strategy} \
      --output-path ${output_path} --block-size ${block_size} --beta ${beta} --tau ${tau}
```

For calculating rouge, install `files2rouge` from [here](https://github.com/pltrdy/files2rouge). Then run the scripts below:

```
./rouge.sh
```

