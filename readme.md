# Generalized Aggressive Decoding

Implementation for the paper "Lossless Speedup of Autoregressive Translation with
Generalized Aggressive Decoding"

### Requirements

- Python >= 3.7
- Pytorch >= 1.5.0

### Installation

```
conda create -n gad python=3.7
cd Generalized-Aggressive-Decoding
pip install --editable .
```

### Train

Train GAD

```
./train.sh
```

### Inference

For vanilla GAD:

```
./inference.sh
```

For GAD++ (with the average decoding iteration and mean accepted tokens):

```
./pass_count.sh
```

Calculating compound split bleu:

```
./ref.sh
```

