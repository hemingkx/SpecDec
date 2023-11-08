import os
import sys
import time
import logging
from tqdm import tqdm

import torch
from fairseq import tasks, options
from fairseq.models.bart import BARTModel
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

from torch import Tensor
from typing import Dict, List, Optional

# CNN_KWARGS = dict(beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s |  [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("inference")


def write_result(results, output_file):
    with open(output_file, 'w') as f:
        for line in results:
            f.write(line + '\n')


@torch.no_grad()
def fairseq_generate(data_lines, bart, batch_size=32, **eval_kwargs):
    data_size = len(data_lines)
    sents = [data_lines[0]]
    remove_bpe_results = []
    logger.info(f'Fairseq generate batch {batch_size}')
    start = time.perf_counter()
    for start_idx in tqdm(range(1, data_size)):
        if (start_idx) % batch_size == 0:
            hypotheses_batch = bart.sample(sents, **eval_kwargs)
            for hypothesis in hypotheses_batch:
                remove_bpe_results.append(hypothesis)
            sents = []
        sents.append(data_lines[(start_idx)])
    if sents != []:
        hypotheses_batch = bart.sample(sents, **eval_kwargs)
        for hypothesis in hypotheses_batch:
            remove_bpe_results.append(hypothesis)
    delta = time.perf_counter() - start
    return remove_bpe_results, delta


@torch.no_grad()
def baseline_forward_decoder(model,
                             input_tokens,
                             encoder_out: Dict[str, List[Tensor]],
                             incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
                             parallel_forward_start_pos=None,
                             temperature: float = 1.0):
    decoder_out = model.decoder.forward(input_tokens,
                                        encoder_out=encoder_out,
                                        incremental_state=incremental_state,
                                        parallel_forward_start_pos=parallel_forward_start_pos)
    decoder_out_tuple = (decoder_out[0].div_(temperature), decoder_out[1])
    pred_tokens = torch.argmax(decoder_out_tuple[0], dim=-1).squeeze(0)
    return pred_tokens


@torch.no_grad()
def baseline_generate(data_lines, bart, task, device, max_len=200):
    # simplified AR greedy decoding
    tgt_dict = task.target_dictionary
    data_size = len(data_lines)
    remove_bpe_results = []
    logger.info(f'Baseline generate')
    start = time.perf_counter()
    for start_idx in tqdm(range(0, data_size)):
        src_tokens = bart.encode(data_lines[start_idx])
        net_input = {'src_tokens': src_tokens.unsqueeze(0).to(device),
                     'src_lengths': torch.LongTensor([src_tokens.numel()]).to(device)}
        encoder_out = bart.model.encoder.forward_torchscript(net_input)
        incremental_state = torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]],
                                               torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {}))
        tokens = [tgt_dict.eos()]
        for step in range(0, max_len):
            cur_input_tokens = torch.tensor([tokens]).to(device).long()
            pred_token = baseline_forward_decoder(bart.model,
                                                  cur_input_tokens,
                                                  encoder_out,
                                                  incremental_state).item()
            if pred_token == tgt_dict.eos():
                break
            else:
                tokens.append(pred_token)
        token_tensor = torch.IntTensor(tokens).to(device)
        result = bart.decode(token_tensor)
        remove_bpe_results.append(result)
    delta = time.perf_counter() - start
    return remove_bpe_results, delta


@torch.no_grad()
def cut_incremental_state(incremental_state, keep_len, encoder_state_ids):
    for n in incremental_state:
        if n[: n.index('.')] in encoder_state_ids:
            continue
        for k in incremental_state[n]:
            if incremental_state[n][k] is not None:
                if incremental_state[n][k].dim() == 4:
                    incremental_state[n][k] = incremental_state[n][k][:, :, :keep_len]
                elif incremental_state[n][k].dim() == 2:
                    incremental_state[n][k] = incremental_state[n][k][:, :keep_len]


@torch.no_grad()
def forward_decoder(model,
                    input_tokens,
                    encoder_out: Dict[str, List[Tensor]],
                    incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
                    parallel_forward_start_pos=None,
                    block_mask=None,
                    temperature: float = 1.0,
                    beta: int = 1,
                    tau: float = 0.0):
    decoder_out = model.decoder.forward(input_tokens,
                                        encoder_out=encoder_out,
                                        incremental_state=incremental_state,
                                        parallel_forward_start_pos=parallel_forward_start_pos,
                                        block_mask=block_mask)
    decoder_out_tuple = (decoder_out[0].div_(temperature), decoder_out[1])
    topk_scores, indexes = torch.topk(decoder_out_tuple[0], beta, dim=-1)
    topk_scores = topk_scores[0].tolist()
    indexes = indexes[0].tolist()
    for i in range(len(topk_scores)):
        for j, s in enumerate(topk_scores[i]):
            if topk_scores[i][0] - s > tau:
                indexes[i][j] = -1
    return indexes


@torch.no_grad()
def specdec_generate(data_lines, model, bart, task, block_size, device, max_len=200, beta=1, tau=0):
    tgt_dict = task.target_dictionary
    encoder_state_ids = []
    data_size = len(data_lines)
    remove_bpe_results = []
    logger.info(f'SpecDec generate')
    pass_tokens = [0] * max_len
    sent_nums = [0] * max_len
    start = time.perf_counter()
    for i in range(len(bart.model.decoder.layers)):
        encoder_state_ids.append(bart.model.decoder.layers[i].encoder_attn._incremental_state_id)
    for start_idx in tqdm(range(0, data_size)):
        src_tokens = bart.encode(data_lines[start_idx])
        net_input = {'src_tokens': src_tokens.unsqueeze(0).to(device),
                     'src_lengths': torch.LongTensor([src_tokens.numel()]).to(device)}
        AR_encoder_out = bart.model.encoder.forward_torchscript(net_input)
        encoder_out = model.encoder.forward_torchscript(net_input)
        incremental_state = torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]],
                                               torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {}))

        prev_output_tokens = [tgt_dict.unk()] * block_size
        start_pos = 0
        for step in range(0, max_len):
            start_pos, prev_output_tokens, pass_token = specdec_forward(incremental_state, encoder_state_ids,
                                                                        start_pos, block_size, tgt_dict,
                                                                        prev_output_tokens,
                                                                        encoder_out, AR_encoder_out, model,
                                                                        AR_model, beta, tau)
            pass_tokens[step] += pass_token
            sent_nums[step] += 1
            if start_pos == -1:
                break
        token_tensor = torch.IntTensor(prev_output_tokens).to(device)
        result = bart.decode(token_tensor)
        remove_bpe_results.append(result)
    total_pass_tokens = 0
    total_sent_nums = 0
    for step in range(max_len):
        if sent_nums[step] > 0:
            total_pass_tokens += pass_tokens[step]
            total_sent_nums += sent_nums[step]
    print("Avg accepted tokens:", total_pass_tokens / total_sent_nums)
    total_iter = 0
    for step in range(max_len):
        if sent_nums[step - 1] > 0:
            if step == 0:
                last_num = data_size
            else:
                last_num = sent_nums[step - 1]
            if (last_num - sent_nums[step]) > 0:
                total_iter += (last_num - sent_nums[step]) * (step)
    print("Avg decoding iteration:", total_iter / data_size)
    delta = time.perf_counter() - start
    return remove_bpe_results, delta


@torch.no_grad()
def specdec_forward(incremental_state, encoder_state_ids, start_pos, block_size, tgt_dict, prev_output_tokens,
                    encoder_out, AR_encoder_out, model, bart, beta, tau, max_len=200):
    output_tokens = torch.tensor([prev_output_tokens]).to(device)
    block_mask = torch.zeros_like(output_tokens).to(output_tokens)
    block_mask[0][start_pos:start_pos + block_size] = 1
    _, _tokens = model.decoder(
        normalize=False,
        prev_output_tokens=output_tokens,
        encoder_out=encoder_out,
        block_mask=block_mask.bool(),
    ).max(-1)

    prev_output_tokens[start_pos:start_pos + block_size] = _tokens[0].tolist()
    cut_incremental_state(incremental_state, keep_len=start_pos, encoder_state_ids=encoder_state_ids)
    cur_span_input_tokens = torch.tensor([[tgt_dict.eos()] + prev_output_tokens]).to(device)
    AR_topk_tokens = forward_decoder(bart.model,
                                     cur_span_input_tokens,
                                     AR_encoder_out,
                                     incremental_state,
                                     parallel_forward_start_pos=start_pos,
                                     beta=beta,
                                     tau=tau)
    bifurcation = block_size
    for i, (token, AR_topk_token) in enumerate(zip(prev_output_tokens[start_pos:], AR_topk_tokens[:-1][:])):
        if token not in AR_topk_token:
            bifurcation = i
            break

    next_output_tokens = prev_output_tokens[:start_pos + bifurcation] + [AR_topk_tokens[bifurcation][0]] + [
        tgt_dict.unk()] * block_size

    pass_token = 0
    find_eos = False
    for i, o in enumerate(prev_output_tokens[start_pos:start_pos + bifurcation] + [AR_topk_tokens[bifurcation][0]]):
        if o == tgt_dict.eos() or i + start_pos == max_len:
            next_output_tokens = next_output_tokens[0:start_pos + i]
            start_pos = -1
            pass_token = i
            find_eos = True
            break

    if not find_eos:
        start_pos = start_pos + bifurcation + 1
        pass_token = bifurcation + 1

    return start_pos, next_output_tokens, pass_token


if __name__ == '__main__':
    parser = options.get_generation_parser()
    parser.add_argument('--input-path', type=str, required=True,
                        help='path to eval file')
    parser.add_argument('--output-path', type=str, default=None,
                        help='path to output file')
    parser.add_argument('--AR-path', type=str, default=None,
                        help='path to AR model')
    parser.add_argument('--strategy', type=str, default='fairseq',
                        help='decoding strategy, choose from: fairseq, AR, specdec')
    parser.add_argument('--batch', type=int, default=1,
                        help='batch size')
    parser.add_argument('--max-len', type=int, default=200,
                        help='max-len')
    parser.add_argument('--block-size', type=int, default=5,
                        help='block size')
    parser.add_argument('--beta', type=int, default=1,
                        help='top-beta hyperparameter')
    parser.add_argument('--tau', type=float, default=0,
                        help='tolerance hyperparameter')
    cmd_args = options.parse_args_and_arch(parser)
    cmd_args.input_path = os.path.expanduser(cmd_args.input_path)
    cmd_args.output_path = os.path.expanduser(cmd_args.output_path)

    cfg = convert_namespace_to_omegaconf(cmd_args)

    CNN_KWARGS = dict(beam=cfg.generation.beam, max_len_b=cmd_args.max_len, lenpen=0)

    # Load dataset splits
    task = tasks.setup_task(cfg.task)

    if cmd_args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    if cmd_args.strategy == 'specdec':
        logger.info("loading model(s) from {}".format(cfg.common_eval.path))
        models, _model_args, _model_task = load_model_ensemble_and_task(filenames=[cfg.common_eval.path], task=task)
        model = models[0].to(device).eval()

    AR_model = BARTModel.from_pretrained(cfg.task.data, checkpoint_file=cmd_args.AR_path,
                                         data_name_or_path=cfg.task.data)
    AR_model = AR_model.to(device).eval()
    logging.info("AR model loaded!")

    with open(cmd_args.input_path, 'r') as f:
        raw_sents = [l.strip() for l in f.readlines()]

    if cmd_args.strategy == 'AR':
        logger.info("Decoding Strategy: Simplified AR")
        remove_bpe_results, delta = baseline_generate(raw_sents, AR_model, task, device, max_len=cmd_args.max_len)
        logger.info(f"AR generate: {delta}")
    elif cmd_args.strategy == 'specdec':
        logger.info("Decoding Strategy: SpecDec")
        remove_bpe_results, delta = specdec_generate(raw_sents, model, AR_model, task, cmd_args.block_size, device,
                                                     max_len=cmd_args.max_len, beta=cmd_args.beta, tau=cmd_args.tau)
        logger.info(f'SpecDec generate: {delta}')
    else:
        logger.info("Decoding Strategy: fairseq")
        eval_kwargs = CNN_KWARGS
        remove_bpe_results, delta = fairseq_generate(raw_sents, AR_model, batch_size=cmd_args.batch, **eval_kwargs)
        logger.info(f'Fairseq generate batch {cmd_args.batch}, beam {cfg.generation.beam}: {delta}')

    if cmd_args.output_path is not None:
        write_result(remove_bpe_results, cmd_args.output_path)