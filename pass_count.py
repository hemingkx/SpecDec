import os
import sys
import time
import logging
from tqdm import tqdm

import torch
from fairseq import utils, tasks, options
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

from torch import Tensor
from typing import Dict, List, Optional

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
def AR_forward_decoder(model,
                       input_tokens,
                       encoder_out: Dict[str, List[Tensor]],
                       incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
                       parallel_forward_start_pos=None,
                       temperature: float = 1.0,
                       use_log_softmax=True,
                       beta: int = 1,
                       tau: float = 0.0):
    decoder_out = model.decoder.forward(input_tokens,
                                        encoder_out=encoder_out,
                                        incremental_state=incremental_state,
                                        parallel_forward_start_pos=parallel_forward_start_pos)
    decoder_out_tuple = (decoder_out[0].div_(temperature), decoder_out[1])
    if use_log_softmax:
        probs = model.get_normalized_probs(decoder_out_tuple, log_probs=True, sample=None)
    else:
        probs = decoder_out_tuple[0]
    topk_scores, indexes = torch.topk(probs, beta, dim=-1)
    topk_scores = topk_scores[0].tolist()
    indexes = indexes[0].tolist()
    for i in range(len(topk_scores)):
        for j, s in enumerate(topk_scores[i]):
            if topk_scores[i][0] - s > tau:
                indexes[i][j] = -1
    return indexes


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


def block_generate(data_lines, model, AR_model, task, block_size, device, max_len=200, beta=1, tau=0):
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary
    encoder_state_ids = []
    for i in range(len(AR_model.decoder.layers)):
        encoder_state_ids.append(AR_model.decoder.layers[i].encoder_attn._incremental_state_id)
    data_size = len(data_lines)
    all_results = []
    logger.info(f'Block generate')
    pass_tokens = [0] * max_len
    sent_nums = [0] * max_len
    start = time.perf_counter()
    for start_idx in tqdm(range(0, data_size)):
        bpe_line = data_lines[start_idx]

        src_tokens = src_dict.encode_line(bpe_line, add_if_not_exist=False).long()
        net_input = {'src_tokens': src_tokens.unsqueeze(0).to(device),
                     'src_lengths': torch.LongTensor([src_tokens.numel()]).to(device)}
        AR_encoder_out = AR_model.encoder.forward_torchscript(net_input)
        encoder_out = model.encoder.forward_torchscript(net_input)

        incremental_state = torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]],
                                               torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {}))

        prev_output_tokens = [tgt_dict.unk()] * block_size
        start_pos = 0
        for step in range(0, max_len):
            start_pos, prev_output_tokens, pass_token = block_forward(incremental_state, encoder_state_ids,
                                                                      start_pos, block_size, tgt_dict,
                                                                      prev_output_tokens,
                                                                      encoder_out, AR_encoder_out, model,
                                                                      AR_model, beta, tau)
            pass_tokens[step] += pass_token
            sent_nums[step] += 1
            if start_pos == -1:
                break
        all_results.append(tgt_dict.string(prev_output_tokens))
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
    remove_bpe_results = [line.replace('@@ ', '') for line in all_results]
    return remove_bpe_results, delta


def block_forward(incremental_state, encoder_state_ids, start_pos, block_size, tgt_dict, prev_output_tokens,
                  encoder_out, AR_encoder_out, model, AR_model, beta, tau, max_len=200):
    output_tokens = torch.tensor([prev_output_tokens]).to(device)
    _scores, _tokens = model.decoder(
        normalize=False,
        prev_output_tokens=output_tokens,
        encoder_out=encoder_out,
    ).max(-1)

    prev_output_tokens[start_pos:start_pos + block_size] = _tokens[0].tolist()[start_pos:start_pos + block_size]

    cut_incremental_state(incremental_state, keep_len=start_pos, encoder_state_ids=encoder_state_ids)

    cur_span_input_tokens = torch.tensor([[tgt_dict.eos()] + prev_output_tokens]).to(device)
    AR_topk_tokens = AR_forward_decoder(AR_model,
                                        cur_span_input_tokens,
                                        AR_encoder_out,
                                        incremental_state,
                                        use_log_softmax=False,
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
                        help='path to AT verifier model')
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

    # Load dataset splits
    task = tasks.setup_task(cfg.task)

    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    # load model
    models, _model_args, _model_task = load_model_ensemble_and_task(filenames=[cfg.common_eval.path],
                                                                    task=task)

    if cmd_args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    model = models[0].to(device).eval()

    AR_model = None
    AR_models = None
    _AR_model_task = None
    if cmd_args.AR_path is not None:
        AR_models, _AR_model_args, _AR_model_task = load_model_ensemble_and_task(filenames=[cmd_args.AR_path],
                                                                                 arg_overrides={'data': cfg.task.data})
        AR_model = AR_models[0].to(device).eval()
        logging.info("AR model loaded!")

    with open(cmd_args.input_path, 'r') as f:
        bpe_sents = [l.strip() for l in f.readlines()]

    logger.info("Decoding Strategy: Block")
    remove_bpe_results, delta = block_generate(bpe_sents, model, AR_model, task, cmd_args.block_size, device, beta=cmd_args.beta, tau=cmd_args.tau)
    logger.info(f'Block generate: {delta}')

    if cmd_args.output_path is not None:
        write_result(remove_bpe_results, cmd_args.output_path)
