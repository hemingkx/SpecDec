import os
import sys
import time
import logging
from tqdm import tqdm

import torch
from fairseq import utils, tasks, options
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

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
def drafter_generate(data_lines, model, task, block_size, device, max_len=200):
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary
    data_size = len(data_lines)
    all_results = []
    logger.info(f'Spec-Drafter generate')
    pass_tokens = [0] * max_len
    sent_nums = [0] * max_len
    start = time.perf_counter()
    for start_idx in tqdm(range(0, data_size)):
        bpe_line = data_lines[start_idx]
        src_tokens = src_dict.encode_line(bpe_line, add_if_not_exist=False).long()
        net_input = {'src_tokens': src_tokens.unsqueeze(0).to(device),
                     'src_lengths': torch.LongTensor([src_tokens.numel()]).to(device)}
        encoder_out = model.encoder.forward_torchscript(net_input)

        prev_output_tokens = [tgt_dict.unk()] * block_size
        start_pos = 0
        for step in range(0, max_len):
            start_pos, prev_output_tokens, pass_token = drafter_forward(start_pos, block_size, tgt_dict,
                                                                        prev_output_tokens, encoder_out,  model)
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


@torch.no_grad()
def drafter_forward(start_pos, block_size, tgt_dict, prev_output_tokens, encoder_out, model, max_len=200):
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

    next_output_tokens = prev_output_tokens + [tgt_dict.unk()] * block_size
    pass_token = 0
    find_eos = False
    for i, o in enumerate(prev_output_tokens[start_pos:start_pos + block_size]):
        if o == tgt_dict.eos() or i + start_pos == max_len:
            next_output_tokens = next_output_tokens[0:start_pos + i]
            start_pos = -1
            pass_token = i
            find_eos = True
            break

    if not find_eos:
        start_pos = start_pos + block_size
        pass_token = block_size

    return start_pos, next_output_tokens, pass_token


if __name__ == '__main__':
    parser = options.get_generation_parser()
    parser.add_argument('--input-path', type=str, required=True,
                        help='path to eval file, e.g., /to/path/conll14.bpe.txt')
    parser.add_argument('--output-path', type=str, default=None,
                        help='path to output file, e.g., /to/path/conll14.pred.txt')
    parser.add_argument('--AR-path', type=str, default=None,
                        help='path to AR model')
    parser.add_argument('--block-size', type=int, default=5,
                        help='block size')
    cmd_args = options.parse_args_and_arch(parser)
    cmd_args.input_path = os.path.expanduser(cmd_args.input_path)
    cmd_args.output_path = os.path.expanduser(cmd_args.output_path)

    cfg = convert_namespace_to_omegaconf(cmd_args)

    task = tasks.setup_task(cfg.task)

    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, _model_args, _model_task = load_model_ensemble_and_task(filenames=[cfg.common_eval.path],
                                                                    task=task)

    if cmd_args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    model = models[0].to(device).eval()

    with open(cmd_args.input_path, 'r') as f:
        bpe_sents = [l.strip() for l in f.readlines()]

    logger.info("Decoding Strategy: Spec-Drafter")
    remove_bpe_results, delta = drafter_generate(bpe_sents, model, task, cmd_args.block_size, device)
    logger.info(f'Spec-Drafter generate: {delta}')

    if cmd_args.output_path is not None:
        write_result(remove_bpe_results, cmd_args.output_path)
