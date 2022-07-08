import torch


def model_preparing(model_path, save_path):
    """only save the AT encoder params to initialize the NAT drafter's encoder"""
    key_l = []
    raw_model = torch.load(model_path)
    for key in raw_model['model']:
        if key.startswith('decoder'):
            key_l.append(key)
    for key in key_l:
        del raw_model['model'][key]
    print('*' * 100)
    for key in raw_model['model']:
        print(key)
    torch.save(raw_model, save_path)


def param_checking(model1, model2):
    """check the parameters of the AT verifier and the NAT drafter"""
    key_l1 = []
    key_l2 = []
    raw_model1 = torch.load(model1)
    for key in raw_model1['model']:
        key_l1.append(key)
    raw_model2 = torch.load(model2)
    for key in raw_model2['model']:
        key_l2.append(key)
    print(key_l1)
    print(key_l2)
    # print(raw_model1['model']['encoder.embed_positions.weight'].size())
    # print(raw_model2['model']['encoder.embed_positions.weight'].size())
    for k1, k2 in zip(key_l1, key_l2):
        if k1 != k2:
            print(k1)
            print(k2)


if __name__ == "__main__":
    AR_path = './checkpoints/wmt14-en-de-base-at-verifier.pt'  # the dir that contains AT verifier checkpoint
    save_path = './checkpoints/initial_checkpoint.pt'  # the save dir of your fairseq NAT drafter checkpoints
    model_preparing(AR_path, save_path)
