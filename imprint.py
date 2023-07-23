# coding: utf-8
# url: https://github.com/YU1ut/imprinted-weights/blob/HEAD/imprint_ft.py
import torch
import torch.nn as nn
from tqdm import tqdm


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output


def weight_norm(w):
    norm = w.norm(p=2, dim=1, keepdim=True)
    return w.div(norm.expand_as(w))


def imprint(novel_loader, model, num_classes, device, is_random, embedding_size):
    """
    Parameters
    ----------
    novel_loader : Dataloader
        Dataloader
    model : model
        The model to imprint
    num_classes : Int
        Number of classes
    device : device
    random : Boolean
        DESCRIPTION.

    Returns
    -------
    model : model
        Our model imprinted
    """
    # Switch to evaluate mode
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, (img, lbl) in tqdm(enumerate(novel_loader)):
            img = img.to(device)
            lbl = lbl.float().to(device)

            # compute output
            output = model(img)
            output = l2_norm(output)  # lose embeding function (nn.Linear)

            if batch_idx == 0:
                output_stack = output
                target_stack = lbl
            else:
                output_stack = torch.cat((output_stack, output), 0)
                target_stack = torch.cat((target_stack, lbl), 0)

    new_weight = torch.zeros(num_classes, embedding_size)
    init_classes_num = 0
    for i in range(num_classes):
        tmp = (
            output_stack[target_stack == (i + init_classes_num)].mean(0)
            if not is_random
            else torch.randn(embedding_size)
        )
        new_weight[i] = tmp / tmp.norm(p=2)

    # weight = torch.cat((model.classifier.fc.weight.data, new_weight.cuda()))
    return new_weight
    model.fc = nn.Linear(embedding_size, num_classes, bias=False)
    model.fc.weight.data = new_weight
    # model.fc.weight.data = weight_norm(new_weight)

    return model
