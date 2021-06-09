import torch
import torch.nn.functional as F
import torchvision.models as models
from rlkit.pythonplusplus import identity
from typing import Tuple
from rlkit.torch.sac.policies import CustomScriptPolicy, CustomTanhScriptPolicy
import argparse

#r18 = models.resnet18(pretrained=True)   
#r18_scripted = torch.jit.script(r18)
def main(args):
    custom = CustomScriptPolicy(
        100,
        100,
        3,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        8,
        args.net
        ).cuda()
    inp = torch.ones((args.batch_size, 3*args.image_size*args.image_size+8)).cuda()
    out = custom(inp)
    print(out)
    custom_scripted = torch.jit.script(custom)

    #custom_scripted.save("custom_scripted.pt")
    #r18_scripted.save('r18_scripted.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument('batch_size', type=int)
    parser.add_argument('image_size', type=int)
    parser.add_argument('net', choices=["resnet", "densenet"])

    args = parser.parse_args()

    main(args)