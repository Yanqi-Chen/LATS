import numpy as np

__all__ = ["step_lr", "cosine_lr", "constant_lr", "efficientnet_lr", "get_policy"]


def get_policy(name):
    if name is None:
        return constant_lr

    out_dict = {
        "constant_lr": constant_lr,
        "cosine_lr": cosine_lr,
        "efficientnet_lr": efficientnet_lr,
        "step_lr": step_lr,
        "multistep_lr": multistep_lr,
    }

    return out_dict[name]


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def constant_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        else:
            lr = args.lr

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def cosine_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        else:
            e = epoch - args.warmup_length
            es = args.epochs - args.warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * (args.lr - args.lr_min) + args.lr_min 

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def efficientnet_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        else:
            lr = args.lr * (0.97 ** (epoch / 2.4))

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def step_lr(optimizer, args, **kwargs):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    def _lr_adjuster(epoch, iteration):
        lr = args.lr * (args.lr_gamma ** (epoch // args.lr_adjust))

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster

############# 指定步长scheduler
def multistep_lr(optimizer, args, **kwargs):
    lr = args.lr
    def _lr_adjuster(epoch, iteration):
        nonlocal lr
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        elif epoch in args.lr_milestones:
            lr = args.lr_gamma * lr

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster
##############################



def _warmup_lr(base_lr, warmup_length, epoch):
    return base_lr * (epoch + 1) / warmup_length
