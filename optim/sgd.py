import torch
from .optimizer import Optimizer, required


class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, model, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, use_dgc=False,
                 ratio=0.6, reduce_time=False, collect_ratio=0.5):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        self.ratio = ratio
        self.reduce_time = reduce_time
        self.collect_ratio = collect_ratio
        self.use_dgc = use_dgc
        if self.use_dgc:
            model.optim = self
        super(SGD, self).__init__(model.parameters(), defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if self.use_dgc:
                    p.data.add_(-group['lr'], d_p)
                    continue
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss

    def get_v_accum(self, param):
        """
        not sure if work
        :param p: params in model
        :return: sparsed tensor v_local
        """
        param_state = self.state[param]
        group = self.param_groups[0]
        momentum = group['momentum']
        dampening = group['dampening']
        if 'momentum_buffer' not in param_state:
            u_local = param_state['momentum_buffer'] = torch.zeros_like(param.data)
            u_local.mul_(momentum).add_(param.grad.data)
        else:
            u_local = param_state['momentum_buffer']
            u_local.mul_(momentum).add_(1 - dampening, param.grad.data)
        if "v_accum" not in param_state:
            v_accum = param_state["v_accum"] = torch.zeros_like(param.data)
        else:
            v_accum = param_state["v_accum"]
        v_accum.add_(u_local)

        threshold = self._find_threshold(v_accum)

        mask = ((v_accum >= threshold) +
                (v_accum <= -threshold)) * torch.ones(v_accum.size()).byte().cuda()
        param_state['mask'] = mask
        not_mask = mask ^ 1

        result = v_accum.masked_fill(not_mask, 0)
        v_accum.masked_fill_(mask, 0)
        return result

    def _find_threshold(self, source):
        """
        对Tensor source执行比例为collect_ratio的随机采样(是否采样根据reduce_time判断)
        找到近似绝对值top-ratio的数
        不改变source
        """
        if self.reduce_time:
            abs_source = source.abs()
            collect_size = int(source.numel() * self.collect_ratio)
            abs_source.resize_(abs_source.numel(), 1)
            perm = torch.randperm(min(589, abs_source.size(0)))
            idx = perm[:collect_size]
            abs_source = abs_source[idx]  # collect_size行1列
        else:
            abs_source = source.abs()

        top_k = int(abs_source.numel() * self.ratio)
        if top_k <= 0:
            return float('inf')
        abs_source.resize_(1, abs_source.numel())  # 原地resize成一行再排序
        abs_source = torch.topk(abs_source, top_k, sorted=True)[0]
        thr = float(abs_source[0, top_k - 1])
        return thr
