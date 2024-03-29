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
                 reduce_time=False, collect_ratio=0.5,
                 relative=False, eps=1e-5):
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
        self.ratio = None
        self.reduce_time = reduce_time
        self.collect_ratio = collect_ratio
        self.use_dgc = use_dgc
        self.relative = relative
        self.eps = eps
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
            count = 0
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if self.use_dgc:
                    p.data.add_(-group['lr'], d_p)
                    # print(count)
                    # print(d_p.view(-1)[:10])
                    count += 1
                    continue
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

        mask = self._find_mask(v_accum, param.data)
        param_state['mask'] = mask
        not_mask = mask ^ 1

        result = v_accum.masked_fill(not_mask, 0)
        v_accum.masked_fill_(mask, 0)
        return result

    def _find_mask(self, grad, weight):
        """
        对Tensor grad执行比例为collect_ratio的随机采样(是否采样根据reduce_time判断)
        找到近似绝对值top-ratio的数
        不改变grad
        """
        if self.reduce_time:
            abs_grad = grad.abs()
            collect_size = int(grad.numel() * self.collect_ratio)
            abs_grad.resize_(abs_grad.numel(), 1)
            perm = torch.randperm(min(589, abs_grad.size(0)))
            idx = perm[:collect_size]
            abs_grad = abs_grad[idx]
        else:
            abs_grad = grad.abs()

        top_k = max(int(abs_grad.numel() * self.ratio), 1)
        if self.relative:
            target_gard = grad.abs() / (weight.abs() + self.eps)
        else:
            target_gard = grad.abs()
        abs_grad, _ = torch.topk(target_gard.view(-1), top_k)
        threshold = torch.min(abs_grad)
        mask = (target_gard >= threshold) * torch.ones(grad.size()).byte().cuda()
        return mask
