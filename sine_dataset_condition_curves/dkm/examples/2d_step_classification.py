#%% Imports
import geotorch
import torch
from torch import nn
from torch.distributions import MultivariateNormal, Normal

from torch.profiler import profile, record_function, ProfilerActivity

from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR

import sys
sys.path.append('../')
import dkm

from sklearn.svm import SVC

import argparse
from timeit import default_timer as timer

from matplotlib import pyplot as plt

import pandas as pd

import psutil

import numpy as np

import math

torch.set_num_threads(4)

#%% CustomAdam
from typing import List, Optional, Union, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import (Optimizer, params_t, _use_grad_for_differentiable, _get_value,
                        _stack_if_compiling, _dispatch_sqrt, _default_to_fused_or_foreach,
                        _capturable_doc, _differentiable_doc, _foreach_doc, _fused_doc,
                        _maximize_doc)
from torch.utils._foreach_utils import _get_fused_kernels_supported_devices

__all__ = ['Adam2', 'adam']



class Adam2(Optimizer):
    def __init__(self,
                 params: params_t,
                 lr: Union[float, Tensor] = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0,
                 amsgrad: bool = False,
                 *,
                 foreach: Optional[bool] = None,
                 maximize: bool = False,
                 capturable: bool = False,
                 differentiable: bool = False,
                 fused: Optional[bool] = None):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if isinstance(lr, Tensor) and foreach and not capturable:
            raise ValueError("lr as a Tensor is not supported for capturable=False and foreach=True")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        maximize=maximize, foreach=foreach, capturable=capturable,
                        differentiable=differentiable, fused=fused)
        super().__init__(params, defaults)

        if fused:
            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            self._step_supports_amp_scaling = True
            # TODO(crcrpar): [low prec params & their higher prec copy]
            # Support AMP with FP16/BF16 model params which would need
            # higher prec copy of params to do update math in higher prec to
            # alleviate the loss of information.
            fused_supported_devices = _get_fused_kernels_supported_devices()
            if not all(
                p.device.type in fused_supported_devices and
                torch.is_floating_point(p) for pg in self.param_groups for p in pg['params']
            ):
                raise RuntimeError("`fused=True` requires all the params to be floating point Tensors of "
                                   f"supported devices: {fused_supported_devices}.")
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
            group.setdefault('differentiable', False)
            group.setdefault('fused', None)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps
    ):
        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    # note(crcrpar): [special device hosting for step]
                    # Deliberately host `step` on CPU if both capturable and fused are off.
                    # This is because kernel launches are costly on CUDA and XLA.
                    state['step'] = (
                        torch.zeros((), dtype=torch.float, device=p.device)
                        if group['capturable'] or group['fused']
                        else torch.tensor(0.)
                    )
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if group['amsgrad']:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                if group['differentiable'] and state['step'].requires_grad:
                    raise RuntimeError('`requires_grad` is not supported for `step` in differentiable mode')

                # Foreach without capturable does not support a tensor lr
                if group['foreach'] and torch.is_tensor(group['lr']) and not group['capturable']:
                    raise RuntimeError('lr as a Tensor is not supported for capturable=False and foreach=True')

                state_steps.append(state['step'])


    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps)

            adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group['amsgrad'],
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                maximize=group['maximize'],
                foreach=group['foreach'],
                capturable=group['capturable'],
                differentiable=group['differentiable'],
                fused=group['fused'],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        return loss



Adam2.__doc__ = r"""Implements Adam algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \beta_1, \beta_2
                \text{ (betas)},\theta_0 \text{ (params)},f(\theta) \text{ (objective)}          \\
            &\hspace{13mm}      \lambda \text{ (weight decay)},  \: \textit{amsgrad},
                \:\textit{maximize}                                                              \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                v_0\leftarrow 0 \text{ (second moment)},\: \widehat{v_0}^{max}\leftarrow 0\\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\

            &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{10mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})         \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{5mm}\textbf{if} \: \lambda \neq 0                                           \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
            &\hspace{10mm}\widehat{v_t}^{max} \leftarrow \mathrm{max}(\widehat{v_t}^{max},
                \widehat{v_t})                                                                   \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}^{max}} + \epsilon \big)                                 \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `Adam: A Method for Stochastic Optimization`_.
    """ + fr"""
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, Tensor, optional): learning rate (default: 1e-3). A tensor LR
            is not yet supported for all our implementations. Please use a float
            LR if you are not also specifying fused=True or capturable=True.
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (bool, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        {_foreach_doc}
        {_maximize_doc}
        {_capturable_doc}
        {_differentiable_doc}
        {_fused_doc}
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ

    """


def adam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[Tensor],
         # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
         # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
         foreach: Optional[bool] = None,
         capturable: bool = False,
         differentiable: bool = False,
         fused: Optional[bool] = None,
         grad_scale: Optional[Tensor] = None,
         found_inf: Optional[Tensor] = None,
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: Union[float, Tensor],
         weight_decay: float,
         eps: float,
         maximize: bool):
    r"""Functional API that performs Adam algorithm computation.
    See :class:`~torch.optim.Adam` for details.
    """

    # Respect when the user inputs False/True for foreach or fused. We only want to change
    # the default when neither have been user-specified. Note that we default to foreach
    # and pass False to use_fused. This is not a mistake--we want to give the fused impl
    # bake-in time before making it the default, even if it is typically faster.
    if fused is None and foreach is None:
        _, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)
        # Do not flip on foreach for the unsupported case where lr is a Tensor and capturable=False.
        if foreach and isinstance(lr, Tensor) and not capturable:
            foreach = False
    if fused is None:
        fused = False
    if foreach is None:
        foreach = False

    # this check is slow during compilation, so we skip it
    # if it's strictly needed we can add this check back in dynamo
    if not torch._utils.is_compiling() and not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')
    if fused and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with fused optimizers")

    if fused and not torch.jit.is_scripting():
        func = _fused_adam
    elif foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adam
    else:
        func = _single_tensor_adam

    func(params,
         grads,
         exp_avgs,
         exp_avg_sqs,
         max_exp_avg_sqs,
         state_steps,
         amsgrad=amsgrad,
         beta1=beta1,
         beta2=beta2,
         lr=lr,
         weight_decay=weight_decay,
         eps=eps,
         maximize=maximize,
         capturable=capturable,
         differentiable=differentiable,
         grad_scale=grad_scale,
         found_inf=found_inf)


def _single_tensor_adam(params: List[Tensor],
                        grads: List[Tensor],
                        exp_avgs: List[Tensor],
                        exp_avg_sqs: List[Tensor],
                        max_exp_avg_sqs: List[Tensor],
                        state_steps: List[Tensor],
                        grad_scale: Optional[Tensor],
                        found_inf: Optional[Tensor],
                        *,
                        amsgrad: bool,
                        beta1: float,
                        beta2: float,
                        lr: Union[float, Tensor],
                        weight_decay: float,
                        eps: float,
                        maximize: bool,
                        capturable: bool,
                        differentiable: bool):

    assert grad_scale is None and found_inf is None

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if not torch._utils.is_compiling() and capturable:
            assert (
                (param.is_cuda and step_t.is_cuda) or (param.is_xla and step_t.is_xla)
            ), "If capturable=True, params and state_steps must be CUDA or XLA tensors."

        # update step
        step_t += 1

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            if amsgrad:
                max_exp_avg_sqs[i] = torch.view_as_real(max_exp_avg_sqs[i])
            param = torch.view_as_real(param)

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

        if capturable or differentiable:
            step = step_t

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()

            bias_correction2_sqrt = bias_correction2.sqrt()

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                if differentiable:
                    max_exp_avg_sq = max_exp_avg_sqs[i].clone()
                else:
                    max_exp_avg_sq = max_exp_avg_sqs[i]

                max_exp_avg_sqs[i].copy_(torch.maximum(max_exp_avg_sq, exp_avg_sq))

                # Uses the max. for normalizing running avg. of gradient
                # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                denom = (max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)
            else:
                denom = (exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)
            param.addcdiv_(exp_avg, denom)
            breakpoint()
        else:
            step = _get_value(step_t)

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            step_size = lr / bias_correction1

            bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])

                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
            #New==>
            #Need to construct G as in GramLayer
            V = param#.tril(diagonal=-1) #+ torch.diag(param.diag().exp())
            V = V #/ V.shape[1] ** 0.5  # Scale nicely for optimiser
            G = V @ V.t()  # Compute Gram matrix
            param -= step_size*(G @ (exp_avg/denom))#.tril()
            #param -= step_size * ( (exp_avg / denom))
            #<==
            #Old==>
            #param.addcdiv_(exp_avg, denom, value=-step_size)
            #<==

        # Lastly, switch back to complex view
        if amsgrad and torch.is_complex(params[i]):
            max_exp_avg_sqs[i] = torch.view_as_complex(max_exp_avg_sqs[i])


def _multi_tensor_adam(params: List[Tensor],
                       grads: List[Tensor],
                       exp_avgs: List[Tensor],
                       exp_avg_sqs: List[Tensor],
                       max_exp_avg_sqs: List[Tensor],
                       state_steps: List[Tensor],
                       grad_scale: Optional[Tensor],
                       found_inf: Optional[Tensor],
                       *,
                       amsgrad: bool,
                       beta1: float,
                       beta2: float,
                       lr: Union[float, Tensor],
                       weight_decay: float,
                       eps: float,
                       maximize: bool,
                       capturable: bool,
                       differentiable: bool):
    if len(params) == 0:
        return

    if isinstance(lr, Tensor) and not capturable:
        raise RuntimeError("lr as a Tensor is not supported for capturable=False and foreach=True")

    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    if not torch._utils.is_compiling() and capturable:
        assert all(p.is_cuda and step.is_cuda for p, step in zip(params, state_steps)), \
            "If capturable=True, params and state_steps must be CUDA tensors."

    assert grad_scale is None and found_inf is None

    assert not differentiable, "_foreach ops don't support autograd"

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps])
    for ((
        device_params,
        device_grads,
        device_exp_avgs,
        device_exp_avg_sqs,
        device_max_exp_avg_sqs,
        device_state_steps,
    ), _) in grouped_tensors.values():

        if maximize:
            device_grads = torch._foreach_neg(device_grads)

        # Handle complex parameters
        device_grads = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_grads]
        device_exp_avgs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_exp_avgs]
        device_exp_avg_sqs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_exp_avg_sqs]
        device_max_exp_avg_sqs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_max_exp_avg_sqs]
        device_params = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_params]

        # update steps
        torch._foreach_add_(device_state_steps, 1)

        if weight_decay != 0:
            # Re-use the intermediate memory (device_grads) already allocated for maximize
            if maximize:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = torch._foreach_add(device_grads, device_params, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        torch._foreach_lerp_(device_exp_avgs, device_grads, 1 - beta1)

        torch._foreach_mul_(device_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(device_exp_avg_sqs, device_grads, device_grads, 1 - beta2)

        # Delete the local intermediate since it won't be used anymore to save on peak memory
        del device_grads

        if capturable:
            bias_correction1 = torch._foreach_pow(beta1, device_state_steps)
            bias_correction2 = torch._foreach_pow(beta2, device_state_steps)
            # foreach_sub doesn't allow a scalar as the first arg
            torch._foreach_sub_(bias_correction1, 1)
            torch._foreach_sub_(bias_correction2, 1)
            # we do not negate bias_correction1 as it'll need to be negated later anyway
            torch._foreach_neg_(bias_correction2)

            # foreach_div doesn't allow a scalar as the first arg
            torch._foreach_div_(bias_correction1, lr)
            torch._foreach_reciprocal_(bias_correction1)

            torch._foreach_sqrt_(bias_correction2)

            # Re-assign for clarity as we maintain minimal intermediates: we'll have
            # step_size = - lr / (1 - beta1 ^ t) where t = num_steps
            # bias_correction2_sqrt = sqrt(1 - beta2 ^ t)
            step_size = bias_correction1
            bias_correction2_sqrt = bias_correction2

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)  # type: ignore[assignment]

                # Set intermediate to the max. for normalizing running avg. of gradient when amsgrad
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)
            torch._foreach_div_(exp_avg_sq_sqrt, step_size)

            # at this point, exp_avg_sq_sqrt = - (1 - beta^t) * [sqrt(exp_avg_sq / (1 - beta2^t)) + eps] / lr
            torch._foreach_addcdiv_(device_params, device_exp_avgs, exp_avg_sq_sqrt)
        else:
            bias_correction1 = [1 - beta1 ** _get_value(step) for step in device_state_steps]
            bias_correction2 = [1 - beta2 ** _get_value(step) for step in device_state_steps]

            step_size = _stack_if_compiling([(lr / bc) * -1 for bc in bias_correction1])

            bias_correction2_sqrt = [_dispatch_sqrt(bc) for bc in bias_correction2]

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)

                # Use the max. for normalizing running avg. of gradient
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)
            torch._foreach_addcdiv_(device_params, device_exp_avgs, exp_avg_sq_sqrt, step_size)


def _fused_adam(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,  # Needed for consistency.
    differentiable: bool,
) -> None:
    if not params:
        return
    if differentiable:
        raise RuntimeError("Adam with fused=True does not support differentiable=True")

    grad_scale_dict = {grad_scale.device: grad_scale} if grad_scale is not None else None
    found_inf_dict = {found_inf.device: found_inf} if found_inf is not None else None

    # We only shuffle around the lr when it is a Tensor and on CUDA, otherwise, we prefer
    # treating it as a scalar.
    lr_dict = {lr.device: lr} if isinstance(lr, Tensor) and str(lr.device) != "cpu" else None

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps])
    for (device, _), ((device_params,
                       device_grads,
                       device_exp_avgs,
                       device_exp_avg_sqs,
                       device_max_exp_avg_sqs,
                       device_state_steps,), _) in grouped_tensors.items():
        device_grad_scale, device_found_inf = None, None
        if grad_scale is not None:
            if device not in grad_scale_dict:
                grad_scale_dict[device] = grad_scale.to(device, non_blocking=True)
            device_grad_scale = grad_scale_dict[device]
        if found_inf is not None:
            if found_inf not in found_inf_dict:
                found_inf_dict[device] = found_inf.to(device, non_blocking=True)
            device_found_inf = found_inf_dict[device]
        if lr_dict is not None and device not in lr_dict:
            lr_dict[device] = lr.to(device=device, non_blocking=True)
            lr = lr_dict[device]
        torch._foreach_add_(device_state_steps, 1)
        torch._fused_adam_(
            device_params,
            device_grads,
            device_exp_avgs,
            device_exp_avg_sqs,
            device_max_exp_avg_sqs,
            device_state_steps,
            amsgrad=amsgrad,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eps=eps,
            maximize=maximize,
            grad_scale=device_grad_scale,
            found_inf=device_found_inf,
        )
        if device_found_inf is not None:
            torch._foreach_sub_(device_state_steps, [device_found_inf] * len(device_state_steps))


#%% UCI Data
# uci = UCI("wine", 2, dtype=getattr(torch,"float64"), device=device)
# x_train = uci.X_train
# y_train = uci.Y_train
# x_test = uci.X_test_norm
# y_test = uci.Y_test_unnorm
# no_of_data_points = x_train.shape[0]
# input_features = x_train.shape[1]


#%% Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_folder_path', type=str, default="./data/")
parser.add_argument('--datasets_list', type=str, default="data/datasets.txt")
parser.add_argument('--device', type=str, nargs='?', default='cpu', choices=['cpu', 'cuda'])
parser.add_argument('--dtype', type=str, nargs='?', default='float32', choices=['float32', 'float64'])
parser.add_argument("--dof", type=float, default=0)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--save_name", type=str)
parser.add_argument("--s2_learned", type=bool, default=False)
parser.add_argument("--likelihood", type=str, default="categorical", choices=["gaussian","categorical"])
args = parser.parse_args()


#%% Set PyTorch Device and Dtype
device = torch.device(args.device)
dtype = getattr(torch, args.dtype)
torch.set_default_dtype(dtype)
if args.dtype == "float64":
    torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(0)
batch_size=256
#%% Toy Data Classification
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
import matplotlib.pyplot as plt

def non_linear_function(x):
    return np.sin(5 * x) + np.cos(3 * x)

no_of_data_points = 1000
no_of_inducing_points = 100
X = np.linspace(-1, 1, no_of_data_points).reshape(-1, 1)
y = np.where(non_linear_function(X) > 0, 1, 0).reshape(-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

def plot_data():
    # Plot the data
    plt.figure(figsize=(10, 6))
    # Train data in blue
    plt.scatter(X_train, y_train, color='blue', label='Train Data')

    # Test data in red
    plt.scatter(X_test, y_test, color='red', label='Test Data')

    # Plot the non-linear function
    X_range = np.linspace(-1, 1, 1000)
    plt.plot(X_range, non_linear_function(X_range), color='green', label='Non-linear Function')

    plt.title('plot of train/test data')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()

plot_data()

shuffleperm = torch.randperm(X_train_tensor.shape[0])
x_full = X_train_tensor[shuffleperm]
x_ind = x_full[:no_of_inducing_points]
x_train = x_full[no_of_inducing_points:]

y_full = y_train_tensor[shuffleperm]
y_ind = y_full[:no_of_inducing_points]
y_train = y_full[no_of_inducing_points:]

x_test = X_test_tensor
y_test = y_test_tensor

input_features = x_train.shape[1]
n_classes = 2

# n_classes = 2
# no_of_inducing_points = 50
# no_of_data_points =1000
# xs_0 = torch.linspace(-2,2,no_of_data_points//2) + torch.randn(no_of_data_points//2)*0.1
# xs_1 = torch.linspace(-2,2,no_of_data_points//2) + torch.randn(no_of_data_points//2)*0.1
# xs_0 = torch.stack([xs_0, torch.sin(xs_0*3)*6+torch.randn(no_of_data_points//2)], dim=1)
# xs_1 = torch.stack([xs_1, torch.sin(xs_1*3)*6+torch.randn(no_of_data_points//2)+5], dim=1)
# toy_classifier_xs_class_0 = xs_0
# toy_classifier_xs_class_1 = xs_1
#
# shuffle_perm = torch.randperm(no_of_data_points)
# toy_classif_xs = torch.cat([toy_classifier_xs_class_0, toy_classifier_xs_class_1])[shuffle_perm]
# toy_classif_ys = torch.cat([torch.zeros(no_of_data_points//2), torch.ones(no_of_data_points//2)])[shuffle_perm]#.resize(no_of_data_points,1)
#
# #plt.scatter(toy_classif_xs[:,0], toy_classif_xs[:,1],c=toy_classif_ys)
#
# xs_0_test = torch.linspace(-2,2,no_of_data_points//4) + torch.randn(no_of_data_points//4)*0.1
# xs_1_test = torch.linspace(-2,2,no_of_data_points//4) + torch.randn(no_of_data_points//4)*0.1
# xs_0_test = torch.stack([xs_0_test, torch.sin(xs_0_test*3)*6+torch.randn(no_of_data_points//4)], dim=1)
# xs_1_test = torch.stack([xs_1_test, torch.sin(xs_1_test*3)*6+torch.randn(no_of_data_points//4)+5], dim=1)
# toy_classifier_xs_class_0_test = xs_0_test
# toy_classifier_xs_class_1_test = xs_1_test
# shuffle_perm2 = torch.randperm(no_of_data_points//2)
# toy_classif_xs_test = torch.cat([toy_classifier_xs_class_0_test, toy_classifier_xs_class_1_test])[shuffle_perm2]
# toy_classif_ys_test = torch.cat([torch.zeros(no_of_data_points//4), torch.ones(no_of_data_points//4)])[shuffle_perm2]#.resize(no_of_data_points//2,1)
#
# x_full =  (toy_classif_xs - toy_classif_xs.mean(dim=0)) / toy_classif_xs.std(dim=0)
# x_ind = x_full[:no_of_inducing_points,:]
# x_train = x_full[no_of_inducing_points:,:]
#
#
# y_full = toy_classif_ys
# y_ind = y_full[:no_of_inducing_points]
# y_train = y_full[no_of_inducing_points:]
#
# x_test = (toy_classif_xs_test - toy_classif_xs.mean(dim=0)) / toy_classif_xs.std(dim=0)
# y_test = toy_classif_ys_test
#
# input_features = x_train.shape[1]
#
# torch.manual_seed(args.seed)

#%% No_DKM New Output
kwargs = {}
layer_kwargs = {**kwargs, 'sqrt' : dkm.sym_sqrt, 'MAP' : False}
#sum_kwargs = {**kwargs, 'noise_var_learned' : args.s2_learned, 'likelihood' : args.likelihood}
sum_kwargs = {**kwargs}

x_ind_copy = torch.clone(x_ind)

model = nn.Sequential(dkm.Input(x_ind, learned=True),
                      dkm.F2G(),
                      dkm.ReluKernel(),
                      dkm.GramLayer(x_ind.shape[0], args.dof),
                      dkm.ReluKernel(),
                      dkm.GramLayer(x_ind.shape[0], args.dof),
                      dkm.ReluKernel(),
                      dkm.GramLayer(x_ind.shape[0], args.dof),
                      dkm.ReluKernel(),
                      # dkm.Output(x_ind.shape[0], 2, mc_samples=100, init_mu=torch.nn.functional.one_hot(y_ind.long(), num_classes=2).to(dtype=dtype), **sum_kwargs),
                      dkm.Output(x_ind.shape[0], 2, mc_samples=100,  **sum_kwargs),
                      )

print(f"Model Architecture: {model}")
print(f"s2 learned: {args.s2_learned}")
print(f"Random Seed: {args.seed}")
print(f"Initial learning rate: {args.lr}")
print(f"dof: {args.dof}")

# model[-1]._mu.data = torch.nn.functional.one_hot(y_ind.long(), num_classes=2).double()
# y_ind_old = torch.clone(y_ind)
# y_ind = model[-1].mu.data.argmax(dim=1).clone()

model = model.to(device=device)

from torch.utils import checkpoint

print("CUDA Checks")
print(f"Model: {next(model.parameters()).is_cuda}")
x_train = x_train.to(device)
y_train = y_train.to(device=device, dtype=dtype)
x_test = x_test.to(device=device, dtype=dtype)
y_test = y_test.to(device=device, dtype=dtype)
print(f"x_train: {x_train.is_cuda}")
print(f"y_train: {y_train.is_cuda}")
print(f"x_test: {x_test.is_cuda}")
print(f"y_test: {y_test.is_cuda}")

times = []
lls = []
rmses = []
objs = []
# print(f"x_train shape: {x_train.shape}")
# print(f"y_train shape: {y_train.shape}")


trainset = torch.utils.data.TensorDataset(x_train, y_train)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

if args.save_name:
    torch.save(model, "temp_prior_model_" + args.save_name)
previous_obj = -float("inf")
model.eval()
model(next(iter(trainloader))[0].to(device, dtype))
#opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999), eps=1e-5)
#opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.8,0.9))

# Extract parameters V from GramLayer instances
V_params = [param for module in model.modules() if isinstance(module, dkm.GramLayer) for param in module.parameters()]
#Use Adam for V
#opt_V = Adam2(V_params, lr=args.lr, betas=(0.8,0.9))
opt_V = torch.optim.Adam(V_params, lr=args.lr, betas=(0.8,0.9))
#Use Adam for the other parameters
opt_other = torch.optim.Adam([param for module in next(model.modules()) if not isinstance(module, dkm.GramLayer) for param in module.parameters()], lr=args.lr, betas=(0.8,0.9))

previous_lr_V = opt_V.param_groups[0]['lr']
previous_lr_other = opt_other.param_groups[0]['lr']

# Define milestones and learning rates
milestones = [40, 80]  # Epochs where learning rate will change
gammas = [0.1, 0.1]  # Factors by which the learning rate will be multiplied
#gammas=[1,1]
# Create the scheduler
scheduler_V = MultiStepLR(opt_V, milestones=milestones, gamma=gammas[0])  # Assuming gamma is the same for each step for simplicity
scheduler_other = MultiStepLR(opt_other, milestones=milestones, gamma=gammas[0])  # Assuming gamma is the same for each step for simplicity

def print_grad_norms(model):
    print("Gradient norms:")
    for name, param in model.named_parameters():
        if 'V' in name:
            grad_norm = param.grad.norm().item()
            print(f"{name}: {grad_norm}")

gradient_norms = {}
def save_grad_norms(model):
    for name, param in model.named_parameters():
        if 'V' in name:
            grad_norm = param.grad.norm().item()
            if name in gradient_norms:
                gradient_norms[name].append(grad_norm)
            else:
                gradient_norms[name] = [grad_norm]



for i in range(10):
    # if i == 30:
    #     opt.param_groups[0]['lr'] = 1e-1
    #     scheduler = scheduler_plateau
    model.train()
    start = timer()
    train_obj = 0
    train_lls = []
    batch = 0
    for X, Y in trainloader:
        batch += 1
        model.train()
        X, Y = X.to(device, dtype), Y.to(device, dtype)
        pred = model(X)
        if args.likelihood == "gaussian":
            obj = dkm.gaussian_expectedloglikelihood(pred, torch.nn.functional.one_hot(Y.long(), num_classes=2)) + dkm.norm_kl_reg(model, no_of_data_points-no_of_inducing_points)
        elif args.likelihood == "categorical":
            obj = dkm.categorical_expectedloglikelihood(pred,Y.long()) + dkm.norm_kl_reg(model, no_of_data_points-no_of_inducing_points)
        opt_V.zero_grad()
        opt_other.zero_grad()
        (-obj).backward()
        #print_grad_norms(model)
        #save_grad_norms(model)
        opt_V.step()
        opt_other.step()

        train_obj += obj.item()*X.size(0)

        #do a breakpoint() if obj is nan
        if math.isnan(obj.item()):
            print("obj is nan")
            breakpoint()

    end = timer()

    mean_train_obj = train_obj / len(trainloader.sampler)
    objs.append(mean_train_obj)


    if args.save_name:
        torch.save(model, "temp_trained_model_"+args.save_name)

    if i%1==0:
        print((i, mean_train_obj, end-start), flush=True)

        # retrieve the maximum memory usage of the CPU
        max_cpu_mem_usage = psutil.Process().memory_info().rss

        # convert to GB
        max_cpu_mem_usage_gb = max_cpu_mem_usage / 1024 ** 3

        # print the result
        print(f"Maximum CPU memory usage: {max_cpu_mem_usage_gb:.2f} GB", flush=True)

    previous_obj=mean_train_obj
    # if i >= 30:
    #     scheduler.step(mean_train_obj)
    # else:
    #     opt.param_groups[0]['lr'] = lr_lambda(i)

    scheduler_V.step()
    scheduler_other.step()

    # if i == 40:
    #     opt = NGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0, update_period=4)
    #     scheduler = MultiStepLR(opt, milestones=milestones[1:], gamma=[1])  # Assuming gamma is the same for each step for simplicity
    model.eval()
    if args.likelihood == "gaussian":
        test_preds = torch.argmax(dkm.gaussian_prediction(model(x_test)).loc, dim=1)
    else:
        test_preds = torch.argmax(dkm.categorical_prediction(model(x_test)).probs, dim=1)
    test_accuracy = (test_preds == y_test.long()).sum() / y_test.numel()
    print(f"Test Accuracy: {test_accuracy}",flush=True)

    if opt_V.param_groups[0]['lr'] != previous_lr_V:
        print(f"V LEARNING RATE HAS CHANGED TO {opt_V.param_groups[0]['lr']}")
        previous_lr_V = opt_V.param_groups[0]['lr']

    if opt_other.param_groups[0]['lr'] != previous_lr_other:
        print(f"OTHER LEARNING RATE HAS CHANGED TO {opt_other.param_groups[0]['lr']}")
        previous_lr_other = opt_other.param_groups[0]['lr']

#%% Manual Classification
model.eval()
if args.likelihood == "gaussian":
    test_preds = torch.nn.functional.one_hot(torch.argmax(dkm.gaussian_prediction(model(x_test)).loc, dim=1), num_classes=y_test.unique().numel())
else:
    test_preds = torch.argmax(dkm.categorical_prediction(model(x_test)).probs, dim=1)
test_accuracy = (test_preds == y_test.long()).sum() / y_test.numel()
print(f"Test Accuracy: {test_accuracy}",flush=True)

def plot_data():
    # Plot the data
    plt.figure(figsize=(10, 6))
    # Train data in blue
    plt.scatter(x_train, y_train, color='blue', label='Train Data')

    # Test data in red
    plt.scatter(x_test, y_test, color='red', label='Test Data')

    # Plot the non-linear function
    X_range = torch.linspace(-1, 1, 1000)
    model_eval = model(X_range[:,None]).mean(0)
    plt.plot(X_range.detach().numpy(), (model_eval[:,1]/model_eval[:,0]).detach().numpy(), color='green', label='Non-linear Function')

    plt.title('plot of train/test data')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()
plot_data()
