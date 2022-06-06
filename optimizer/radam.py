import torch.optim
from manifold.poincare import PoincareBall
from manifold import ManifoldParameter
from .mixin import OptimMixin

_default_manifold = PoincareBall()


class RiemannianAdam(OptimMixin, torch.optim.Adam):
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        with torch.no_grad():
            for group in self.param_groups:
                if 'step' not in group:
                    group['step'] = 0
                betas = group['betas']
                weight_decay = group['weight_decay']
                eps = group['eps']
                learning_rate = group['lr']
                amsgrad = group['amsgrad']
                group['step'] += 1
                for point in group['params']:
                    grad = point.grad
                    if grad is None:
                        continue

                    manifold = _default_manifold
                    if grad.is_sparse:
                        raise RuntimeError(
                            'RiemannianAdam does not support \
                            sparse gradients, use SparseRiemannianAdam'
                        )

                    state = self.state[point]

                    # State initialization
                    if len(state) == 0:
                        # Exponential moving average of grad values
                        state['exp_avg'] = torch.zeros_like(point)
                        # Exponential moving average of sq grad values
                        state['exp_avg_sq'] = torch.zeros_like(point)
                        if amsgrad:
                            # Maintains max of all exp. moving avg.
                            # of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(point)
                    # Make local variable for easy access
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    # Actual step
                    #print(f'point: {point}')
                    #print(f'grad: {grad}')
                    #print(f'betas: {betas}')
                    grad.add_(point, alpha=weight_decay)
                    grad = manifold.egrad2rgrad(point, grad)
                    inner = manifold.inner(point, grad)
                    
                    exp_avg.mul_(betas[0]).add_(grad, alpha=1-betas[0])
                    exp_avg_sq.mul_(betas[1]).add_(inner, alpha=1-betas[1])

                    bias_correction1 = 1 - betas[0] ** group['step']
                    bias_correction2 = 1 - betas[1] ** group['step']
                    if amsgrad:
                        max_exp_avg_sq = state['max_exp_avg_sq']
                        # Maintains the maximum of all 2nd moment running
                        # avg. till now
                        torch.max(max_exp_avg_sq, exp_avg_sq,
                            out=max_exp_avg_sq)
                        # Use the max. for normalising running avg. 
                        # of gradient
                        denom = max_exp_avg_sq.div(bias_correction2).sqrt_()
                    else:
                        denom = exp_avg_sq.div(bias_correction2).sqrt_()
                    # copy the state, we need it for retraction
                    # get the direction for ascend
                    direction = exp_avg.div(bias_correction1) /\
                        denom.add_(eps)
                    '''
                    # transport the exponential averaging to the new point
                    new_point, exp_avg_new = manifold.retr_transp(
                        point, -learning_rate *direction, exp_avg
                    )
                    '''
                    step_size = learning_rate * bias_correction2 ** 0.5 /\
                        bias_correction1
                    
                    new_point = manifold.proj(manifold.expmap(
                        - step_size * direction, point))
                    exp_avg_new = manifold.ptransp(exp_avg, new_point,
                        point)

                    # use copy only for user facing point
                    point.copy_(new_point)
                    exp_avg.copy_(exp_avg_new)
                    # use copy only for user facing point

                
                if (
                    group['stabilize'] is not None 
                    and group['step'] % group['stabilize'] == 0
                ):
                    self.stabilize_group(group)
                '''
                if self._stabilize is not None:
                    self.stabilize_group(group)
                '''
        return loss


    @torch.no_grad()
    def stabilize_group(self,group):
        for p in group['params']:
            if not isinstance(p, ManifoldParameter):
                continue
            state = self.state[p]
            if not state: # due to None grads
                continue
            manifold = _default_manifold
            exp_avg = state['exp_avg']
            ####### need to correct this as I don't know what projtan is
            p.copy_(manifold.proj(p), 0.9)
            exp_avg.copy_(manifold.proj_tan(p, exp_avg))

