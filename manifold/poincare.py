import torch
import torch.linalg as alg
from torch import Tensor
from base import Manifold

class PoincareBall(Manifold):
    def __init__(self):
        super(PoincareBall, self).__init__()
        self.name = 'PoincareBall'
        self.min_norm = 1e-15
        self.epsilon = 1e-5
        

    def proj(self,vectors: Tensor) -> Tensor:
        '''Projection onto the PoincareBall manifold.
        Clip vectors to have a norm of less than one.


        Parameters
        ----------
        vectors : Tensor-like
            Can be a 1-D or 2-D (in this case the norm of each row is checked)

        Returns
        -------
        Tensor-like
            Tensor with norms clipped below 1.
        '''
        thresh = 1.0 - self.epsilon
        one_d = len(vectors.shape) == 1
        if one_d:
            norm = alg.norm(vectors)
            if norm < thresh:
                return vectors
            else:
                return thresh * vectors / norm
        else:
            norms = alg.norm(vectors,axis=1)
            if (norms < thresh).all():
                return vectors
            else:
                vectors_ = torch.clone(vectors)
                vectors_[norms >= thresh] = vectors_[norms >= thresh] *  \
                    (thresh / norms[norms >= thresh])[:,None]
                vectors = vectors_
                return vectors


    def _lambda_x(self,x: Tensor) -> Tensor:
        '''Return the conformal factor
        '''
        norm_x = alg.norm(x,dim=-1)
        return 2. / (1. - norm_x ** 2.).clamp_min(self.min_norm)

                

    def distance(self,x: Tensor,y: Tensor) -> Tensor: 
        '''Compute the geodesic distance between two points.

        Parameters
        ----------
        point_a : Tensor-like, shape=[...,2]
            First point on the disk
        point_b : Tensor-like, shape=[...,2]
            Second point on the disk

        Returns
        -------
        dist : Tensor-like, shape=[...]
            Geodesic distance between points
        '''
        point_a = self._clip_vectors(point_a)
        point_b = self._clip_vectors(point_b)

        sq_norm_x = alg.norm(x, axis=-1) ** 2.
        sq_norm_y = alg.norm(y, axis=-1) ** 2.
        sq_norm_xy = alg.norm(x - y, axis=-1) ** 2.

        cosh_angle = 1 + 2 * sq_norm_xy / (1 - sq_norm_x) / (1 - sq_norm_y) 
        dist = torch.acosh(cosh_angle.clamp_min(1. + 1e-8))
        return dist


    def egrad2rgrad(self, p: Tensor, dp: Tensor) -> Tensor:
        '''Translate Euclidean gradient to Riemannian gradient
        on tangent space
        '''
        lambda_p = self._lambda_x(p)
        dp /= lambda_p.pow(2.)
        return dp


    def expmap(self, vector: Tensor, base_point=torch.zeros_like(vector)
            ) -> Tensor:
        '''Compute the Riemann exponential of a tangent vector.

        Parameters
        ----------
        vector : Tensor-like, shape=[...]
            vector
        base_point : Tensor-like, shape=[...]
            base point, by default the center (0,0)

        Returns
        -------
        exp : Tensor-like, shape=[...]
            Point on the hypersphere equal to the Riemannian exponential
            of tangent vector at the base point.
        '''
        factor = self._lambda_x(base_point)
        norm_vector = alg.norm(vector, dim=-1)
        tn = torch.tanh(factor * norm_vector / 2.) * vector / norm_vector
        exp = self.mobius_add(base_point,tn)
        return exp


    def logmap(self, vector: Tensor, base_point=torch.zeros_like(vector)
            ) -> Tensor:
        '''Compute the logarithmic map on the manifold.

        Parameters
        ----------
        vector : Tensor-like, shape=[...]
            vector
        base_point : Tensor-like, shape=[...]
            base point, by default the center (0,0)

        Returns
        -------
        log : Tensor-like, shape=[]
            Logarithmic of point at the base point.
        '''
        factor = self.conformal_factor(base_point)
        mobius_sum = self.mobius_add(-base_point,vector)
        norm_sum = alg.norm(mobius_sum)
        norm_sum = self._clip_vectors(norm_sum)
        log = th.arctanh(norm_sum) * mobius_sum / norm_sum
        log *= (2. / factor)
        return log


    def mobius_add(self, v1: Tensor, v2: Tensor) -> Tensor:
        '''Addition between two vectors in the manifold.
        This is not commutative.
        Also the difference is taken as a sum with the negative vector.

        Parameters
        ----------
        v1 : Tensor-like, shape[...,2]
            left vector
        v2 : Tensor-like, shape[...,2]
            right vector

        Returns 
        -------
        addition : Tensor-like, shape=[...]
            mobius addition of v1 + v2
        '''
        norm_v1 = alg.norm(v1, dim=-1, keepdim=True) ** 2.
        norm_v2 = alg.norm(v2, dim=-1, keepdim=True) ** 2.
        v1v2 = (v1 * v2).sum(dim=-1, keepdim=True)
        
        numerator = (1. + 2.*v1v2 + norm_v2) * v1 + (1. - norm_v1)*v2
        denominator = 1. + 2.*v1v2 + norm_v1*norm_v2
        addition = numerator/denominator
        return addition


    def mobius_matvec(self, mat: Tensor, vec: Tensor) -> Tensor:
        '''Multiplication between a matrix and a vector in the  
        Poincare ball.

        Parameters
        ----------
        mat : Tensor-like, shape=[m,n]
            matrix, typically the weight matrix
        vec : Tensor-like, shape=[n,1]
            vector

        Returns
        -------
        mobius_mult : Tensor-like, shape[m,1]
            new vector
        '''
        norm_vector = alg.norm(vec, dim=-1)
        mult = torch.matmul(mat,vec)
        norm_mult = alg.norm(mult, dim=-1)
        mobius_mult = torch.tanh(norm_mult / norm_vector * \
                torch.arctanh(norm_vector) ) * mult / norm_mult
        return mobius_mult


    def inner(self, x: Tensor, u: Tensor, v=None) -> Tensor:
        '''Inner product

        Parameters:
            x: Tensor-like
                tensor
            u: Tensor-like
                tensor
            v: Tensor-like
                default None

        Returns:
            inn: Tensor-like
                inner product
        '''
        if v is None:
            v = u
        lambda_x = self._lambda_x(x)
        inn = lambda_x ** 2. * (u * v).sum(dim=-1, keepdim=keepdim)
        

    def ptransp(self, vec: Tensor, point_b: Tensor,
            point_a=torch.zeros_like(point_b)) -> Tensor:
        '''Parallel transport of a vector from a starting point to
        an end point.

        Parameters
        ----------
        vec : Tensor-like, shape[...,2]
            vector to be transported
        point_b : Tensor-like, shape[...,2]
            end point
        point_a : Tensor-like, shape[...,2]
            start point - by default the center of the disk

        Returns
        -------
        transported_vec : Tensor-like, shape[...,2]
            the new vector after transported
        '''
        exp = self.expmap(vec, point_a)
        mobius_sum = self.mobius_add(point_b, exp)
        transported_vec = self.logmap(mobius_sum, point_b)
        return transported_vec


    def bias_translation(self, vec: Tensor, bias: Tensor) -> Tensor:
        '''Translation of a vector by a bias.

        Parameters
        ----------
        vec : Tensor-like, shape=[m,1]
            vector to be translated
        bias : Tensor-like, shape=[m,1]
            vector to apply

        Returns
        -------
        translated_vec : Tensor-like, shape=[m,1]
            translated vector
        '''
        factor_vector = self._lambda_x(vec)
        factor_bias = self._lambda_x(bias)
        factor = factor_vector / factor_bias
        log_vector = self.logmap(bias)
        translated_vec = self.expmap(factor * log_vector, vector)
        return translated_vec
