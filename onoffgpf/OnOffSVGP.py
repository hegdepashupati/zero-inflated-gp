from __future__ import absolute_import
import tensorflow as tf
import numpy as np
import pickle
from gpflow.param import Param
from gpflow.model import Model
from gpflow.mean_functions import Zero
from gpflow import transforms, conditionals, kullback_leiblers
from gpflow.param import AutoFlow, DataHolder
from gpflow._settings import settings
from gpflow.minibatch import MinibatchData
from math import pi
import time

float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64

class OnOffSVGP(Model):
    """
    - X is a data matrix, size N x D
    - Y is a data matrix, size N x 1
    - kernf,kernf are gpflow objects for function & gamma kernels
    - Zf,Zg are matrices of inducing point locations, size Mf x D and Mg x D
    """

    def __init__(self, X, Y, kernf, kerng, likelihood, Zf, Zg, mean_function=None, minibatch_size=None, 
                 name='model'):
        Model.__init__(self, name)
        self.mean_function = mean_function or Zero()
        self.kernf = kernf
        self.kerng = kerng
        self.likelihood = likelihood
        self.whiten = False
        self.q_diag = True

        # save initial attributes for future plotting purpose
        Xtrain = DataHolder(X)
        Ytrain = DataHolder(Y)
        self.Xtrain, self.Ytrain = Xtrain, Ytrain

        # sort out the X, Y into MiniBatch objects.
        if minibatch_size is None:
            minibatch_size = X.shape[0]
        self.num_data = X.shape[0]
        self.num_latent = Y.shape[1]  # num_latent will be 1
        self.X = MinibatchData(X, minibatch_size, np.random.RandomState(0))
        self.Y = MinibatchData(Y, minibatch_size, np.random.RandomState(0))

        # Add variational paramters
        self.Zf = Param(Zf)
        self.Zg = Param(Zg)
        self.num_inducing_f = Zf.shape[0]
        self.num_inducing_g = Zg.shape[0]

        # init variational parameters
        self.u_fm = Param(np.random.randn(self.num_inducing_f, self.num_latent) * 0.01)
        self.u_gm = Param(np.random.randn(self.num_inducing_g, self.num_latent) * 0.01)

        if self.q_diag:
            self.u_fs_sqrt = Param(np.ones((self.num_inducing_f, self.num_latent)),
                                transforms.positive)
            self.u_gs_sqrt = Param(np.ones((self.num_inducing_g, self.num_latent)),
                                transforms.positive)
        else:
            u_fs_sqrt = np.array([np.eye(self.num_inducing_f)
                                  for _ in range(self.num_latent)]).swapaxes(0, 2)
            self.u_fs_sqrt = Param(u_fs_sqrt, transforms.LowerTriangular(u_fs_sqrt.shape[2]))

            u_gs_sqrt = np.array([np.eye(self.num_inducing_g)
                                  for _ in range(self.num_latent)]).swapaxes(0, 2)
            self.u_gs_sqrt = Param(u_gs_sqrt, transforms.LowerTriangular(u_gs_sqrt.shape[2]))

    def build_prior_KL(self):
        # whitening of priors can be implemented here
        """
        This gives KL divergence between inducing points priors and approximated posteriors

        KL(q(u_g)||p(u_g)) + KL(q(u_f)||p(u_f))

        q(u_f) = N(u_f|u_fm,u_fs)
        p(u_f) = N(u_f|0,Kfmm)

        q(u_g) = N(u_g|u_gm,u_gs)
        p(u_g) = N(u_g|0,Kgmm)

        """

        if self.whiten:
            if self.q_diag:
                KL = kullback_leiblers.gauss_kl_white_diag(self.u_fm, self.u_fs_sqrt) + \
                     kullback_leiblers.gauss_kl_white_diag(self.u_gm, self.u_gs_sqrt)
            else:
                KL = kullback_leiblers.gauss_kl_white(self.u_fm, self.u_fs_sqrt) + \
                     kullback_leiblers.gauss_kl_white(self.u_gm, self.u_gs_sqrt)
        else:
            Kfmm = self.kernf.K(self.Zf) + tf.eye(self.num_inducing_f, dtype=float_type) * settings.numerics.jitter_level
            Kgmm = self.kerng.K(self.Zg) + tf.eye(self.num_inducing_g, dtype=float_type) * settings.numerics.jitter_level

            if self.q_diag:
                KL = kullback_leiblers.gauss_kl_diag(self.u_fm, self.u_fs_sqrt, Kfmm) + \
                     kullback_leiblers.gauss_kl_diag(self.u_gm, self.u_gs_sqrt, Kgmm)
            else:
                KL = kullback_leiblers.gauss_kl(self.u_fm, self.u_fs_sqrt, Kfmm) + \
                     kullback_leiblers.gauss_kl(self.u_gm, self.u_gs_sqrt, Kgmm)
        return KL

    def build_likelihood(self):
        # get prior KL
        KL = self.build_prior_KL()

        # get augmented functions
        gfmean, gfvar, gfmeanu, _, _, _, _, _, _ = self.build_predict(self.X)

        # compute likelihood
        # this should be added to GPFlow likelihood script
        var_exp = self.likelihood.variational_expectations(gfmean, gfvar, gfmeanu, self.Y)

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, settings.dtypes.float_type) / \
                tf.cast(tf.shape(self.X)[0], settings.dtypes.float_type)

        return tf.reduce_sum(var_exp) * scale - KL

    def build_predict(self, Xnew):
        '''
        This method builds latent variables - f, g, \Phi(g) from inducing distributions
        '''
        # Get conditionals
        # returns mean, variance for marginal distributions q(f) and q(g)
        # q(f) = \int q(f|u_f) q(u_f) du_f
        # q(f) = N(f|A*u_fm,Kfnn + A(u_fs - Kfmm)t(A))  A = Kfnm*inv(Kfmm)
        fmean, fvar = conditionals.conditional(Xnew, self.Zf, self.kernf, self.u_fm,
                                               full_cov=False, q_sqrt=self.u_fs_sqrt, whiten=self.whiten)
        fmean = fmean + self.mean_function(Xnew)

        gmean, gvar = conditionals.conditional(Xnew, self.Zg, self.kerng, self.u_gm,
                                               full_cov=False, q_sqrt=self.u_gs_sqrt, whiten=self.whiten)

        # probit transformed expectations for  gamma
        ephi_g, ephi2_g, evar_phi_g = self.ProbitExpectations(gmean, gvar)

        # compute augmented f
        # from above computations we have
        # p(f)   = N(f| A*u_fm, Kfnn + A(u_fs - Kfmm)t(A))  A = Kfnm*inv(Kfmm)
        # p(f|g) = N(f| diag(ephi_g)* A*u_fm, diag(evar_phi_g)) * (Kfnn + A(u_fs - Kfmm)t(A)))
        gfmean = tf.multiply(ephi_g, fmean)
        gfvar = tf.multiply(ephi2_g, fvar)
        gfmeanu = tf.multiply(evar_phi_g, tf.square(fmean))

        # return mean and variancevectors of following in order -
        # augmented f, f, g, \Phi(g)
        return gfmean, gfvar, gfmeanu, fmean, fvar, gmean, gvar, ephi_g, evar_phi_g

    def savemodel(self, fname=None):
        if fname is None:
            pickle.dump(self,open("pm_"+time.strftime("%Y%m%d-%H%M")+"_"+str(self.name) + ".pickle", "wb"))
        else:
            pickle.dump(self,open(fname, "wb"))

    @AutoFlow((float_type, [None, None]))
    def predict_onoffgp(self, Xnew):
        return self.build_predict(Xnew)

    @AutoFlow()
    def compute_prior_KL(self):
        return self.build_prior_KL()

    @staticmethod
    def ProbitExpectations(gmean, gvar):
        """
            Compute expectation of probit transformed gaussian variables
            pgmean = \int Phi(g) q(g) dg
            pgmeansq = \int Phi^2(g) q(g) dg
            pgvar = \int Var(Phi(g)) q(g) dg
        """

        def normcdf(x):
            return 0.5 * (1.0 + tf.erf(x / np.sqrt(2.0))) * (1. - 2.e-3) + 1.e-3

        def owent(h, a):
            """
                compute Owen's T approximation (lower bound)
                T(h,a) >= (arctan(a)/2PI)*exp(-h^2(a^2+1)/2)
            """
            h = tf.abs(h)
            term1 = tf.atan(a) / (2 * pi)
            term2 = tf.exp((-1 / 2) * (tf.multiply(tf.square(h), (tf.square(a) + 1))))
            return tf.multiply(term1, term2)

        z = gmean / tf.sqrt(1. + gvar)
        a = 1 / tf.sqrt(1. + (2 * gvar))

        cdfz = normcdf(z)
        tz = owent(z, a)

        pgmean = cdfz
        pgmeansq = (cdfz - 2. * tz)
        pgvar = (cdfz - 2. * tz - tf.square(cdfz))

        # clip negative values from variance terms to zero
        pgmeansq = (pgmeansq + tf.abs(pgmeansq)) / 2.
        pgvar = (pgvar + tf.abs(pgvar)) / 2.

        return pgmean, pgmeansq, pgvar
