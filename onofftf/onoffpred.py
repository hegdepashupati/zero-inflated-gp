import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl

from onofftf.main import Variable, Param, DataSet, GaussKLkron, GPConditional, KernSE
from onofftf.utils import modelmanager

from gpflow import transforms
from scipy.cluster.vq import kmeans

float_type = tf.float64
jitter_level = 1e-6

def predict_onoff(Xtrain,Xtest,checkpointPath,num_inducing_f = np.array([10,100]),num_inducing_g = np.array([10,100]),include_fmu = False):
    tf.reset_default_graph()

    # param initializations
    list_to_np = lambda _list : [np.array(e) for e in _list]

    init_fkell = list_to_np([[8.,8.],[5./1000]])
    init_fkvar = list_to_np([[20.],[20.]])

    init_gkell = list_to_np([[8.,8.],[5./1000]])
    init_gkvar = list_to_np([[10.],[10.]])

    init_noisevar = 0.001

    q_diag = True
    if include_fmu:
        init_f_mu = 0.

    init_Zf_s = kmeans(Xtrain[:,0:2],num_inducing_f[0])[0]
    init_Zf_t = np.expand_dims(np.linspace(Xtrain[:,2].min(),Xtrain[:,2].max(),num_inducing_f[1]),axis=1)

    init_Zf = [init_Zf_s,init_Zf_t]
    init_Zg = init_Zf.copy()


    init_u_fm = np.random.randn(np.prod(num_inducing_f),1)*0.1
    init_u_gm = np.random.randn(np.prod(num_inducing_g),1)*0.1

    init_u_fs_sqrt = np.ones(np.prod(num_inducing_f)).reshape(1,-1).T
    init_u_gs_sqrt = np.ones(np.prod(num_inducing_g)).reshape(1,-1).T

    kern_param_learning_rate = 1e-4
    indp_param_learning_rate = 1e-4


    # tf variable declarations
    X = tf.placeholder(dtype = float_type)
    Y = tf.placeholder(dtype = float_type)

    with tf.name_scope("f_kern"):
        fkell = [Param(init_fkell[i],transform=transforms.Log1pe(),
                       name="lengthscale",learning_rate = kern_param_learning_rate,summ=True)
                 for i in range(len(num_inducing_f))]

        fkvar = [Param(init_fkvar[i],transform=transforms.Log1pe(),
                       name="variance",learning_rate = kern_param_learning_rate,summ=True)
                 for i in range(len(num_inducing_f))]

    fkern_list = [KernSE(fkell[i],fkvar[i]) for i in range(len(num_inducing_f))]

    with tf.name_scope("g_kern"):
        gkell = [Param(init_gkell[i],transform=transforms.Log1pe(),
                       name="lengthscale",learning_rate = kern_param_learning_rate,summ=True)
                 for i in range(len(num_inducing_g))]

        gkvar = [Param(init_gkvar[i],transform=transforms.Log1pe(),
                       name="variance",learning_rate = kern_param_learning_rate,summ=True)
                 for i in range(len(num_inducing_g))]

    gkern_list = [KernSE(gkell[i],gkvar[i]) for i in range(len(num_inducing_g))]

    with tf.name_scope("likelihood"):
        noisevar = Param(init_noisevar,transform=transforms.Log1pe(),
                         name="variance",learning_rate = kern_param_learning_rate,summ=True)


    with tf.name_scope("f_ind"):
        Zf_list = [Param(init_Zf[i],name="z",learning_rate = indp_param_learning_rate,summ=True)
                   for i in range(len(num_inducing_f))]

        u_fm = Param(init_u_fm,name="value",learning_rate = indp_param_learning_rate,summ=True)
        if q_diag:
            u_fs_sqrt = Param(init_u_fs_sqrt,transforms.positive,
                              name="variance",learning_rate = indp_param_learning_rate,summ=True)
        else:
            u_fs_sqrt = Param(init_u_fs_sqrt,transforms.LowerTriangular(init_u_fs_sqrt.shape[0]),
                              name="variance",learning_rate = indp_param_learning_rate,summ=True)

    #     f_mu = Param(init_f_mu,name="fmu",learning_rate = indp_param_learning_rate,summ=True)

    with tf.name_scope("g_ind"):
        Zg_list = [Param(init_Zg[i],name="z",learning_rate = indp_param_learning_rate,summ=True)
                   for i in range(len(num_inducing_g))]

        u_gm = Param(init_u_gm,name="value",learning_rate = indp_param_learning_rate,summ=True)
        if q_diag:
            u_gs_sqrt = Param(init_u_gs_sqrt,transforms.positive,
                              name="variance",learning_rate = indp_param_learning_rate,summ=True)
        else:
            u_gs_sqrt = Param(init_u_gs_sqrt,transforms.LowerTriangular(init_u_gs_sqrt.shape[0]),
                              name="variance",learning_rate = indp_param_learning_rate,summ=True)



    def build_prior_kl(u_fm, u_fs_sqrt, fkern_list, Zf_list,
                       u_gm, u_gs_sqrt, gkern_list, Zg_list, whiten=False):
        if whiten:
            raise NotImplementedError()
        else:
            Kfmm = [fkern_list[i].K(Zf_list[i].get_tfv()) + \
                    tf.eye(num_inducing_f[i], dtype=float_type) * jitter_level
                    for i in range(len(num_inducing_f))]

            Kgmm = [gkern_list[i].K(Zg_list[i].get_tfv()) + \
                    tf.eye(num_inducing_g[i], dtype=float_type) * jitter_level
                    for i in range(len(num_inducing_g))]

            KL = GaussKLkron(u_fm.get_tfv(), u_fs_sqrt.get_tfv(), Kfmm) + \
                 GaussKLkron(u_gm.get_tfv(), u_gs_sqrt.get_tfv(), Kgmm)

        return KL

    def build_predict(Xnew,u_fm,u_fs_sqrt,fkern_list,Zf_list,u_gm,u_gs_sqrt,gkern_list,Zg_list,f_mu=None):

        input_mask_f = _gen_inp_mask(Zf_list)
        input_mask_g = _gen_inp_mask(Zg_list)

        # compute fmean and fvar from the kronecker inference
        fmean,fvar = kron_inf(Xnew,fkern_list,Zf_list,u_fm,u_fs_sqrt,num_inducing_f,input_mask_f)
        # fmean = fmean + mean_function(Xnew)
        if not f_mu is None :
            fmean = fmean + f_mu.get_tfv()


        # compute gmean and gvar from the kronecker inference
        gmean,gvar = kron_inf(Xnew,gkern_list,Zg_list,u_gm,u_gs_sqrt,num_inducing_g,input_mask_g)
        gmean = gmean + tf.cast(tf.constant(-1.0),float_type)

        # compute augemented distributions
        ephi_g, ephi2_g, evar_phi_g = probit_expectations(gmean, gvar)

        # compute augmented f
        # p(f|g) = N(f| diag(ephi_g)* A*u_fm, diag(evar_phi_g)) * (Kfnn + A(u_fs - Kfmm)t(A)))
        gfmean = tf.multiply(ephi_g, fmean)
        gfvar = tf.multiply(ephi2_g, fvar)
        gfmeanu = tf.multiply(evar_phi_g, tf.square(fmean))

        # return mean and variance vectors in order
        return gfmean, gfvar, gfmeanu, fmean, fvar, gmean, gvar, ephi_g, evar_phi_g


    def kron_inf(Xnew,kern_list,Z_list,q_mu,q_sqrt,num_inducing,input_mask):
        # Compute alpha = K_mm^-1 * f_m
        Kmm = [kern_list[p].K(Z_list[p].get_tfv()) + \
               tf.eye(num_inducing[p], dtype=float_type) * jitter_level
               for p in range(len(num_inducing))]

        Kmm_inv = [tf.matrix_inverse(Kmm[p]) for p in range(len(num_inducing))]
        alpha = __kron_mv(Kmm_inv,q_mu.get_tfv(),num_inducing)

        n_batch = tf.stack([tf.shape(Xnew)[0],np.int32(1)])
        Knn = tf.ones(n_batch, dtype=float_type)
        KMN = []

        for p in range(len(num_inducing)):
            xnew = tf.gather(Xnew, input_mask[p], axis=1)
            Knn *= tf.reshape(kern_list[p].Kdiag(xnew), n_batch)
            KMN.append(kern_list[p].K(Z_list[p].get_tfv(), xnew))

        S = tf.diag(tf.squeeze(tf.square(q_sqrt.get_tfv())))

        def loop_rows(n,mu,var):
            Kmn = tf.reshape(KMN[0][:,n], [num_inducing[0],1])
            for p in range(1,len(num_inducing)):
                Kmn = tf_kron(Kmn,tf.reshape(KMN[p][:,n],[num_inducing[p],1]))

            mu_n = tf.matmul(Kmn, alpha, transpose_a=True)
            mu = mu.write(n, mu_n)
            A = __kron_mv(Kmm_inv,Kmn,num_inducing)
            tmp = Knn[n] - tf.matmul(Kmn, A,transpose_a=True) + \
                           tf.matmul(tf.matmul(A,S,transpose_a=True),A)

            var = var.write(n, tmp)
            return tf.add(n,1), mu, var

        def loop_cond(n,mu,var):
            return tf.less(n, n_batch[0])

        mu = tf.TensorArray(float_type, size=n_batch[0])
        var = tf.TensorArray(float_type, size=n_batch[0])
        _, mu, var = tf.while_loop(loop_cond, loop_rows, [0, mu, var])

        mu = tf.reshape(mu.stack(), n_batch)
        var = tf.reshape(var.stack(), n_batch)

        return mu , var

    def __kron_mv( As, x,num_inducing):
        N = np.prod(num_inducing)
        b = tf.reshape(x, [N,1])
        for p in range(len(As)):
            Ap = As[p]
            X = tf.reshape(b, (num_inducing[p],
                               np.round(N/num_inducing[p]).astype(np.int)))
            b = tf.matmul(X, Ap, transpose_a=True, transpose_b=True)
            b = tf.reshape(b, [N,1])
        return b

    def tf_kron(a,b):
        a_shape = [a.shape[0].value,a.shape[1].value]
        b_shape = [b.shape[0].value,b.shape[1].value]
        return tf.reshape(tf.reshape(a,[a_shape[0],1,a_shape[1],1])* \
                          tf.reshape(b,[1,b_shape[0],1,b_shape[1]]),
                          [a_shape[0]*b_shape[0],a_shape[1]*b_shape[1]])

    def _gen_inp_mask(Z_list):
        input_mask = []
        tmp = 0
        for p in range(len(Z_list)):
            p_dim = Z_list[p].shape[1]
            input_mask.append(np.arange(tmp, tmp + p_dim, dtype=np.int32))
            tmp += p_dim
        return input_mask


    def variational_expectations(Y,fmu,fvar,fmuvar,noisevar):
        return -0.5 * np.log(2 * np.pi) - 0.5 * tf.log(noisevar) \
                - 0.5 * (tf.square(Y - fmu) + fvar + fmuvar) / noisevar

    def probit_expectations(gmean, gvar):
        def normcdf(x):
            return 0.5 * (1.0 + tf.erf(x / np.sqrt(2.0))) * (1. - 2.e-3) + 1.e-3

        def owent(h, a):
            h = tf.abs(h)
            term1 = tf.atan(a) / (2 * np.pi)
            term2 = tf.exp((-1 / 2) * (tf.multiply(tf.square(h), (tf.square(a) + 1))))
            return tf.multiply(term1, term2)

        z = gmean / tf.sqrt(1. + gvar)
        a = 1 / tf.sqrt(1. + (2 * gvar))

        cdfz = normcdf(z)
        tz = owent(z, a)

        ephig = cdfz
        ephisqg = (cdfz - 2. * tz)
        evarphig = (cdfz - 2. * tz - tf.square(cdfz))

        # clip negative values from variance terms to zero
        ephisqg = (ephisqg + tf.abs(ephisqg)) / 2.
        evarphig = (evarphig + tf.abs(evarphig)) / 2.

        return ephig, ephisqg, evarphig


    kl = build_prior_kl(u_fm,u_fs_sqrt,fkern_list,Zf_list,
                        u_fm,u_fs_sqrt,fkern_list,Zf_list)
    gfmean, gfvar, gfmeanu, fmean, fvar, gmean, gvar, pgmean, pgvar = build_predict(X,u_fm,u_fs_sqrt,fkern_list,Zf_list,u_gm,u_gs_sqrt,gkern_list,Zg_list)

    # load model
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    modelmngr = modelmanager(saver, sess, checkpointPath)
    modelmngr.load()

    pred_train = {'gfmean' : gfmean.eval(feed_dict = {X:Xtrain}),
                  'fmean' : fmean.eval(feed_dict = {X:Xtrain}),
                  'pgmean' : pgmean.eval(feed_dict = {X:Xtrain})}

    if Xtest is not None:
        pred_test = {'gfmean' : gfmean.eval(feed_dict = {X:Xtest}),
                     'fmean' : fmean.eval(feed_dict = {X:Xtest}),
                     'pgmean' : pgmean.eval(feed_dict = {X:Xtest})}
    sess.close()

    if Xtest is not None:
        return pred_train, pred_test
    else:
        return pred_train
