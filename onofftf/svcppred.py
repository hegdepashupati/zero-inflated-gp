import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl

from onofftf.main import Variable, Param, DataSet, GaussKLkron, GPConditional, KernSE, GaussKLkron
from onofftf.utils import modelmanager

from gpflow import transforms
from scipy.cluster.vq import kmeans

float_type = tf.float64
jitter_level = 1e-6

def predict_scgp(Xtrain,Xtest,checkpointPath,num_inducing_f = np.array([10,100]),include_f_mu = False):
    tf.reset_default_graph()

    # param initializations
    list_to_np = lambda _list : [np.array(e) for e in _list]

    init_fkell = list_to_np([[8.,8.],[5./1000]])
    init_fkvar = list_to_np([[20.],[20.]])

    init_noisevar = 0.001

    q_diag = True
    if include_f_mu:
        init_f_mu = 0.

    init_Zf_s = kmeans(Xtrain[:,0:2],num_inducing_f[0])[0]
    init_Zf_t = np.expand_dims(np.linspace(Xtrain[:,2].min(),Xtrain[:,2].max(),num_inducing_f[1]),axis=1)

    init_Zf = [init_Zf_s,init_Zf_t]

    init_u_fm = np.random.randn(np.prod(num_inducing_f),1)*0.1
    init_u_fs_sqrt = np.ones(np.prod(num_inducing_f)).reshape(1,-1).T

    kern_param_learning_rate = 1e-4
    indp_param_learning_rate = 1e-4


    # ****************************************************************
    # define tensorflow variables and placeholders
    # ****************************************************************
    X = tf.placeholder(dtype = float_type)
    Y = tf.placeholder(dtype = float_type)

    with tf.name_scope("f_kern"):
        fkell = [Param(init_fkell[i],transform=transforms.Log1pe(),
                       name="lengthscale",learning_rate = kern_param_learning_rate,summ=False)
                 for i in range(len(num_inducing_f))]

        fkvar = [Param(init_fkvar[i],transform=transforms.Log1pe(),
                       name="variance",learning_rate = kern_param_learning_rate,summ=False)
                 for i in range(len(num_inducing_f))]

    fkern_list = [KernSE(fkell[i],fkvar[i]) for i in range(len(num_inducing_f))]

    with tf.name_scope("f_ind"):
        Zf_list = [Param(init_Zf[i],name="z",learning_rate = indp_param_learning_rate,summ=False)
                   for i in range(len(num_inducing_f))]

        u_fm = Param(init_u_fm,name="value",learning_rate = indp_param_learning_rate,summ=False)
        if q_diag:
            u_fs_sqrt = Param(init_u_fs_sqrt,transforms.positive,
                              name="variance",learning_rate = indp_param_learning_rate,summ=False)
        else:
            u_fs_sqrt = Param(init_u_fs_sqrt,transforms.LowerTriangular(init_u_fs_sqrt.shape[0]),
                              name="variance",learning_rate = indp_param_learning_rate,summ=False)

    # ****************************************************************
    # define model support functions
    # ****************************************************************
    def build_prior_kl(u_fm, u_fs_sqrt, fkern_list, Zf_list,whiten=False):
        if whiten:
            raise NotImplementedError()
        else:
            Kfmm = [fkern_list[i].K(Zf_list[i].get_tfv()) + \
                    tf.eye(num_inducing_f[i], dtype=float_type) * jitter_level
                    for i in range(len(num_inducing_f))]

            KL = GaussKLkron(u_fm.get_tfv(), u_fs_sqrt.get_tfv(), Kfmm)

        return KL

    def build_predict(Xnew,u_fm,u_fs_sqrt,fkern_list,Zf_list,f_mu=None):

        input_mask_f = _gen_inp_mask(Zf_list)

        # compute fmean and fvar from the kronecker inference
        fmean,fvar = kron_inf(Xnew,fkern_list,Zf_list,u_fm,u_fs_sqrt,num_inducing_f,input_mask_f)


        if f_mu is not None:
            fmean = fmean + f_mu.get_tfv()

        p = probit(fmean / tf.sqrt(1 + fvar))
        pfmean, pfvar = p, p - tf.square(p)

        return pfmean, pfvar

    def kron_inf(Xnew,kern_list,Z_list,q_mu,q_sqrt,num_inducing,input_mask):
        # Compute alpha = K_mm^-1 * f_m
        Kmm = [kern_list[p].K(Z_list[p].get_tfv()) + \
               tf.eye(num_inducing[p], dtype=float_type) * jitter_level
               for p in range(len(num_inducing))]

        Kmm_inv = [tf.matrix_inverse(Kmm[p]) for p in range(len(num_inducing))]
        alpha = __kron_mv(Kmm_inv,q_mu.get_tfv())

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
            A = __kron_mv(Kmm_inv,Kmn)
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

    def __kron_mv( As, x):
        num_inducing = [int(As[p].get_shape()[0]) for p in range(len(As))]
        N = np.prod(num_inducing)
        b = tf.reshape(x, [N,1])
        for p in range(len(As)):
            Ap = As[p]
            X = tf.reshape(b, (num_inducing[p],
                               np.round(N/num_inducing[p]).astype(np.int)))
            b = tf.matmul(X, Ap, transpose_a=True, transpose_b=True)
            b = tf.reshape(b, [N,1])
        return b

    def tf_kron(*args):
        def __tf_kron(a,b):

            a_shape = [tf.shape(a)[0],tf.shape(a)[1]]
            b_shape = [tf.shape(b)[0],tf.shape(b)[1]]

            return tf.reshape(tf.reshape(a,[a_shape[0],1,a_shape[1],1])* \
                              tf.reshape(b,[1,b_shape[0],1,b_shape[1]]),
                              [a_shape[0]*b_shape[0],a_shape[1]*b_shape[1]])

        kron_pord = tf.constant(1.,shape=[1,1],dtype=float_type)
        for Ap in args:
            kron_pord = __tf_kron(kron_pord,Ap)

        return kron_pord

    def _gen_inp_mask(Z_list):
        input_mask = []
        tmp = 0
        for p in range(len(Z_list)):
            p_dim = Z_list[p].shape[1]
            input_mask.append(np.arange(tmp, tmp + p_dim, dtype=np.int32))
            tmp += p_dim
        return input_mask

    def variational_expectations(Y,pfmean):
        return bernoulli(pfmean,Y)

    def bernoulli(p, y):
        return tf.log(tf.where(tf.equal(y, 1), p, 1-p))

    def probit(x):
        return 0.5 * (1.0 + tf.erf(x / np.sqrt(2.0))) * (1 - 2e-3) + 1e-3

    # ****************************************************************
    # build model and define lower bound
    # ****************************************************************
    with tf.name_scope("model_build"):
        if include_f_mu:
            pfmean,pfvar = build_predict(X,u_fm,u_fs_sqrt,fkern_list,Zf_list,f_mu)
        else:
            pfmean,pfvar = build_predict(X,u_fm,u_fs_sqrt,fkern_list,Zf_list)

    # load model
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    modelmngr = modelmanager(saver, sess, checkpointPath)
    modelmngr.load()

    # return inside a dictionary
    pred_train = {'pfmean' : pfmean.eval(feed_dict = {X:Xtrain}),
                  'pfvar' : pfvar.eval(feed_dict = {X:Xtrain})}

    if Xtest is not None:
        pred_test = {'pfmean' : pfmean.eval(feed_dict = {X:Xtest}),
                     'pfvar' : pfvar.eval(feed_dict = {X:Xtest})}

    sess.close()

    if Xtest is not None:
        return pred_train, pred_test
    else:
        return pred_train
