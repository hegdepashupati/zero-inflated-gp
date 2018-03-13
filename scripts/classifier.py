# ****************************************************************
# library import block
# ****************************************************************
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import logging
import time
import sys
from scipy.cluster.vq import kmeans
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
plt.switch_backend('agg')

float_type = tf.float64
jitter_level = 1e-5
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main(scriptPath):
    tf.reset_default_graph()
    parentDir = '/'.join(os.path.dirname(os.path.realpath(scriptPath)).split('/')[:-1])  #
    subDir = "/" + scriptPath.split("/")[-2].split(".py")[0] + "/"
    sys.path.append(parentDir)
    modelPath = 'model.ckpt'

    from onofftf.main import Param, DataSet, GaussKL, KernSE, GPConditional, GaussKLkron
    from onofftf.utils import modelmanager
    from gpflow import transforms

    modelPath = parentDir + subDir + 'model_scgp.ckpt'
    tbPath    = parentDir + subDir + "/scgp"
    logPath   = parentDir + subDir + 'modelsumm_scgp.log'

    logger = logging.getLogger('log')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.FileHandler(logPath))

    data = pickle.load(open(parentDir + subDir +"data.pickle","rb"))
    Xtrain = data['Xtrain']
    Ytrain = data['Ytrain']
    Ytrain = Ytrain > 0 * 1
    Xtest = data['Xtest']
    Ytest = data['Ytest']
    Ytest = Ytest > 0 * 1
    traindf = pd.DataFrame({'ndatehour':Xtrain[:,2].flatten()*1000,'pptr':Ytrain.flatten()})
    train_data = DataSet(Xtrain, Ytrain)

    logger.info("traning size   = " + str(Xtrain.shape[0]))
    logger.info("test size   = " + str(Xtest.shape[0]))


    # ****************************************************************
    # parameter initializations
    # ****************************************************************
    list_to_np = lambda _list : [np.array(e) for e in _list]

    num_iter = 500
    num_inducing_f = np.array([10,100])
    num_data = Xtrain.shape[0]
    num_minibatch = 1000

    init_fkell = list_to_np([[5.,5.],[5./1000]])
    init_fkvar = list_to_np([[20.],[20.]])
    init_noisevar = 0.01

    q_diag = True
    include_f_mu = False
    if include_f_mu:
        init_f_mu = 0.

    init_Zf_s = kmeans(Xtrain[:,0:2],num_inducing_f[0])[0]
    init_Zf_t = np.expand_dims(np.linspace(Xtrain[:,2].min(),Xtrain[:,2].max(),num_inducing_f[1]),axis=1)
    init_Zf = [init_Zf_s,init_Zf_t]

    init_u_fm = np.random.randn(np.prod(num_inducing_f),1)*0.01
    init_u_fs_sqrt = np.ones(np.prod(num_inducing_f)).reshape(1,-1).T

    kern_param_learning_rate = 1e-3
    indp_param_learning_rate = 1e-3

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
        Kmn_kron = []

        for p in range(len(num_inducing)):
            xnew = tf.gather(Xnew, input_mask[p], axis=1)
            Knn *= tf.reshape(kern_list[p].Kdiag(xnew), n_batch)
            Kmn_kron.append(kern_list[p].K(Z_list[p].get_tfv(), xnew))

        S = tf.diag(tf.squeeze(tf.square(q_sqrt.get_tfv())))

        Kmn = tf.reshape(tf.multiply(tf.expand_dims(Kmn_kron[0],1),Kmn_kron[1]),[np.prod(num_inducing),-1])
        A  = tf.matmul(tf_kron(*Kmm_inv),Kmn)

        mu = tf.matmul(Kmn, alpha, transpose_a=True)
        var = Knn - tf.reshape(tf.matrix_diag_part(tf.matmul(Kmn, A,transpose_a=True) - \
                               tf.matmul(tf.matmul(A,S,transpose_a=True),A)),[-1,1])

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
    # get kl term
    with tf.name_scope("kl"):
        kl = build_prior_kl(u_fm,u_fs_sqrt,fkern_list,Zf_list)

    with tf.name_scope("model_build"):
        if include_f_mu:
            pfmean,pfvar = build_predict(X,u_fm,u_fs_sqrt,fkern_list,Zf_list,f_mu)
        else:
            pfmean,pfvar = build_predict(X,u_fm,u_fs_sqrt,fkern_list,Zf_list)

    # compute likelihood
    with tf.name_scope("var_exp"):
        var_exp = tf.reduce_sum(variational_expectations(Y,pfmean))
        scale =  tf.cast(num_data, float_type) / tf.cast(num_minibatch, float_type)
        var_exp_scaled = var_exp * scale

    # final lower bound
    with tf.name_scope("cost"):
        cost =  -(var_exp_scaled - kl)

    # ****************************************************************
    # define optimizer op
    # ****************************************************************
    all_var_list = tf.trainable_variables()
    all_lr_list = [var._learning_rate for var in all_var_list]

    train_opt_group = []

    for group_learning_rate in set(all_lr_list):
        _ind_bool = np.where(np.isin(np.array(all_lr_list),group_learning_rate))[0]
        group_var_list = [all_var_list[ind] for ind in _ind_bool]
        group_tf_optimizer = tf.train.AdamOptimizer(learning_rate = group_learning_rate)
        group_grad_list = tf.gradients(cost,group_var_list)
        group_grads_and_vars = list(zip(group_grad_list,group_var_list))

        group_train_op = group_tf_optimizer.apply_gradients(group_grads_and_vars)
        train_opt_group.append({'names':[var.name for var in group_var_list],
                                'vars':group_var_list,
                                'learning_rate':group_learning_rate,
                                'grads':group_grad_list,
                                'train_op':group_train_op})

    train_op = tf.group(*[group['train_op'] for group in train_opt_group])



    # ****************************************************************
    # define graph and run optimization
    # ****************************************************************
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())


    logger.info('*******  started optimization at ' + time.strftime('%Y%m%d-%H%M') + " *******")
    optstime = time.time()
    logger.info(
        '{:>16s}'.format("iteration") + '{:>6s}'.format("time"))

    for i in range(num_iter):
        optstime = time.time()
        batch = train_data.next_batch(num_minibatch)
        try:
            sess.run([train_op],feed_dict={X : batch[0],Y : batch[1]})
            if i% 100 == 0:
                logger.info(
                    '{:>16d}'.format(i) + '{:>6.3f}'.format((time.time() - optstime)/60))

            if i% 10000 == 0:
                modelmngr = modelmanager(saver, sess, modelPath)
                modelmngr.save()

                # ****************************************************************
                # plot inducing monitoring plots
                # ****************************************************************
                lp_u_fm = u_fm.get_tfv().eval().flatten()

                lp_zf_t = Zf_list[1].get_tfv().eval().flatten()

                lp_zf_sort_ind = np.argsort(lp_zf_t)

                scale_z = 1000
                mpl.rcParams['figure.figsize'] = (16,8)
                fig, (ax1,ax2) = plt.subplots(2, 1, sharex=True)

                mean_pptr = traindf.groupby('ndatehour')['pptr'].mean()
                ax1.bar(mean_pptr.index, mean_pptr.values, align='center')

                for m in np.arange(num_inducing_f[0]):
                    u_fm_temporal = lp_u_fm[m*num_inducing_f[1]:(m+1)*num_inducing_f[1]]
                    ax2.plot(np.round(lp_zf_t[lp_zf_sort_ind] * scale_z,4),u_fm_temporal[lp_zf_sort_ind],alpha=0.7)
                ax2.scatter(np.round(lp_zf_t[lp_zf_sort_ind] * scale_z,4),np.ones([num_inducing_f[1],1])*lp_u_fm.min(),color="#514A30")

                fig.savefig(parentDir+ subDir + "scgp_inducing_"+str(i)+".png")

        except KeyboardInterrupt as e:
            print("Stopping training")
            break

    modelmngr = modelmanager(saver, sess, modelPath)
    modelmngr.save()
    tf.reset_default_graph()

    # ****************************************************************
    # param summary
    # ****************************************************************
    logger.info("Kf spatial lengthscale  = " + str(fkell[0].get_tfv().eval()))
    logger.info("Kf spatial variance     = " + str(fkvar[0].get_tfv().eval()))
    logger.info("Kf temporal lengthscale = " + str(fkell[1].get_tfv().eval()))
    logger.info("Kf temporal variance    = " + str(fkvar[1].get_tfv().eval()))

    # ****************************************************************
    # model predictions
    # ****************************************************************
    from onofftf.svcppred import predict_scgp
    def accuracy(predict,actual):
        predict = predict > 0.5 * 1
        return accuracy_score(actual, predict)

    def precision(predict,actual):
        predict = predict > 0.5 * 1
        return precision_score(actual, predict)

    def recall(predict,actual):
        predict = predict > 0.5 * 1
        return recall_score(actual, predict)

    def auc(predict,actual):
        return roc_auc_score(actual, predict)

    pred_train_scgp, pred_test_scgp = predict_scgp(Xtrain = Xtrain,
                                                   Xtest = Xtest,
                                                   checkpointPath = parentDir+ subDir + "model_scgp.ckpt")

    train_accuracy = accuracy(pred_train_scgp["pfmean"],Ytrain)
    train_precision = precision(pred_train_scgp["pfmean"],Ytrain)
    train_recall = recall(pred_train_scgp["pfmean"],Ytrain)
    train_auc = auc(pred_train_scgp["pfmean"],Ytrain)
    logger.info("accuracy on train set for scgp : "+str(train_accuracy))
    logger.info("precision on train set for scgp : "+str(train_precision))
    logger.info("recall on train set for scgp : "+str(train_recall))
    logger.info("auc on train set for scgp : "+str(train_auc))

    test_accuracy = accuracy(pred_test_scgp["pfmean"],Ytest)
    test_precision = precision(pred_test_scgp["pfmean"],Ytest)
    test_recall = recall(pred_test_scgp["pfmean"],Ytest)
    test_auc = auc(pred_test_scgp["pfmean"],Ytest)
    logger.info("accuracy on test set for scgp : "+str(test_accuracy))
    logger.info("precision on test set for scgp : "+str(test_precision))
    logger.info("recall on test set for scgp : "+str(test_recall))
    logger.info("auc on test set for scgp : "+str(test_auc))
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

    # ****************************************************************
    # return values
    # ****************************************************************
    results = {
               'pred_train':pred_train_scgp,
               'pred_test':pred_test_scgp,
               'train_accuracy':train_accuracy,
               'train_precision':train_precision,
               'train_recall':train_recall,
               'train_auc':train_auc,
               'test_accuracy':test_accuracy,
               'test_precision':test_precision,
               'test_recall':test_recall,
               'test_auc':test_auc
               }
    pickle.dump(results,open(parentDir+ subDir +"results_scgp.pickle","wb"))


if __name__ == "__main__":
    scriptPath = sys.argv[0]
    main(scriptPath)
