import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib import ticker
import gpflow

def PlotOnOff1D(m):
    mpl.rcParams['figure.figsize'] = (11.0,10.0)
    mpl.rcParams.update({'font.size': 20})

    _X = m.Xtrain.value
    _Y = m.Ytrain.value

    _gfmean,_gfvar,_,_fmean,_fvar,_gmean,_gvar,_pgmean,_pgvar = m.predict_onoffgp(_X)
    _Zf = m.Zf.value
    _Kf = m.kernf.compute_K_symm(_X)
    _u_fm = m.u_fm.value
    _u_fs_sqrt = m.u_fs_sqrt.value

    _Zg = m.Zg.value
    _Kg = m.kerng.compute_K_symm(_X)
    _u_gm = m.u_gm.value
    _u_gs_sqrt = m.u_gs_sqrt.value

    _variance = m.likelihood.variance.value[0]

    _Kpg = _pgmean.reshape(-1,1) * _pgmean.reshape(1,-1)
    _Kfg = _Kpg * _Kf

    _X = _X.flatten()
    _Y = _Y.flatten()
    _gfmean = _gfmean.flatten()
    _gfvar  = _gfvar.flatten()
    _fmean  = _fmean.flatten()
    _fvar   = _fvar.flatten()
    _gmean  = _gmean.flatten()
    _gvar   = _gvar.flatten()
    _pgmean = _pgmean.flatten()
    _pgvar  = _pgvar.flatten()


    gs = gridspec.GridSpec(4, 4)
    ax1 = plt.subplot(gs[0, 0:-1])
    ax2 = plt.subplot(gs[1, 0:-1])
    ax3 = plt.subplot(gs[2, 0:-1])
    ax4 = plt.subplot(gs[3, 0:-1])
    ax5 = plt.subplot(gs[0, -1])
    ax6 = plt.subplot(gs[1, -1])
    ax7 = plt.subplot(gs[2, -1])
    ax8 = plt.subplot(gs[3, -1])


    # plot y
    ax1.plot(_X,_gfmean,'-',color='#ff7707')
    y1 = (_gfmean-1.5*((np.sqrt(_fvar) * _pgmean + np.sqrt(_pgvar)*(1-_pgmean)) + np.sqrt(_variance)))
    y2 = (_gfmean+1.5*((np.sqrt(_fvar) * _pgmean +  np.sqrt(_pgvar)*(1-_pgmean)) + np.sqrt(_variance)))
    ax1.fill_between(_X,y1,y2,facecolor='#ff7707',alpha=0.5)
    ax1.scatter(_X,_Y,s=8,
             color='black',alpha=0.7)
    ax1.set_xlim(0,10)
    # ax1.set_ylabel("Data" + r"$\mathbf{y}|\mathbf{f}$")
    ax1.set_title("(a) Predictive function \n"+r"$\mathbf{y}$",fontsize=18)
    ax1.set_yticks([-1,0,1], [])
    ax1.set_xticks([], [])

    # plot f and f|g
    ax2.plot(_X,_fmean,'-',color='#008b62',label=r"$f$")
    f1 = (_fmean-1.5*np.sqrt(_fvar))
    f2 = (_fmean+1.5*np.sqrt(_fvar))
    ax2.fill_between(_X,f1,f2,facecolor='#008b62',alpha=0.5)
    ax2.plot(_Zf,_u_fm,
             marker='o',linestyle = 'None',
             markeredgecolor = 'None',
             markerfacecolor='#008b62',alpha=0.7) #,label = 'uf (optimized)')

    ax2.plot(_X,_gfmean,'-',color='#ff7707',label=r"$f|g$")
    f3 = (_gfmean-1.5*(np.sqrt(_fvar) * _pgmean + np.sqrt(_pgvar)*(1-_pgmean)))
    f4 = (_gfmean+1.5*(np.sqrt(_fvar) * _pgmean + np.sqrt(_pgvar)*(1-_pgmean)))
    ax2.fill_between(_X,f3,f4,facecolor='#ff7707',alpha=0.5)

    ax2.set_xlim(0,10)
    # ax2.set_ylabel("Augmnted "+ r"$\mathbf{f}|\mathbf{g}$",fontsize=18)
    ax2.set_xticks([], [])
    ax2.set_yticks([-1,0,1], [])
    ax2.set_title("(c) Sparse latent function \n"+ r"$\mathbf{f}|\mathbf{g}$",fontsize=18)
    ax2.legend(loc="lower right",ncol=1,fontsize=18)

    # plot phi(gamma)
    ax3.plot(_X,_pgmean,'-',color='#003366')
    pg1 = (_pgmean-2*np.sqrt(_pgvar))
    pg2 = (_pgmean+2*np.sqrt(_pgvar))
    ax3.fill_between(_X,pg1,pg2,facecolor='#6684a3',alpha=0.7)
    ax3.axhline(y=0.5,linestyle='--',color='#333333')
    ax3.set_xlim(0,10)
    ax3.set_title("(e) Probit support function \n" + r"$\Phi(\mathbf{g})$",fontsize=18)
    ax3.set_xticks([], [])
    # ax3.set_ylabel("Support " + r"$\Phi(\mathbf{g})$",fontsize=18)

    # plot gamma
    ax4.plot(_X,_gmean,'-',color='#003366')
    ax4.plot(_Zg,_u_gm,
             marker='o',linestyle = 'None',
             markeredgecolor = 'None',
             markerfacecolor='#003366',alpha=0.8) #,label = 'ug (optimized)')
    g1 = (_gmean-2*np.sqrt(_gvar))
    g2 = (_gmean+2*np.sqrt(_gvar))
    ax4.fill_between(_X,g1,g2,facecolor='#6684a3',alpha=0.7)
    # ax4.set_ylabel("Latent  " +r"$\mathbf{g}$",fontsize=18)
    ax4.axhline(y=0.0,linestyle='--',color='#333333')
    ax4.set_title("(g) Latent function \n" +r"$\mathbf{g}$",fontsize=18)
    ax4.set_xlim(0,10)
    # plt.title('kernel lengthscale = %.3f, variance = %.3f' % (klg,ksg))

    im5 = ax5.imshow(_Kfg,cmap="viridis")
    cb = plt.colorbar(im5,ax=ax5,fraction=0.046, pad=0.03,extend="max")
    ax5.set_title("(b) Sparse kernel \n" + r"$\Phi(\mathbf{g}) \Phi(\mathbf{g})^T \circ K_f$",fontsize=18)
    ax5.set_xticks([], [])
    ax5.set_yticks([], [])
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb.locator = tick_locator
    cb.update_ticks()


    im6 = ax6.imshow(_Kf,cmap="viridis")
    cb = plt.colorbar(im6,ax=ax6,fraction=0.046, pad=0.03,extend="max")
    ax6.set_title("(d) Latent kernel \n"+r"$K_f$",fontsize=18)
    ax6.set_xticks([], [])
    ax6.set_yticks([], [])
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb.locator = tick_locator
    cb.update_ticks()


    im7 = ax7.imshow(_Kpg,cmap="viridis")
    cb = plt.colorbar(im7,ax=ax7,fraction=0.046, pad=0.03,extend="max")
    ax7.set_title("(f) Probit kernel \n" + r"$\Phi(\mathbf{g}) \Phi(\mathbf{g})^T$",fontsize=18)
    ax7.set_xticks([], [])
    ax7.set_yticks([], [])
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb.locator = tick_locator
    cb.update_ticks()


    im8 = ax8.imshow(_Kg,cmap="viridis")
    cb =  plt.colorbar(im8,ax=ax8,fraction=0.046, pad=0.03,extend="max")
    ax8.set_title("(h) Latent kernel \n"+r"$K_g$",fontsize=18)
    ax8.set_xticks([], [])
    ax8.set_yticks([], [])
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb.locator = tick_locator
    cb.update_ticks()

    plt.tight_layout()
    plt.subplots_adjust(hspace = 0.5,wspace=0.1)
    plt.savefig("plots/toy.png")
    plt.show()
