import time
import numpy as np

def printtime(start):
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return str("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

def show(image):
    """
    copied shamelessly from here : https://gist.github.com/akesling/5358964
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    mpl.rcParams['figure.figsize'] = (5, 5)
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    fig.colorbar(imgplot)
    pyplot.show()

class kernse_np:
    '''
    Taken from GPFlow
    '''
    def __init__(self, lengthscales,variance):
        self.lengthscales = lengthscales
        self.variance = variance

    def square_dist(self,X, X2=None):
        X = X / self.lengthscales
        Xs = np.sum(np.square(X), 1)
        if X2 is None:
            return -2 * np.matmul(X, X.T) + \
                        np.reshape(Xs, (-1, 1)) + \
                        np.reshape(Xs, (1, -1))
        else:
            X2 = X2 / self.lengthscales
            X2s = np.sum(np.square(X2), 1)
            return -2 * np.matmul(X, X2.T) + \
                        np.reshape(Xs, (-1, 1)) + \
                        np.reshape(X2s,(1, -1))

    def K(self,X,X2=None):
        if X2 is None:
            return self.variance * np.exp(-self.square_dist(X) / 2)
        else:
            return self.variance * np.exp(-self.square_dist(X, X2) / 2)

    def Ksymm(self,X):
        return self.variance * np.exp(-self.square_dist(X) / 2)

    def Kdiag(self,X):
        return np.full(np.stack([np.shape(X)[0]]), np.squeeze(self.variance))


class modelmanager:
    def __init__(self,saver,sess,path):
        self.sess = sess
        self.saver = saver
        self.path = path

    def save(self):
        self.saver.save(self.sess,self.path)
        print("model saved in : "+self.path)

    def load(self):
        self.saver.restore(self.sess,self.path)
        print("model loaded from : "+self.path)
