import numpy as np
import pandas as pd

class preprocessing:
    def __init__(self,data):
        self.data = {'raw':{'traindf':data['traindf'],
                            'testdf' :data['testdf'],
                            'Xtrain' :data['Xtrain'],
                            'Ytrain' :data['Ytrain'],
                            'Xtest'  :data['Xtest'],
                            'Ytest'  :data['Ytest']}}
        self.filter_flg = False
        self.scale_flg = False

    def filter_time(self,min_idx=0, max_idx=np.inf):
        self.filter_flg = True

        train_filt_idx = np.logical_and(self.data['raw']['traindf'].ndatehour >= min_idx, self.data['raw']['traindf'].ndatehour <= max_idx)
        test_filt_idx  = np.logical_and(self.data['raw']['testdf'].ndatehour >= min_idx, self.data['raw']['testdf'].ndatehour <= max_idx)

        self.data.update({'filt':{'traindf':self.data['raw']['traindf'][train_filt_idx],
                                  'testdf':self.data['raw']['testdf'][test_filt_idx],
                                  'Xtrain':self.data['raw']['Xtrain'][train_filt_idx],
                                  'Ytrain':self.data['raw']['Ytrain'][train_filt_idx],
                                  'Xtest':self.data['raw']['Xtest'][test_filt_idx],
                                  'Ytest':self.data['raw']['Ytest'][test_filt_idx]}})

    def scale(self,scale_loc = False, scale_time = False):
        if scale_loc :
            self.scale_flg = True
            self.scale_flg_loc = True
        else:
            self.scale_flg_loc = False
        if scale_time:
             self.scale_flg = True
             self.scale_flg_time = True
        else:
             self.scale_flg_time = False


        if self.filter_flg:
            self.data.update({'scaled':self.data['filt']})
        else:
            self.data.update({'scaled':self.data['raw']})

        if scale_loc:
            self.scale_param = {'lat':{'min':min(self.data['scaled']['traindf'].lat.min(),self.data['scaled']['testdf'].lat.min()),
                                       'range':max(self.data['scaled']['traindf'].lat.max(),self.data['scaled']['testdf'].lat.max()) - \
                                               min(self.data['scaled']['traindf'].lat.min(),self.data['scaled']['testdf'].lat.min())},
                                'lon':{'min':min(self.data['scaled']['traindf'].lon.min(),self.data['scaled']['testdf'].lon.min()),
                                       'range':max(self.data['scaled']['traindf'].lon.max(),self.data['scaled']['testdf'].lon.max()) - \
                                               min(self.data['scaled']['traindf'].lon.min(),self.data['scaled']['testdf'].lon.min())}}
        if scale_time:
            self.scale_param.update({'ndatehour':{'min':min(self.data['scaled']['traindf'].ndatehour.min(),self.data['scaled']['testdf'].ndatehour.min()),
                                                  'range':max(self.data['scaled']['traindf'].ndatehour.max(),self.data['scaled']['testdf'].ndatehour.max()) - \
                                                          min(self.data['scaled']['traindf'].ndatehour.min(),self.data['scaled']['testdf'].ndatehour.min())}})

        if self.scale_flg:
            for key in ['Xtrain','Xtest','traindf','testdf']:
                dataset = self.data['scaled'][key]
                if scale_loc:
                    if type(dataset) is np.ndarray:
                        dataset[:,0] = (dataset[:,0] - self.scale_param['lat']['min'])/\
                                        self.scale_param['lat']['range']
                        dataset[:,1] = (dataset[:,1] - self.scale_param['lon']['min'])/\
                                        self.scale_param['lon']['range']
                    else:
                        dataset['lat'] = (dataset['lat'] - self.scale_param['lat']['min'])/\
                                            self.scale_param['lat']['range']
                        dataset['lon'] = (dataset['lon'] - self.scale_param['lon']['min'])/\
                                            self.scale_param['lon']['range']
                if scale_time:
                    if type(dataset) is np.ndarray:
                        dataset[:,2] = (dataset[:,2] - self.scale_param['ndatehour']['min'])/\
                                        self.scale_param['ndatehour']['range']
                    else:
                        dataset['ndatehour'] = (dataset['ndatehour'] - self.scale_param['ndatehour']['min'])/\
                                                self.scale_param['ndatehour']['range']
                self.data['scaled'][key] = dataset


    @property
    def model_data(self):
        if self.scale_flg:
            return self.data['scaled']
        elif self.filter_flg:
            return self.data['filt']
        else:
            return self.data['raw']

    @ property
    def shape(self):
        if self.scale_flg:
            return self.data['scaled']['Xtrain'].shape, self.data['scaled']['Xtest'].shape
        elif self.filter_flg:
            return self.data['filt']['Xtrain'].shape, self.data['filt']['Xtest'].shape
        else:
            return self.data['raw']['Xtrain'].shape, self.data['raw']['Xtest'].shape

    @property
    def kernel_params(self):
        obser_matrix = self.model_data['Xtrain']
        target = self.model_data['Ytrain']

        variance = np.max(target)
        if self.scale_flg:
            if self.scale_flg_loc:
                lengthscales = [round(3./self.scale_param['lat']['range'],4),
                                round(3./self.scale_param['lon']['range'],4)]
            else:
                lengthscales = [3.,3.]
        else:
            lengthscales = [3.,3.]

        if self.scale_flg:
            if self.scale_flg_time:
                lengthscales.append(round(3./self.scale_param['ndatehour']['range'],4))
            else:
                lengthscales.append(3.)
        else:
            lengthscales.append(3.)

        return (variance,lengthscales)
