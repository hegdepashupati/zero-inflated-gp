import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pickle
from copy import deepcopy
import matplotlib.gridspec as gridspec
import matplotlib as mpl

def PlotOnOffPrecipitationModel(plotdf,gpmodel,plotvar = "pptr",timevar='ndatehour',ngrid = 100):
    mpl.rcParams['figure.figsize'] = (16,8)

    Xtest = plotdf[['lat','lon',timevar]].values.reshape(plotdf.shape[0],3)
    timevalue = plotdf['ndatehour'].tolist()[0]


    # create basemap instance
    pmap = Basemap(resolution='l', # c, l, i, h, f or None
            projection='merc',
            lat_0=65.0, lon_0=25.0,
            llcrnrlon=19.08, llcrnrlat= 59.45, urcrnrlon=31.59, urcrnrlat=70.09)

    xco = []
    yco =[]
    for row in plotdf[['lat','lon']].itertuples():
        lat,lon = row[1],row[2]
        x, y = pmap(lon,lat)
        xco.append(x);yco.append(y)

    # prediction on test data
    pgfmean,pgfmean2,pgfvar,pfmean,pfvar,pgmean,pgvar,ppgmean,ppgvar = gpmodel.predict_onoffgp(Xtest)

    # prediction on grid data
    lonsgrid,latsgrid = pmap.makegrid(ngrid,ngrid)
    xgrid, ygrid = pmap(lonsgrid, latsgrid)
    griddata = np.array([latsgrid.flatten(),
                         lonsgrid.flatten(),
                         np.repeat(timevalue,ngrid*ngrid)])
    griddata = griddata.T

    tgfmean,tgfmean2,tgfvar,tfmean,tfvar,tgmean,tgvar,tpgmean,tpgvar = gpmodel.predict_onoffgp(griddata)

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 7)

    # plot 01 : actual data
    ax1 = fig.add_subplot(gs[0:2,0:2])
    p1 = deepcopy(pmap)
    p1.drawcountries(linewidth=0.8)
    p1.drawcoastlines(linewidth=0.4,linestyle="--")
    p1.scatter(xco,yco,c=plotdf[plotvar].tolist(),alpha=0.7,s=40, marker='o')
    p1.colorbar(extend='max')
    ax1.set_title("data")


    # plot 02: model prediction
    ax2 = fig.add_subplot(gs[0:2,2:4])
    p2 = deepcopy(pmap)
    p2.drawcountries(linewidth=0.8)
    p2.drawcoastlines(linewidth=0.4,linestyle="--")
    p2.scatter(xco,yco,c=pgfmean.tolist(),alpha=0.7,s=40, marker='o')
    p2.colorbar(extend='max')
    ax2.set_title("augmented mean prediction")



    # plot 03
    ax3 = fig.add_subplot(gs[0,4])
    p3 = deepcopy(pmap)
    p3.drawcountries(linewidth=0.4)
    p3.drawcoastlines(linewidth=0.2,linestyle="--")
    # p3.scatter(xco,yco,c=pgfmean.tolist(),alpha=1,s=30, marker='o')
    p3.contourf(xgrid,ygrid,tfmean.reshape(ngrid,ngrid),cmap='RdYlGn',alpha=0.5)
    p3.colorbar(extend='max')
    ax3.set_title("f mean")


    # plot 04
    ax4 = fig.add_subplot(gs[1,4])
    p4 = deepcopy(pmap)
    p4.drawcountries(linewidth=0.4)
    p4.drawcoastlines(linewidth=0.2,linestyle="--")
    p4.contourf(xgrid,ygrid,tfvar.reshape(ngrid,ngrid),cmap='RdPu',alpha=0.5)
    p4.colorbar(extend='max')
    ax4.set_title("f var")

    # plot 05
    ax5 = fig.add_subplot(gs[0,5])
    p5 = deepcopy(pmap)
    p5.drawcountries(linewidth=0.4)
    p5.drawcoastlines(linewidth=0.2,linestyle="--")
    p5.contourf(xgrid,ygrid,tpgmean.reshape(ngrid,ngrid),cmap="Blues",alpha=0.5)
    p5.colorbar(extend='max')
    ax5.set_title("\Phi(g) mean")

    # plot 06
    ax6 = fig.add_subplot(gs[1,5])
    p6 = deepcopy(pmap)
    p6.drawcountries(linewidth=0.4)
    p6.drawcoastlines(linewidth=0.2,linestyle="--")
    p6.contourf(xgrid,ygrid,tpgvar.reshape(ngrid,ngrid),cmap="Oranges",alpha=0.5)
    p6.colorbar(extend='max')
    ax6.set_title("\Phi(g) var")


    # plot 07
    ax7 = fig.add_subplot(gs[0,6])
    p7 = deepcopy(pmap)
    p7.drawcountries(linewidth=0.4)
    p7.drawcoastlines(linewidth=0.2,linestyle="--")
    p7.contourf(xgrid,ygrid,tgmean.reshape(ngrid,ngrid),cmap='RdYlGn',alpha=0.5)
    p7.colorbar(extend='max')
    ax7.set_title("g mean")


    # plot 08
    ax8 = fig.add_subplot(gs[1,6])
    p8 = deepcopy(pmap)
    p8.drawcountries(linewidth=0.4)
    p8.drawcoastlines(linewidth=0.2,linestyle="--")
    p8.contourf(xgrid,ygrid,tgvar.reshape(ngrid,ngrid),cmap='RdPu',alpha=0.5)
    p8.colorbar(extend='max')
    ax8.set_title("g var")


    fig.tight_layout()
    plt.show()


def PlotOnOffPrecipitationMap(plotdf,plotvar):
    mpl.rcParams['figure.figsize'] = (12,8)

    fig, ax = plt.subplots()
    m = Basemap(resolution='i', # c, l, i, h, f or None
                projection='merc',
                lat_0=65.0, lon_0=25.0,
                llcrnrlon=19.08, llcrnrlat= 59.45, urcrnrlon=31.59, urcrnrlat=70.09)
    m.drawcountries(linewidth=0.5)
    m.drawcoastlines(linewidth=0.2,linestyle="--")

    xco = []
    yco =[]
    for row in plotdf[['lat','lon']].itertuples():
        lat,lon = row[1],row[2]
        x, y = m(lon,lat)
        xco.append(x);yco.append(y)
    m.scatter(xco,yco,c=plotdf[plotvar].tolist(),alpha=0.5,s=30, marker='o')
    m.colorbar(extend='max')

    plt.show()


def PlotOnOffPrecipitationInducing(m1):

    mpl.rcParams['figure.figsize'] = (8,10)

    indfdf = pd.DataFrame({'lat':m1.Zf.value[:,0],
                           'lon':m1.Zf.value[:,1],
                           'ndatehour':m1.Zf.value[:,2],
                           'value':m1.u_fm.value.flatten()})
    indgdf = pd.DataFrame({'lat':m1.Zg.value[:,0],
                           'lon':m1.Zg.value[:,1],
                           'ndatehour':m1.Zg.value[:,2],
                           'value':m1.u_gm.value.flatten()})

    indfdf['value'] = indfdf['value']*50
    indgdf['value'] = indgdf['value']*50

    fig = plt.figure()
    gs = gridspec.GridSpec(1,2)

    pmap1 = Basemap(resolution='l', # c, l, i, h, f or None
            projection='merc',
            lat_0=65.0, lon_0=25.0,
            llcrnrlon=19.08, llcrnrlat= 59.45, urcrnrlon=31.59, urcrnrlat=70.09)

    xco = []
    yco =[]
    for row in indfdf[['lat','lon']].itertuples():
        lat,lon = row[1],row[2]
        x, y = pmap1(lon,lat)
        xco.append(x);yco.append(y)

    ax1 = fig.add_subplot(gs[0,0])
    pmap1.drawcountries(linewidth=0.4)
    pmap1.drawcoastlines(linewidth=0.2,linestyle="--")
    pmap1.scatter(xco,yco,c=indfdf['ndatehour'].tolist(),alpha=0.5
                  ,s= indfdf['value'].tolist()*100
                  ,marker='o')
    pmap1.colorbar(extend='max')
    ax1.set_title("ind f")
    ax1.legend()


    pmap2 = Basemap(resolution='l', # c, l, i, h, f or None
            projection='merc',
            lat_0=65.0, lon_0=25.0,
            llcrnrlon=19.08, llcrnrlat= 59.45, urcrnrlon=31.59, urcrnrlat=70.09)

    xco = []
    yco =[]
    for row in indgdf[['lat','lon']].itertuples():
        lat,lon = row[1],row[2]
        x, y = pmap2(lon,lat)
        xco.append(x);yco.append(y)


    ax2 = fig.add_subplot(gs[0,1])
    pmap2.drawcountries(linewidth=0.4)
    pmap2.drawcoastlines(linewidth=0.2,linestyle="--")
    pmap2.scatter(xco,yco,c=indgdf['ndatehour'].tolist(),alpha=0.5
                  ,s= indgdf['value'].tolist()
                  , marker='o')
    pmap2.colorbar(extend='max')
    ax2.set_title("ind g")
    ax2.legend()

    plt.tight_layout()
    plt.show()
