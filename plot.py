import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import matplotlib
import pandas as pd
import os
import argparse
from cep_process import extract_cep_data
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.basemap import Basemap


def plot_brazil_dist(df, figsize):
    df = df[(df['country'] == 'Brasil')]
    df['count'] = pd.to_numeric(df['count'], errors='coerce')
    result_df = {}
    states = df['state'].value_counts().axes[0]
    for state in states:
        count = df['count'][(df['state'] == state)].sum()
        result_df[state] = count

    labels = list(result_df.keys())
    sizes = list(result_df.values())
    plt.figure(1)
    ax1 = plt.subplot(111)
    wedges, _, _ = ax1.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        textprops=dict(color="w"),
    )
    ax1.axis(
        'equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.legend(wedges, labels, title="Groups", loc="center left")

    plt.savefig('power_level.png', format="png", dpi=300, bbox_inches="tight")
    plt.close()

    # Bar plot
    total = np.sum(sizes)
    ind = np.arange(len(labels))
    plt.figure(1)
    ax1 = plt.subplot(111)
    p_array = []
    for i, key in enumerate(list(result_df.keys())):
        p = ax1.bar(ind[i], result_df[key] * 100 / total)
        p_array.append(p)

    plt.ylabel('Porcentagem')
    plt.xticks(ind, ['' for i in ind])
    plt.legend(p_array, labels)
    plt.savefig('power_level_bar.png',
                format="png",
                dpi=300,
                bbox_inches="tight")
    plt.close()
    percent = [i * 100 / total for i in sizes]
    df = {'States': labels, 'Count': sizes, 'Percent': percent}
    # print(df)
    df = pd.DataFrame(df)
    df.to_csv('cep_distribution.csv')


def cluster_per_state(df):
    # Get the states
    states = df['state'].value_counts().axes[0]
    result_df = {'state': [], 'lat': [], 'lon': [], 'count': []}
    # Get the states locations
    for state in states:
        address = f"{state}, Brasil"
        _, _, _, location = extract_cep_data(address)
        result_df['state'].append(state)
        result_df['lat'].append(location.latitude)
        result_df['lon'].append(location.longitude)
        result_df['count'].append(df['count'][(df['state'] == state)].sum())
    result_df = pd.DataFrame(result_df)
    return result_df


def make_map(mundi):
    if mundi:
        m = Basemap(projection='merc',
                    llcrnrlat=-60,
                    urcrnrlat=65,
                    llcrnrlon=-180,
                    urcrnrlon=180,
                    lat_ts=0,
                    resolution='c')
    else:
        m = Basemap(
            # projection='merc',
            llcrnrlat=-34,
            urcrnrlat=6,
            llcrnrlon=-75,
            urcrnrlon=-34,
            resolution='i')

        m.readshapefile('gadm36_BRA_shp/gadm36_BRA_1',
                        'gadm36_BRA_1',
                        drawbounds=True)
    # m.fillcontinents()
    # m.drawcountries()   # thin white line for country borders
    # m.fillcontinents()  # dark grey land, black lakes
    # m.drawmapboundary() # black background
    # m.drawcountries(linewidth=0.1,
    #                 color="k")
    return m


def plot_geo_data(df, mundi=False):
    # Get the latitude and longitude data
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df.dropna()

    if mundi is False:
        # Filter to only Brazil data
        df = df[(df['country'] == 'Brasil')]
        df = cluster_per_state(df)
    # Begin the figure
    fig = plt.figure(1, figsize=(12, 6))
    ax = fig.add_subplot(111)
    # Get the map
    m = make_map(mundi)

    # Plot the data
    mxy = m(df["lon"].tolist(), df["lat"].tolist())
    if mundi:
        m.scatter(mxy[0], mxy[1], s=3, c="r", lw=0, alpha=1, zorder=5)
    else:
        area = df['count'].values
        plt.scatter(
            mxy[0],
            mxy[1],
            c=area,
            s=area * 3.5,
            #   cmap='Reds',
            lw=0,
            alpha=0.5,
            zorder=5)
        shape_dict = {}

        plt.colorbar(label='Total de participantes')
        plt.clim()
        for c, s in zip(df['count'].values, df['state'].values):
            shape_dict[s] = c
        patches = []
        total = np.amax(area)
        cmap = matplotlib.cm.get_cmap()
        color = []
        for info, shape in zip(m.gadm36_BRA_1_info, m.gadm36_BRA_1):
            if info['NAME_1'] in df['state'].values:
                patches.append(Polygon(np.array(shape), True))
                color.append(cmap(shape_dict[info['NAME_1']] / total))
        ax.add_collection(
            PatchCollection(patches,
                            facecolor=['#d5d6d6'],
                            edgecolor=['k'],
                            linewidths=1.,
                            zorder=2))

    map_name = "cep_mundi.png" if mundi else "cep.png"
    total = np.amax(area)
    for a in [10, 200, total]:

        plt.scatter([], [], c='k', alpha=0.5, s=a * 3.5, label=str(a))
        plt.legend(scatterpoints=1,
                   frameon=False,
                   labelspacing=3,
                   loc='lower left',
                   bbox_to_anchor=(0.05, 0.05, 0, 0))
    plt.savefig(map_name, format="png", dpi=300, bbox_inches="tight")
    plt.close()


def main():

    # Get the dataset
    df = pd.read_csv('processed_ceps.csv')

    # Update the Rc params to generate a paper quality plot
    aspect_ratio = 1.4
    max_width = 3.39
    # params = {
    #     "text.usetex": True,
    #     "axes.labelsize": 8,
    #     "axes.titlesize": 8,
    #     "font.size": 8,
    #     "legend.fontsize": 6,
    #     "legend.edgecolor": 'black',
    #     "xtick.labelsize": 8,
    #     "ytick.labelsize": 8,
    #     "grid.color": "k",
    #     "grid.linestyle": ":",
    #     "grid.linewidth": 0.5,
    # }
    # matplotlib.rcParams.update(params)

    figsize = (max_width, max_width / aspect_ratio)

    # plot_brazil_dist(df, figsize)
    plot_geo_data(df)


if __name__ == "__main__":
    main()
