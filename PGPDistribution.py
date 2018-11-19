import networkx as nx
import matplotlib
matplotlib.use("TkAgg") #Hack to ensure it works in virtual env.
import matplotlib.pyplot as plt
import numpy as np
import collections
from collections import Counter
from scipy.optimize import curve_fit
from scipy.special import factorial
from scipy import optimize
from scipy import stats
from array import array

# Define function for calculating a power law
powerlaw = lambda x, amp, index: amp * (x**index)
# define our (line) fitting function
fitfunc = lambda p, x: p[0] + p[1] * x
errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err

def drop_zeros(a_list):
    return [i for i in a_list if i>0]

def log_binning(counter_dict, bin_count=35):
    keys = list(counter_dict.keys())
    vals = list(counter_dict.keys())
    max_x = np.log10(max(keys))
    max_y = np.log10(max(vals))
    max_base = max([max_x,max_y])

  
    keys_without_zeroes = drop_zeros(keys)
    min_x = np.log10(min(keys_without_zeroes))

    bins = np.logspace(min_x,max_base,num=bin_count)

    # Based off of: http://stackoverflow.com/questions/6163334/binning-data-in-python-with-scipy-numpy
    bin_means_y = (np.histogram(keys,bins,weights=vals)[0] / np.histogram(keys,bins)[0])
    bin_means_x = (np.histogram(keys,bins,weights=keys)[0] / np.histogram(keys,bins)[0])

    return bin_means_x,bin_means_y

def plot_degree_histogram(G): 
    degree_centrality = nx.degree_centrality(G)
    centrality_vals = list(degree_centrality.values())
    centrality_keys = list(degree_centrality.keys())
    ba_c2 = dict(Counter(centrality_vals))

    ba_x,ba_y = log_binning(ba_c2,50)

    plt.xscale('log')
    plt.yscale('log')
    plt.scatter(ba_x,ba_y,c='r',marker='s',s=50)
    plt.scatter(centrality_keys,centrality_vals,c='b',marker='x')
    # plt.xlim((1e-4,1e-1))
    # plt.ylim((.9,1e4))
    plt.xlabel('Connections (normalized)')
    plt.ylabel('Frequency')
    plt.show()

def plot_degree_histogram_std(G): 
    degree_sequence = sorted([degree for node, degree in G.degree()], reverse=True) 
    degree_count = collections.Counter(degree_sequence)
    deg, cnt = zip(*degree_count.items())

    plt.xscale('log')
    plt.yscale('log')
    plt.scatter(deg,cnt,c='b',marker='.',s=15)

    # powerlaw
    params =  stats.powerlaw.fit(cnt)
    f = stats.powerlaw.freeze(params[0], params[1], params[2])
    x = np.linspace(f.ppf(0.001), f.ppf(0.999), 100)
    pdf = f.pdf(x)
    plt.plot(x, pdf, 'r', lw=1, label="powerlaw")
    # a = 3
    # num_points =  500 #len(cnt)
    # rv = stats.powerlaw(a)
    # start_ppf = rv.ppf(0.001)
    # stop_ppf = rv.ppf(0.999)
    # x = np.linspace(start_ppf, stop_ppf, num_points)
    # pdf = rv.pdf(x) #return probability density
    # plt.plot(x, pdf, 'r', lw=5, alpha=1, label='powerlaw pdf')

    #plt.scatter(x,stats.powerlaw.pdf(x, a),c='r',marker='x',s=15)
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.show()

def plot(data):
    params =  stats.powerlaw.fit(data)
    f = eval("stats.powerlaw.freeze"+str(params))
    x = np.linspace(f.ppf(0.001), f.ppf(0.999), 500)
    norm3 = x / np.linalg.norm(x)
    pdf = f.pdf(x)

    bin_size = int(max(50, len(data)/10))
    plt.hist(pdf, density=True, bins=bin_size)

   # plt.plot(norm3, pdf, label="powerlaw")
    #plt.legend(loc='best', frameon=False)
    plt.title("Powerlaw fit")
    plt.show()

# The Kolmogorov-Smirnov test can be applied more broadly than Shapiro, since it is comparing any two distributions 
# against each other, not necessarily one distriubtion to a normal one. These tests can be one-sided or both-sides, 
# but the latter only applies if both distributions are continuous.

# the variable must be continuous
# not suited for large samples, since also small deviations can be significant
def kolmogorov_smirnov_fit(G):
    num_nodes = G.number_of_nodes()

    # sorted sequence of degress [205, 70, 5, 5, 1]
    # http://www.aizac.info/simple-check-of-a-sample-against-80-distributions/
    degree_sequence = sorted([degree for node, degree in G.degree()], reverse=True) 
    degree_count = collections.Counter(degree_sequence)
    deg, cnt = zip(*degree_count.items())
    params =  stats.powerlaw.fit(cnt)
    D, pval = stats.kstest(cnt, "powerlaw", args=params,  N=num_nodes)
    plot(cnt)
    #significant divergence of the tested distribution if p < .05
    print(D)
    print(pval)


def test_degree_distribution_for_powerlaw(G):
    num_nodes = G.number_of_nodes()

    # sorted sequence of degress [205, 70, 5, 5, 1]
    degree_sequence = sorted([degree for node, degree in G.degree()], reverse=True) 
    degree_count = collections.Counter(degree_sequence)
    deg, cnt = zip(*degree_count.items())

    # TODO determine y-exponent
    xdata = np.linspace(1, 10, len(cnt))
    ydata = powerlaw(xdata, 10, -2)

    print(cnt)
    print(ydata)
    #plot_degree_histogram(ydata, cnt)
    ##########
    # Fitting the data -- Least Squares Method
    ##########

    # Power-law fitting is best done by first converting
    # to a linear equation and then fitting to a straight line.
    # Note that the `logyerr` term here is ignoring a constant prefactor.
    #
    #  y = a * x^b
    #  log(y) = log(a) + b*log(x)
    logx = np.log10(xdata)
    logy = np.log10(ydata)
    logyerr = cnt / ydata

    pinit = [1.0, -1.0]
    out = optimize.leastsq(errfunc, pinit,
                       args=(logx, logy, logyerr), full_output=1)
    pfinal = out[0]
    covar = out[1]

    print(pfinal)
    print(covar)

    index = pfinal[1]
    amp = 10.0**pfinal[0]

    indexErr = np.sqrt(covar[1][1] )
    ampErr = np.sqrt(covar[0][0] ) * amp

    ##########
    # Plotting data
    ##########

    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(xdata, powerlaw(xdata, amp, index))     # Fit
    plt.errorbar(xdata, ydata, yerr=cnt, fmt='k.')  # Data
    plt.text(5, 6.5, 'Ampli = %5.2f +/- %5.2f' % (amp, ampErr))
    plt.text(5, 5.5, 'Index = %5.2f +/- %5.2f' % (index, indexErr))
    plt.title('Best Fit Power Law')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(1, 11)

    plt.subplot(2, 1, 2)
    plt.loglog(xdata, powerlaw(xdata, amp, index))
    plt.errorbar(xdata, ydata, yerr=cnt, fmt='k.')  # Data
    plt.xlabel('X (log scale)')
    plt.ylabel('Y (log scale)')
    plt.xlim(1.0, 11)
    plt.show()

A = np.loadtxt('Data/arenas-pgp/out.arenas-pgp', dtype=int, usecols=range(2), comments="%")
G=nx.Graph()
for n in A:
    G.add_edge(n[0], n[1])

#kolmogorov_smirnov_fit(G)
plot_degree_histogram_std(G)