import time as tsys
import argparse

import numpy as np
import scipy.stats


def powerlaw(x, a, xmin):
    """
    power law probability distribution
    """
    prefac = (a-1.)/xmin
    rest = (x/xmin)**a
    return prefac*rest


def from_uniform(r, a, xmin):
    """
    Inverse CDF of the power law distribution
    """
    x = xmin*(1.-r)**(-1./(a-1.))
    return x


def make_eventfile(time, cr, a, xmin=0.2):
    """
    Take the estimated count rate in each time bin 
    and randomly distribute events of energy E in each 
    time bin.
    The energy for each event is sampled from the power law 
    distribution.
    """
    # empty arrays for the photon arrival times and energies
    toa, energies = [], []
    # iterate over each bin in the time and count rate arrays
    for i,(t,c) in enumerate(zip(time, cr)):
        # if the count rate is zero, leave the bin empty
        if c == 0:
            t = []
            e = []
        else:
            # if the count rate is not zero, sample arrival times 
            # from a uniform distribution and energies from the 
            # power law
            t = np.random.uniform(t, time[i+1], size=c)
            r = np.random.uniform(size=c)
            e = from_uniform(r, a, xmin)
        # append new values to array
        toa.append(t)
        energies.append(e)
    return toa, energies

def simulate_pileup(evts):
    """
    This function simulates a piled-up data set.
    """
    # empty arrays for the new counts and new energies
    new_counts = []
    new_energies = []
    # loop over all events
    for e in evts:
        # if there are no counts in the array, leave it empty
        if len(e) == 0:
            new_counts.append(0)
            new_energies.append(0.0)
        else:
            # compute the sum of energies of photons in this bin 
            esum = np.sum(e)
            # if this sum is larger than 10 keV, drop the photons
            if esum > 10.0:
                new_counts.append(0.0)
            # if not, then report a single photon with the combined energy
            else:
                new_counts.append(1)
                new_energies.append(esum)
            
    return new_counts, new_energies


def make_data():
    # lower and upper bounds for the energy bins
    emin = 0.2
    emax = 10.0

    # the energy bins
    energy = np.linspace(emin, emax, 1000)


    # Now we need to make a time series
    # length of the segment
    tseg = 25000.0 
    # length of each frame
    dt = 2.3
    
    # this is the number of bins in the light curve
    bins = int(tseg/dt)
    print("The number of bins in the light curve is: " + str(bins))
    
    # make an array of time bins
    time = np.arange(0.0, tseg, dt)
    
    # the input count rate: 
    cr_real = 0.5
    
    # let's produce some data via the poisson distribution
    cr_data = np.random.poisson(cr_real, size=bins)
    
    # this is the assumed true power law index in the data
    a_real = 2.5
 
    # let's make an event file we'll assume to be an actual observation
    toa, evts = make_eventfile(time, cr_data, a_real)

    # now we can make a piled-up energy array
    pu_counts, pu_energies = simulate_pileup(evts)    
    pu_energies = np.array(pu_energies)
    
    # let's compute the pile-up fraction in the data
    counter = 0
    for e in evts:
        if len(e) > 1:
            counter += 1

    print("The pile-up fraction is %.4f"%(counter/float(len(evts))))

    return time, a_real, cr_real, pu_energies


def ks_metric(data, sim):
    """
    Metric based on a 2-sample KS-test between the distribution of 
    energies in the data and the simulations

    Note: make sure to remove all the zeros from the energy array before 
    using this metric!
    """
    d, p = scipy.stats.ks_2samp(data, sim)
    return d, p

def l2_distance(data, sim):
    """
    L2 (Euclidean) distance between a histogram of the data and 
    the model
  
    Note: Be sure to actually make a histogram before sticking it into this 
    function!  
    """ 
    m = np.sqrt(np.sum((data-sim)**2.))
    return m

def datamodel_residuals(x, xedges, logdata, logsim, sim_pars, model):
    """
    Metric based on the data-model residuals for a power-law model.
    CURRENTLY NOT USABLE!  

    """ 
    mean_model = model(x, *sim_pars)
    
    sim_hist, be = np.histogram(logdata, bins=xedges)
    data_hist, be = np.histogram(logsim, bins=xedges,)
    
    sim_res = np.sum((sim_hist-mean_model)**2.)
    data_res = np.sum((data_hist-mean_model)**2.)
    
    m = np.sqrt(np.sum((data-sim)**2.))
    
    return m

def simulate_dataset(x, a, log_cr):
    """
    Function to simulate some data sets

    Parameters
    ----------
    x : numpy.ndarray
        The time array

    a : float
        The power law index 

    log_cr : float
        The count rate in counts/s
    """
    # first, figure out count rate
    nbins = len(x)-1
    counts = np.random.poisson(np.exp(log_cr), size=nbins)
    
    # next, give photons energies:
    toa, evts = make_eventfile(x, counts, a)
    
    # now we can simulate the piled-up observation:
    new_counts, new_energies = simulate_pileup(evts)
    
    return np.array(new_counts), np.array(new_energies)

def from_prior():
    """ 
    Sample from the prior for the power law index and 
    the logarithm of the count rate.

    There's a log-uniform prior on the count rate, and a 
    Laplacian prior on the power law index
    """
    a = np.random.laplace(loc=2.5, scale=0.5, size=1)
    cr = np.random.uniform(-2, 1.5, size=1)
    return a, cr
    
def logprior(a, log_cr):
    """
    Compute the log-prior for the power-law index 
    and the log-countrate.
    """

    # prior on the power law index
    pr_a = ((a >= 0.0) & (a < 4.0))
    
    # prior on the total number of counts
    pr_logcr = ((log_cr >= -2.0) & (log_cr < 1.5))
    
    return np.log(pr_a) + np.log(pr_logcr)

def sample_from_ground_truth(time, a_real, cr_real, pu_energies, nsim=1000, 
                             emin = 0.2, emax = 10.0):
    """
    Sample data sets from the ground truth and compute the typical distributions of
    metrics expected from that ground truth.

    Parameters
    ----------
    time : numpy.ndarray
        The time array

    a_real : float
        The ground truth power law index
  
    cr_real : float
        The ground truth count rate

    pu_energies : numpy.ndarray
        List of piled-up photon energies

    nsim : int, default = 500
        The number of simulations to run

    emin : float 
        The lower bound of the energy range

    emax : float 
        The upper bound of the energy range
    """
    # the empty arrays for the metrics
    d_all_real, p_all_real, m2_all_real = [], [], [] 

    # make an array that contains only the non-zero energies
    pue_nonzero = pu_energies[pu_energies>0.0]

    # make a histogram of the energies in the data    
    data_hist, data_bins = np.histogram(np.log(np.hstack(pue_nonzero)), bins=100, 
                                        range=[emin, emax])
    
    # loop over the number of simulations
    for i in range(nsim):
        # simulate a data set with the ground truth index and log-count rate
        sim_counts, sim_energies = simulate_dataset(time, a_real, np.log(cr_real))

        # make an array of only non-zero simulated energies
        sim_en_nonzero = sim_energies[sim_energies > 0.0]

        # compute the KS-test metric
        d,p = ks_metric(pue_nonzero, sim_en_nonzero)
        
        # make a histogram of the simulated data
        sim_hist, sim_bins = np.histogram(np.log(np.hstack(sim_en_nonzero)), bins=100,
                                          range=[emin, emax])
        # compute the L2-metric
        m2 = l2_distance(data_hist, sim_hist)
    
        # append metrics to list
        d_all_real.append(d)
        p_all_real.append(p)
        m2_all_real.append(m2)
    
    d_all_real = np.array(d_all_real)
    p_all_real = np.array(p_all_real)
    m2_all_real = np.array(m2_all_real)

    return d_all_real, p_all_real, m2_all_real

def sample(time, a_real, cr_real, pu_energies, nsim=1000,
                             emin = 0.2, emax = 10.0):

    """
    Perform rejection sampling on the ABC problem:
        * Pick a parameter set from the prior
        * compute a piled-up data set from that parameter set
        * compute metrics to measure distance between real data set 
          and the simulated one
        * Return metrics and parameter sets; we'll do the actual rejection 
          step elsewhere.


    Parameters
    ----------
    time : numpy.ndarray
        The time array

    a_real : float
        The ground truth power law index
  
    cr_real : float
        The ground truth count rate

    pu_energies : numpy.ndarray
        List of piled-up photon energies

    nsim : int, default = 500
        The number of simulations to run

    emin : float 
        The lower bound of the energy range

    emax : float 
        The upper bound of the energy range
    """

    # empty arrays for metrics and parameters    
    d_all,p_all, m2_all = [], [], []
    new_pars = []
   
    # make an array that contains only the non-zero energies
    pue_nonzero = pu_energies[pu_energies>0.0]

    # make a histogram of the energies in the data    
    data_hist, data_bins = np.histogram(np.log(np.hstack(pue_nonzero)), bins=100,
                                        range=[emin, emax])
 
    for i in range(nsim):
        # get a parameter set from the prior
        a, cr = from_prior()
        
        # simulate a data set
        sim_counts, sim_energies = simulate_dataset(time, a, cr)

        # get non-zero energies
        sim_en_nonzero = sim_energies[sim_energies > 0.0]
        sim_hist, sim_bins = np.histogram(np.log(np.hstack(sim_en_nonzero)), bins=100,
                                          range=[emin, emax])
       
        # compute distance metrics 
        m2 = l2_distance(data_hist, sim_hist)
    
     
        d,p = ks_metric(pue_nonzero, sim_energies[sim_energies>0.0])
        d_all.append(d)
        p_all.append(p)
        m2_all.append(m2)
        
        new_pars.append([a[0], cr[0]])

    new_pars = np.array(new_pars)
    d_all = np.array(d_all)
    p_all = np.array(p_all)
    m2_all = np.array(m2_all)
    
    return new_pars, d_all, p_all, m2_all

def main():

    tstart = tsys.clock()
 
    # make a data set
    time, a_real, cr_real, pu_energies = make_data()

    if clargs.real is True:
        d_all_r, p_all_r, m2_all_r = sample_from_ground_truth(time, a_real, cr_real, pu_energies,
                                                              nsim=clargs.nsim)

        df = np.vstack([d_all_r, p_all_r, m2_all_r]).T
        np.savetxt(clargs.outroot + "_real.txt", df)
    else:
        new_pars, d_all, p_all, m2_all = sample(time, a_real, cr_real, pu_energies,
                                                nsim = clargs.nsim)

        df = np.vstack([new_pars.T, d_all, p_all, m2_all]).T
        np.savetxt(clargs.outroot + "_sim.txt", df)

    tend = tsys.clock()

    print("total runtime: " + str(tend - tstart))

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Toy simulations for pile-up proposal.")

    parser.add_argument("-o", "--outroot", action="store", required=False,
                        dest="outroot", default="test",
                        help = "root for the output file name.")

    parser.add_argument("-n", "--nsim", action="store", required=False, 
                        dest="nsim", default=1000, type=int, 
                        help = "number of simulations to run.")

    parser.add_argument("-r", "--real", action="store", required=False,
                        dest="real", default=False, type=bool,
                        help = "Run on the ground truth parameters or sample from prior?")

    clargs = parser.parse_args()

    main()


