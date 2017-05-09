"""Make plots for displaying results of BEST test.

This module produces plots similar to those in

Kruschke, J. (2012) Bayesian estimation supersedes the t
    test. Journal of Experimental Psychology: General.
"""
from __future__ import division
import numpy as np
from best import calculate_sample_statistics

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import matplotlib.lines as mpllines
import matplotlib.ticker as mticker
from pymc.distributions import noncentral_t_like

pretty_blue = '#89d1ea'

def plot_posterior( sample_vec, bins=None, ax=None, title=None, stat='mode',
                    label='', draw_zero=False ):

    stats = calculate_sample_statistics( sample_vec )
    stat_val = stats[stat]
    hdi_min = stats['hdi_min']
    hdi_max = stats['hdi_max']

    if ax is not None:
        if bins is not None:
            kwargs = {'bins':bins}
        else:
            kwargs = {}
        ax.hist( sample_vec, rwidth=0.8,
                 facecolor=pretty_blue, edgecolor='none', **kwargs )
        if title is not None:
            ax.set_title( title )

        trans = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text( stat_val, 0.99, '%s = %.3g'%(stat,stat_val),
                 transform=trans,
                 horizontalalignment='center',
                 verticalalignment='top',
                 )
        if draw_zero:
            ax.axvline(0,linestyle=':')

        # plot HDI
        hdi_line, = ax.plot( [hdi_min, hdi_max], [0,0],
                             lw=5.0, color='k')
        hdi_line.set_clip_on(False)
        ax.text( hdi_min, 0.04, '%.3g'%hdi_min,
                 transform=trans,
                 horizontalalignment='center',
                 verticalalignment='bottom',
                 )
        ax.text( hdi_max, 0.04, '%.3g'%hdi_max,
                 transform=trans,
                 horizontalalignment='center',
                 verticalalignment='bottom',
                 )

        ax.text( (hdi_min+hdi_max)/2, 0.14, '95\% HDI',
                 transform=trans,
                 horizontalalignment='center',
                 verticalalignment='bottom',
                 )

        # make it pretty
        ax.spines['bottom'].set_position(('outward',2))
        for loc in ['left','top','right']:
            ax.spines[loc].set_color('none') # don't draw
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks([]) # don't draw
        ax.xaxis.set_major_locator( mticker.MaxNLocator(nbins=4) )
        for line in ax.get_xticklines():
            line.set_marker(mpllines.TICKDOWN)
        ax.set_xlabel(label)


def plot_data_and_prediction( data, means, stds, numos, ax=None, bins=None,
                              n_curves=50, group='x'):
    assert ax is not None

    ax.hist( data, bins=bins, rwidth=0.5,
             facecolor='r', edgecolor='none', normed=True )

    if bins is not None:
        if hasattr(bins,'__len__'):
            xmin = bins[0]
            xmax = bins[-1]
        else:
            xmin = np.min(data)
            xmax = np.max(data)

    n_samps = len(means)
    idxs = map(int,np.round( np.random.uniform(size=n_curves)*n_samps ))

    x = np.linspace(xmin, xmax, 100)
    ax.set_xlabel('y')
    ax.set_ylabel('p(y)')

    for i in idxs:
        m = means[i]
        s = stds[i]
        lam = 1/s**2
        numo = numos[i]
        nu = numo+1

        v = np.exp([noncentral_t_like(xi,m,lam,nu) for xi in x])
        ax.plot(x,v, color=pretty_blue, zorder=-10)

    ax.text(0.95,0.95,'$\mathrm{N}_{%s}=%d$'%( group, len(data), ),
            transform=ax.transAxes,
            horizontalalignment='right',
            verticalalignment='top'
            )
    ax.xaxis.set_major_locator( mticker.MaxNLocator(nbins=4) )
    ax.yaxis.set_major_locator( mticker.MaxNLocator(nbins=4) )
    ax.set_title('Data Group %s w. Post. Pred.'%(group,))


def make_figure(M, n_bins=30):
    # plotting stuff
    # ToDo: Enable separate normality plotting!

    param_names = []

    for node in M.nodes:
        param_names += [node.__name__]

    for key in param_names:
        if key[:2] == "1~":
            group1_name = key[2:]
        if key[:2] == "2~":
            group2_name = key[2:]

    group1_data = M.get_node("1~" + group1_name).value
    group2_data = M.get_node("2~" + group2_name).value

    N1 = len(group1_data)
    N2 = len(group2_data)

    if "nu1_minus_one" in param_names:
        separate_nu = True
        pn = 6
    else:
        separate_nu = False
        pn = 5

    posterior_mean1 = M.trace('group1_mean')[:]
    posterior_mean2 = M.trace('group2_mean')[:]
    diff_means = posterior_mean1 - posterior_mean2

    posterior_means = np.concatenate( (posterior_mean1,posterior_mean2) )
    _, bin_edges_means = np.histogram( posterior_means, bins=n_bins )

    posterior_std1 = M.trace('group1_std')[:]
    posterior_std2 = M.trace('group2_std')[:]
    diff_stds = posterior_std1 - posterior_std2

    posterior_stds = np.concatenate( (posterior_std1,posterior_std2) )
    _, bin_edges_stds = np.histogram( posterior_stds, bins=n_bins )

    pooled_var = ((N1-1)*posterior_std1**2 + (N2-1)*posterior_std2**2) / (N1+N2-2)
    effect_size = diff_means / np.sqrt(pooled_var)

    if separate_nu:
        post_nu1_minus_one = M.trace('nu1_minus_one')[:]
        post_nu2_minus_one = M.trace('nu2_minus_one')[:]
        lognu1p = np.log10(post_nu1_minus_one + 1)
        lognu2p = np.log10(post_nu2_minus_one + 1)
        diff_lognup = lognu1p - lognu2p
        posterior_nus = np.concatenate((lognu1p, lognu2p))
        _, bin_edges_nus = np.histogram(posterior_nus, bins=n_bins)
    else:
        post_nu1_minus_one = M.trace('nu_minus_one')[:]
        post_nu2_minus_one = post_nu1_minus_one
        lognu1p = np.log10(post_nu1_minus_one+1)
        bin_edges_nus = n_bins

    f = plt.figure(figsize=(8.2,1 + 2*pn),facecolor='white')
    ax1 = f.add_subplot(pn,2,1,axisbg='none')
    plot_posterior( posterior_mean1, bins=bin_edges_means, ax=ax1,
                    title='%s Mean' % group1_name, stat='mean',
                    label=r'$\mu_1$')

    ax3 = f.add_subplot(pn,2,3,axisbg='none')
    plot_posterior( posterior_mean2, bins=bin_edges_means, ax=ax3,
                    title='%s Mean' % group2_name, stat='mean',
                    label=r'$\mu_2$')

    ax5 = f.add_subplot(pn,2,5,axisbg='none')
    plot_posterior( posterior_std1, bins=bin_edges_stds, ax=ax5,
                    title='%s Std. Dev.' % group1_name,
                    label=r'$\sigma_1$')

    ax7 = f.add_subplot(pn,2,7,axisbg='none')
    plot_posterior( posterior_std2, bins=bin_edges_stds, ax=ax7,
                    title='%s Std. Dev.' % group2_name,
                    label=r'$\sigma_2$')

    ax9 = f.add_subplot(pn, 2, 9, axisbg='none')
    plot_posterior(lognu1p, bins=bin_edges_nus, ax=ax9,
                   title='{} Normality'
                   .format(group1_name if separate_nu else ''),
                   label=r'$\mathrm{}(\nu{})$'
                   .format('{log10}', '_1' if separate_nu else ''))

    if separate_nu:
        ax11 = f.add_subplot(pn, 2, 11, axisbg='none')
        plot_posterior(lognu2p, bins=bin_edges_nus, ax=ax11,
                       title='{} Normality'.format(group2_name),
                       label=r'$\mathrm{log10}(\nu_2)$')

        ax12 = f.add_subplot(pn, 2, 12, axisbg='none')
        plot_posterior(diff_lognup, bins=n_bins, ax=ax12,
                       title='Normality Difference',
                       draw_zero=True,
                       label=r'$\mathrm{log10}(\nu_1)-\mathrm{log10}(\nu_2)$')

    ax6 = f.add_subplot(pn,2,6,axisbg='none')
    plot_posterior( diff_means, bins=n_bins, ax=ax6,
                    title='Difference of Means',
                    stat='mean',
                    draw_zero=True,
                    label=r'$\mu_1 - \mu_2$')

    ax8 = f.add_subplot(pn,2,8,axisbg='none')
    plot_posterior( diff_stds, bins=n_bins, ax=ax8,
                    title='Difference of Std. Dev.s',
                    draw_zero=True,
                    label=r'$\sigma_1 - \sigma_2$')

    ax10 = f.add_subplot(pn,2,10,axisbg='none')
    plot_posterior( effect_size, bins=n_bins, ax=ax10,
                    title='Effect Size',
                    draw_zero=True,
                    label=r'$(\mu_1 - \mu_2)/\sqrt{((N_1-1)\sigma_1^2 + (N_2-1)\sigma_2^2)/(N_1+N_2-2)}$')

    orig_vals = np.concatenate( (group1_data, group2_data) )
    bin_edges = np.linspace( np.min(orig_vals), np.max(orig_vals), 30 )

    ax2 = f.add_subplot(pn,2,2,axisbg='none')
    plot_data_and_prediction(group1_data, posterior_mean1, posterior_std1,
                             post_nu1_minus_one, ax=ax2, bins=bin_edges, group=group1_name)

    ax4 = f.add_subplot(pn,2,4,axisbg='none',sharex=ax2,sharey=ax2)
    plot_data_and_prediction(group2_data, posterior_mean2, posterior_std2,
                             post_nu2_minus_one, ax=ax4, bins=bin_edges, group=group2_name)

    f.subplots_adjust(hspace=0.82,top=0.97,bottom=0.06,
                      left=0.09, right=0.95, wspace=0.45)

    return f

