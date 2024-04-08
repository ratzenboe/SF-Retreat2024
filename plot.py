import jax.numpy as np
import nifty8.re as jft
import matplotlib.pyplot as pl
import cmasher as cm


def plot1d(data_path, name, grid, n_pix, res, data_position, data, std, samples, stellar_age_model, stellar_age_truth, mean_model, mean_truth, std_model, std_truth, cluster_mean, cluster_mean_truth, cluster_std, cluster_std_truth, bg_mean, bg_mean_truth, bg_std, bg_std_truth):
    n_data = len(data)
    
    stellar_age_mean  = jft.mean(tuple(stellar_age_model(s) for s in samples))
    
    age_mean_samples = [mean_model(s) for s in samples]
    age_mean_mean = jft.mean(age_mean_samples)
    
    age_std_samples = [std_model(s) for s in samples]
    age_std_mean = jft.mean(age_std_samples)

    mean_field_mean = jft.mean(tuple(mean_model(s) for s in samples))
    std_field_mean = jft.mean(tuple(std_model(s) for s in samples))

    position_grid = np.linspace(0, n_pix*res, n_pix)

    fig, axes = pl.subplots(ncols=2, nrows=2, figsize=(6,5))

    axes[0, 0].plot(position_grid, age_mean_mean, color='blue', label='mean')
    for s in age_mean_samples:
        axes[0, 0].plot(position_grid, s, color='lightblue', alpha=0.6 )
    if mean_truth is not None: 
        axes[0, 0].plot(position_grid, mean_truth, color='blue', label='truth')      
    axes[0, 0].set_ylabel(r"$\ln(\mathrm{Age})$")
    axes[0, 0].set_xlabel('position')
    axes[0, 0].set_title('Age mean field')
    
    axes[0, 1].plot(position_grid, age_std_mean, color='blue', label='mean')
    for s in age_std_samples:
        axes[0, 1].plot(position_grid, s, color='lightblue', alpha=0.6 )
    if mean_truth is not None: 
        axes[0, 0].plot(position_grid, std_truth, color='blue', label='truth')      
    axes[0, 0].set_ylabel(r"$\ln(\mathrm{Age})$")
    axes[0, 0].set_xlabel('position')
    axes[0, 0].set_title('Age std field')
    
    axes[1, 1].plot(quantile_grid, post_ppf_mean)
    axes[1, 1].set_xlabel('quantiles')
    axes[1, 1].set_ylabel('x')
    axes[1, 1].set_title('ppf')
    
    axes[1, 0].plot(quantile_grid, post_qdf_mean)
    for s in samples:
        axes[1, 0].plot(quantile_grid, qdf_model(s), color='lightblue', alpha=0.6)
    axes[1, 0].set_xlabel('quantiles')
    axes[1, 0].set_ylabel(r"$\Delta x$")
    axes[1, 0].set_title('qdf')
    
    y_vol = true_pdf.max() - true_pdf.min()
    
    for s in samples:
        axes[0, 1].plot(grid, transform_ppf_to_pdf(ppf_model(s), quantile_grid, grid), color='lightblue', alpha=0.6)
    axes[0, 1].plot(grid, transform_ppf_to_pdf(post_ppf_mean, quantile_grid, grid), label="inferred pdf", color='blue')
    axes[0, 1].plot(grid, true_pdf, label="true pdf", color='red')
    axes[0, 1].plot(grid, data_kernel, label="data kde", color='green')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('pdf')
    axes[0, 1].set_ylim([0, y_vol*1.7])
    axes[0, 1].set_title('true pdf, pdf estimate')
    axes[0, 1].legend()
    # axes[1].plot(grid, qdf(position).val, label="qdf")
    
    fig.tight_layout()
    fig.savefig(data_path + 'ages_{}.png'.format(name), dpi = 100, format="png")


    fig, axes = pl.subplots(ncols=2, nrows=2, figsize=(6,5))
    
    extent = [np.floor(data.min()), np.ceil(data.max())]

    axes[0, 0].scatter(truth, post_ppf_mean, marker='+')
    axes[0, 0].set_xlabel('truth, x')
    axes[0, 0].set_ylabel('pred, x')
    axes[0, 0].set_xlim(extent)
    axes[0, 0].set_ylim(extent)
    axes[0, 0].set_title('truth vs prediction')

    
    axes[0, 1].scatter(data - truth, post_ppf_mean - truth, marker='+')
    axes[0, 1].set_xlabel('data - truth, x')
    axes[0, 1].set_ylabel('pred - truth, x')
    axes[0, 1].set_xlim(extent)
    axes[0, 1].set_ylim(extent)
    axes[0, 1].set_title('residuals')

    axes[1, 0].scatter(data, post_ppf_mean, marker='+')
    axes[1, 0].set_xlabel('data, x')
    axes[1, 0].set_ylabel('pred, x')
    axes[1, 0].set_xlim(extent)
    axes[1, 0].set_ylim(extent)
    axes[1, 0].set_title('data vs prediction')
    
    axes[1, 1].scatter(truth, data, marker='+')
    axes[1, 1].set_xlabel('truth, x')
    axes[1, 1].set_ylabel('data, x')
    axes[1, 1].set_xlim(extent)
    axes[1, 1].set_ylim(extent)
    axes[1, 1].set_title('truth vs data')

    fig.tight_layout()
    fig.savefig(data_path + 'oned_scatter_{}.png'.format(name), dpi = 100, format="png")
    

    fig, axes = pl.subplots(ncols=2, nrows=2, figsize=(6,5))

    axes[0, 0].hist(truth, bins=int(n_data/10))
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel(r"\#")
    axes[0, 0].set_title('truth')
    
    axes[0, 1].hist(data, bins=int(n_data/10))
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel(r"\#")
    axes[0, 1].set_title('data')

    axes[1, 0].hist(post_ppf_mean, bins=int(n_data/10))
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel(r"\#")
    axes[1, 0].set_title('posterior mean')
    
    axes[1, 1].hist((data-truth)/std, bins=int(n_data/10))
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel(r"\#")
    axes[1, 1].set_title('noise weighted residual')

    fig.tight_layout()
    fig.savefig(data_path + 'oned_histogram_{}.png'.format(name), dpi = 100, format="png")
                  