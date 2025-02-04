import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from ....utils.plotting import set_style

def plot_diffusivity_profile(diffusivity_profile: pd.DataFrame,
                             ax: Optional[plt.Axes] = None,
                             save_path: Optional[str] = None,
                             display: bool = True) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot diffusivity profile evolution
    
    Parameters
    ----------
    diffusivity_profile : pd.DataFrame
        Diffusivity profile data
    ax : plt.Axes, optional
        Matplotlib axes for plotting
    save_path : str, optional
        Path to save the figure
    display : bool, optional
        Whether to display the plot (default: True)
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    set_style()
    
    times = diffusivity_profile.index
    positions = [float(col.split('=')[1]) for col in diffusivity_profile.columns]
    
    X, Y = np.meshgrid(positions, times)
    Z = diffusivity_profile.values
    
    cf = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(cf, ax=ax, label='Diffusion Coefficient / cm² s⁻¹')
    
    ax.set_xlabel('Position / cm')
    ax.set_ylabel('Time / s')
    ax.set_title('Concentration Profile Evolution')
    
    if save_path:
        plt.tight_layout()
        fig.savefig(save_path, dpi=1200, bbox_inches='tight')
    
    if display:
        plt.tight_layout()
        plt.show()
    # else:
    #     plt.close(fig)
    
    return fig, ax

def plot_norm_flux_over_time(flux_data: pd.DataFrame,
                       experimental_data: Optional[pd.DataFrame] = None,
                       ax: Optional[plt.Axes] = None,
                       save_path: Optional[str] = None,
                       display: bool = True) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot flux evolution over time
    
    Parameters
    ----------
    flux_data : pd.DataFrame
        Model flux data
    experimental_data : pd.DataFrame, optional
        Experimental flux data
    ax : plt.Axes, optional
        Matplotlib axes for plotting
    save_path : str, optional
        Path to save the figure
    display : bool, optional
        Whether to display the plot (default: True)
    """
    # Check if 'time' and 'flux' columns are present
    if 'time' not in flux_data.columns or 'flux' not in flux_data.columns:
        raise ValueError("Missing required columns: 'time', 'flux'")
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    set_style()
    
    # Plot calculated flux
    ax.plot(flux_data['time'], flux_data['flux'], 'b-', label='Model')
    
    # Plot experimental data if provided
    if experimental_data is not None:
        ax.plot(experimental_data['time'], experimental_data['flux'],
                'ko', label='Experimental', alpha=0.5)
    
    ax.set_xlabel('Time / s')
    ax.set_ylabel('Dimensionless Flux')
    ax.set_title('Flux Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.tight_layout()
        fig.savefig(save_path, dpi=1200, bbox_inches='tight')
    
    if display:
        plt.tight_layout()
        plt.show()
    # else:
    #     plt.close(fig)
    
    return fig, ax

def plot_norm_flux_over_tau(flux_data: pd.DataFrame,
                            experimental_data: Optional[pd.DataFrame] = None,
                            ax: Optional[plt.Axes] = None,
                            save_path: Optional[str] = None,
                            display: bool = True) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot normalised flux evolution over normalised time
    
    Parameters
    ----------
    flux_data : pd.DataFrame
        Model flux data
    experimental_data : pd.DataFrame, optional
        Experimental flux data
    ax : plt.Axes, optional
        Matplotlib axes for plotting
    save_path : str, optional
        Path to save the figure
    display : bool, optional
        Whether to display the plot (default: True)
    """
    # Check if 'time' and 'normalised_flux' columns are present
    if 'tau' not in flux_data.columns or 'normalised_flux' not in flux_data.columns:
        raise ValueError("Missing required columns: 'tau', 'normalised_flux'")
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    set_style()
    
    # Plot calculated flux
    ax.plot(flux_data['tau'], flux_data['normalised_flux'], 'b-', label='Model')
    
    # Plot experimental data if provided
    if experimental_data is not None:
        ax.plot(experimental_data['tau'], experimental_data['normalised_flux'],
                'ko', label='Experimental', alpha=0.5)
    
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel('Normalised Flux')
    ax.set_title('Flux Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.tight_layout()
        fig.savefig(save_path, dpi=1200, bbox_inches='tight')
    
    if display:
        plt.tight_layout()
        plt.show()
    # else:
    #     plt.close(fig)
    
    return fig, ax

def plot_norm_flux_over_time(flux_data: pd.DataFrame,
                           experimental_data: Optional[pd.DataFrame] = None,
                           ax: Optional[plt.Axes] = None,
                           save_path: Optional[str] = None,
                           display: bool = True) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot normalised flux evolution over time
    
    Parameters
    ----------
    flux_data : pd.DataFrame
        Model flux data
    experimental_data : pd.DataFrame, optional
        Experimental flux data
    ax : plt.Axes, optional
        Matplotlib axes for plotting
    save_path : str, optional
        Path to save the figure
    display : bool, optional
        Whether to display the plot (default: True)
    """
    # Check if required columns are present
    if 'time' not in flux_data.columns or 'normalised_flux' not in flux_data.columns:
        raise ValueError("Missing required columns: 'time', 'normalised_flux'")
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    set_style()
    
    # Plot calculated flux
    ax.plot(flux_data['time'], flux_data['normalised_flux'], 'b-', label='Model')
    
    # Plot experimental data if provided
    if experimental_data is not None:
        ax.plot(experimental_data['time'], experimental_data['normalised_flux'],
                'ko', label='Experimental', alpha=0.5)
    
    ax.set_xlabel('Time / s')
    ax.set_ylabel('Normalised Flux')
    ax.set_title('Normalised Flux Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.tight_layout()
        fig.savefig(save_path, dpi=1200, bbox_inches='tight')
    
    if display:
        plt.tight_layout()
        plt.show()
    
    return fig, ax

def plot_diffusivity_location_profile(diffusivity_profile: pd.DataFrame,
                                      L: float,
                                      T: float,
                                      ax: Optional[plt.Axes] = None,
                                      save_path: Optional[str] = None,
                                      display: bool = True) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot diffusivity-location profile at different times
    
    Parameters
    ----------
    diffusivity_profile : pd.DataFrame
        Diffusivity profile data
    L : float
        Thickness of the polymer [cm]
    T : float
        Total time [s]
    ax : plt.Axes, optional
        Matplotlib axes for plotting
    save_path : str, optional
        Path to save the figure
    display : bool, optional
        Whether to display the plot (default: True)
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    set_style()
    
    # Select time points for plotting (initial, and 5 logarithmically spaced points)
    time_points = np.concatenate(([0], np.logspace(np.log10(T/100), np.log10(T), 5)))
    positions = np.linspace(0, L, len(diffusivity_profile.columns))
    
    for t in time_points:
        # Find nearest time index
        t_idx = int(t / T * (len(diffusivity_profile) - 1))
        ax.plot(positions, diffusivity_profile.iloc[t_idx, :], 
                label=f't = {t:.0f} s')
    
    ax.set_xlabel('Position / cm')
    ax.set_ylabel('Diffusion Coefficient / cm² s⁻¹')
    ax.set_title('Diffusivity-Location Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.tight_layout()
        fig.savefig(save_path, dpi=1200, bbox_inches='tight')
    
    if display:
        plt.tight_layout()
        plt.show()
    # else:
    #     plt.close(fig)
    
    return fig, ax
