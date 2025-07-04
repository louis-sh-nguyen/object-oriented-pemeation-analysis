import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from ....utils.plotting import set_style

def plot_timelag_analysis(model, data: pd.DataFrame, 
                         ax: Optional[plt.Axes] = None,
                         fig: Optional[plt.Figure] = None,
                         save_path: Optional[str] = None,
                         display: bool = True) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot time-lag analysis results
    
    Parameters
    ----------
    model : TimelagModel
        Fitted time-lag model
    data : pd.DataFrame
        Experimental data
    ax : plt.Axes, optional
        Matplotlib axes for plotting
    save_path : str, optional
        Path to save the figure
    display : bool, optional
        Whether to display the plot (default: True)
        
    Returns
    -------
    fig : plt.Figure
        Figure object
    ax : plt.Axes
        Axes object
    """
    if 'time' not in data.columns or 'cumulative_flux' not in data.columns:
        raise ValueError("Data must contain 'time' and 'cumulative_flux' columns")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    elif fig is None:
        fig = ax.figure
    set_style()
    
    # Plot experimental data
    ax.plot(data['time'], data['cumulative_flux'], 'ko', label='Experimental', alpha=0.5)
    
    # Plot fitted line using stored regression results
    x_fit = np.linspace(0, max(data['time']), 100)
    y_fit = model.get_steady_state_line(x_fit)
    ax.plot(x_fit, y_fit, 'r-', label=f'Fitted Line (R² = {model.results["steady_state_r_squared"]:.4f})')
    
    # Add time lag line
    time_lag = model.results['time_lag']
    ax.axvline(time_lag, color='g', ls='--', label=f'Time Lag = {time_lag:.1f} s')
    
    ax.set_xlabel('Time / s')
    ax.set_ylabel('Cumulative Flux / cm³(STP) cm⁻²')
    ax.set_title('Time-Lag Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if display:
        plt.tight_layout()
        plt.show()
    # else:
    #     plt.close(fig)
    
    return fig, ax

def plot_concentration_profile(conc_profile: pd.DataFrame,
                             ax: Optional[plt.Axes] = None,
                             fig: Optional[plt.Figure] = None,
                             save_path: Optional[str] = None,
                             display: bool = True) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot concentration profile evolution
    
    Parameters
    ----------
    conc_profile : pd.DataFrame
        Concentration profile data
    ax : plt.Axes, optional
        Matplotlib axes for plotting
    save_path : str, optional
        Path to save the figure
    display : bool, optional
        Whether to display the plot (default: True)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    elif fig is None:
        fig = ax.figure
    set_style()
    
    times = conc_profile.index
    positions = [float(col.split('=')[1]) for col in conc_profile.columns]
    
    X, Y = np.meshgrid(positions, times)
    Z = conc_profile.values
    
    cf = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(cf, ax=ax, label='Concentration / cm³(STP) cm⁻³')
    
    ax.set_xlabel('Position / cm')
    ax.set_ylabel('Time / s')
    ax.set_title('Concentration Profile Evolution')
    
    if save_path:
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if display:
        plt.tight_layout()
        plt.show()
    # else:
    #     plt.close(fig)
    
    return fig, ax

def plot_flux_over_time(flux_data: pd.DataFrame,
                       experimental_data: Optional[pd.DataFrame] = None,
                       ax: Optional[plt.Axes] = None,
                       fig: Optional[plt.Figure] = None,
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
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    elif fig is None:
        fig = ax.figure
    set_style()
    
    # Plot calculated flux
    ax.plot(flux_data['time'], flux_data['flux'], 'b-', label='Model')
    
    # Plot experimental data if provided
    if experimental_data is not None:
        ax.plot(experimental_data['time'], experimental_data['flux'],
                'ko', label='Experimental', alpha=0.5)
    
    ax.set_xlabel('Time / s')
    ax.set_ylabel('Flux / cm³(STP) cm⁻² s⁻¹')
    ax.set_title('Flux Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if display:
        plt.tight_layout()
        plt.show()
    # else:
    #     plt.close(fig)
    
    return fig, ax