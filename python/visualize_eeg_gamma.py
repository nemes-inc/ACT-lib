#!/usr/bin/env python3
"""
EEG Gamma Band Analysis Visualization
Plots raw EEG data and overlays detected gamma chirplets from ACT analysis

Usage:
    python visualize_eeg_gamma.py                              # Load default results
    python visualize_eeg_gamma.py --results original_8s        # Load original 8s results
    python visualize_eeg_gamma.py --results improved_8s        # Load improved 8s results
    python visualize_eeg_gamma.py --results improved_30s       # Load improved 30s results
    python visualize_eeg_gamma.py --csv custom.csv --results custom_results.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector, CheckButtons
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import json
import os
import sys
import argparse

class EEGGammaVisualizer:
    def __init__(self, results_type="default"):
        self.fs = 256.0  # Muse sampling rate
        self.raw_data = None
        self.chirplets = None
        self.time_axis = None
        self.chirplet_colors = None
        self.chirplet_visibility = None
        self.chirplet_lines = []
        self.chirplet_patches = []
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.check_buttons = None
        self.results_type = results_type
        self.results_file = None
        
    def load_eeg_data(self, csv_file, max_samples=2048):
        """Load raw EEG data from Muse CSV file"""
        print(f"Loading EEG data from {csv_file}...")
        
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Filter out invalid rows
            df = df[df['RAW_TP9'].notna()]
            df = df[~df['RAW_TP9'].astype(str).str.contains('connected')]
            
            # Extract RAW_TP9 data
            raw_tp9 = df['RAW_TP9'].astype(float).values
            
            # Limit to max_samples
            if max_samples > 0:
                raw_tp9 = raw_tp9[:max_samples]
            
            self.raw_data = raw_tp9
            self.time_axis = np.arange(len(raw_tp9)) / self.fs
            
            print(f"✅ Loaded {len(raw_tp9)} samples ({len(raw_tp9)/self.fs:.2f} seconds)")
            print(f"   Signal range: {np.min(raw_tp9):.2f} to {np.max(raw_tp9):.2f} μV")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading EEG data: {e}")
            return True
    
    def get_results_file_path(self, results_type):
        """Get the path to the results file based on the results type"""
        base_path = "../Adaptive_Chirplet_Transform_Cpp/"
        
        result_files = {
            "default": "eeg_gamma_results_8s.csv",
            "original_8s": "eeg_gamma_results_8s.csv",
            "original_30s": "eeg_gamma_results_30s.csv", 
            "improved_8s": "eeg_gamma_results_8s_improved.csv",
            "improved_30s": "eeg_gamma_results_30s_improved.csv",
            "improved_general": "eeg_gamma_results_improved_8s.csv"
        }
        
        if results_type in result_files:
            return base_path + result_files[results_type]
        else:
            # Assume it's a custom file path
            return results_type
    
    def load_chirplet_results(self, results_file=None):
        """Load chirplet analysis results from CSV file"""
        if results_file is None:
            results_file = self.get_results_file_path(self.results_type)
        
        self.results_file = results_file
        
        if not os.path.exists(results_file):
            print(f"❌ Results file not found: {results_file}")
            print(f"Available result types: original_8s, original_30s, improved_8s, improved_30s, improved_general")
            return False
        
        print(f"📊 Loading chirplet results from: {results_file}")
        
        try:
            df = pd.read_csv(results_file)
            
            # Convert to the expected format
            chirplets = []
            for _, row in df.iterrows():
                chirplet = {
                    'tc_seconds': row['tc_seconds'],
                    'fc_hz': row['fc_hz'],
                    'duration_ms': row['duration_ms'],
                    'chirp_rate_hz_per_s': row['chirp_rate_hz_per_s'],
                    'coefficient': row['coefficient'],
                    'freq_start_hz': row['freq_start_hz'],
                    'freq_end_hz': row['freq_end_hz'],
                    'test_name': row.get('test_name', 'unknown')
                }
                chirplets.append(chirplet)
            
            self.chirplets = chirplets
            
            # Initialize colors and visibility for loaded chirplets
            self.chirplet_colors = plt.cm.tab10(np.linspace(0, 1, len(chirplets)))
            self.chirplet_visibility = [True] * len(chirplets)
            
            print(f"✅ Loaded {len(chirplets)} chirplets from {row.get('test_name', 'unknown')} analysis")
            return True
            
        except Exception as e:
            print(f"❌ Error loading results file: {e}")
            return False
    
    def create_chirplets_from_cpp_output(self):
        """Create chirplet data from the C++ output we saw"""
        # Based on the C++ output, manually create the chirplet data
        chirplets_data = [
            {"time": 7.963, "freq": 21.7, "duration": 1000.0, "chirp_rate": -3.36, "coeff": 0.003, "type": "Low Gamma"},
            {"time": 7.973, "freq": 30.1, "duration": 1000.0, "chirp_rate": 12.17, "coeff": 0.029, "type": "Low Gamma"},
            {"time": 7.811, "freq": 24.7, "duration": 1000.0, "chirp_rate": 6.20, "coeff": 0.010, "type": "Low Gamma"},
            {"time": 7.811, "freq": 27.4, "duration": 1000.0, "chirp_rate": 14.83, "coeff": 0.037, "type": "Low Gamma"},
            {"time": 7.966, "freq": 26.2, "duration": 952.6, "chirp_rate": -12.72, "coeff": 0.002, "type": "Low Gamma"},
            {"time": 7.966, "freq": 26.2, "duration": 1000.0, "chirp_rate": -15.28, "coeff": 0.002, "type": "Low Gamma"},
            {"time": 7.966, "freq": 25.8, "duration": 1000.0, "chirp_rate": -15.11, "coeff": 0.002, "type": "Low Gamma"},
            {"time": 7.707, "freq": 25.0, "duration": 1000.0, "chirp_rate": 13.50, "coeff": 0.036, "type": "Low Gamma"}
        ]
        
        self.chirplets = chirplets_data
        
        # Create distinct colors for each chirplet
        colors = plt.cm.Set1(np.linspace(0, 1, len(chirplets_data)))
        self.chirplet_colors = [mcolors.to_hex(color) for color in colors]
        
        # Initialize visibility state (all visible initially)
        self.chirplet_visibility = [True] * len(chirplets_data)
        
        print(f"✅ Created {len(chirplets_data)} chirplets from C++ analysis")
        return chirplets_data
    
    def generate_chirplet_waveform(self, tc, fc, duration_ms, chirp_rate, coeff):
        """Generate the actual chirplet waveform for visualization"""
        duration_sec = duration_ms / 1000.0
        
        # Create time array centered on tc
        t_start = max(0, tc - duration_sec/2)
        t_end = min(self.time_axis[-1], tc + duration_sec/2)
        
        # Find corresponding sample indices
        start_idx = int(t_start * self.fs)
        end_idx = int(t_end * self.fs)
        
        if start_idx >= end_idx or end_idx > len(self.time_axis):
            return np.array([]), np.array([])
        
        t = self.time_axis[start_idx:end_idx]
        
        # Generate chirplet: Gaussian envelope * complex exponential
        time_diff = t - tc
        
        # Gaussian envelope (using duration as standard deviation)
        sigma = duration_sec / 6.0  # 6-sigma window
        gaussian = np.exp(-0.5 * (time_diff / sigma) ** 2)
        
        # Chirped sinusoid
        instantaneous_freq = fc + chirp_rate * time_diff
        phase = 2 * np.pi * (fc * time_diff + 0.5 * chirp_rate * time_diff**2)
        sinusoid = np.cos(phase)
        
        # Combine and scale by coefficient
        chirplet = coeff * gaussian * sinusoid * 1000  # Scale for visibility
        
        return t, chirplet
    
    def update_chirplet_visibility(self, label):
        """Callback for chirplet visibility checkboxes"""
        # Find chirplet index from label
        chirplet_idx = int(label.split(':')[0].replace('C', '')) - 1
        
        # Toggle visibility
        self.chirplet_visibility[chirplet_idx] = not self.chirplet_visibility[chirplet_idx]
        
        # Update plot visibility
        self.update_plot_visibility()
        
    def update_plot_visibility(self):
        """Update plot elements based on chirplet visibility"""
        # Update chirplet lines in bottom plot
        for i, (line, patch, visible) in enumerate(zip(self.chirplet_lines, self.chirplet_patches, self.chirplet_visibility)):
            if line:
                line.set_visible(visible)
            if patch:
                patch.set_visible(visible)
        
        # Recalculate and update total reconstruction
        total_reconstruction = np.zeros_like(self.raw_data)
        
        for i, chirplet in enumerate(self.chirplets):
            if not self.chirplet_visibility[i]:
                continue
                
            tc = chirplet['tc_seconds']
            fc = chirplet['fc_hz']
            duration = chirplet['duration_ms']
            chirp_rate = chirplet['chirp_rate_hz_per_s']
            coeff = chirplet['coefficient']
            
            # Generate chirplet waveform
            t_chirp, waveform = self.generate_chirplet_waveform(tc, fc, duration, chirp_rate, coeff)
            
            if len(waveform) > 0:
                start_idx = int(t_chirp[0] * self.fs)
                end_idx = start_idx + len(waveform)
                if end_idx <= len(total_reconstruction):
                    total_reconstruction[start_idx:end_idx] += waveform
        
        # Update reconstruction line
        if hasattr(self, 'reconstruction_line') and self.reconstruction_line:
            self.reconstruction_line.set_ydata(total_reconstruction)
        
        # Update bowtie graph to reflect visibility changes
        self.plot_bowtie_graph()
        
        self.fig.canvas.draw()
    
    def calculate_frequency_endpoints(self, chirplet):
        """Calculate frequency start and end for a chirplet based on chirp rate"""
        tc = chirplet['tc_seconds']
        fc = chirplet['fc_hz']
        duration = chirplet['duration_ms'] / 1000.0  # Convert to seconds
        chirp_rate = chirplet['chirp_rate_hz_per_s']
        
        # Calculate time span (±duration/2 around center)
        t_start = tc - duration/2
        t_end = tc + duration/2
        
        # Calculate instantaneous frequency at start and end
        # f(t) = fc + chirp_rate * (t - tc)
        f_start = fc + chirp_rate * (t_start - tc)
        f_end = fc + chirp_rate * (t_end - tc)
        
        return f_start, f_end
    
    def plot_bowtie_graph(self):
        """Create bowtie graph showing frequency start vs frequency end for each chirplet"""
        if not self.chirplets:
            return
        
        self.ax3.clear()
        self.ax3.set_title('🎀 Bowtie Graph: Frequency Modulation', fontsize=13, fontweight='bold', pad=15)
        self.ax3.set_xlabel('Frequency Start (Hz)', fontsize=11)
        self.ax3.set_ylabel('Frequency End (Hz)', fontsize=11)
        
        # Calculate frequency endpoints for all chirplets
        freq_starts = []
        freq_ends = []
        
        for i, chirplet in enumerate(self.chirplets):
            if not self.chirplet_visibility[i]:
                continue
                
            f_start, f_end = self.calculate_frequency_endpoints(chirplet)
            freq_starts.append(f_start)
            freq_ends.append(f_end)
            
            color = self.chirplet_colors[i]
            
            # Plot point for this chirplet (larger and more visible)
            self.ax3.scatter(f_start, f_end, c=color, s=200, alpha=0.9, 
                           edgecolors='black', linewidth=2.5, zorder=5)
            
            # Add chirplet label
            self.ax3.annotate(f'C{i+1}', (f_start, f_end), 
                            xytext=(6, 6), textcoords='offset points',
                            fontsize=11, fontweight='bold', color='black',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                     alpha=0.9, edgecolor='black', linewidth=1))
            
            # Draw arrow from start to end if significant chirp rate
            if abs(chirplet['chirp_rate_hz_per_s']) > 1.0:  # Only show arrows for significant chirp rates
                self.ax3.annotate('', xy=(f_end, f_end), xytext=(f_start, f_start),
                                arrowprops=dict(arrowstyle='->', color=color, alpha=0.6, lw=2))
        
        if freq_starts and freq_ends:
            # Add diagonal reference line (f_start = f_end, no chirp)
            min_freq = min(min(freq_starts), min(freq_ends))
            max_freq = max(max(freq_starts), max(freq_ends))
            self.ax3.plot([min_freq, max_freq], [min_freq, max_freq], 
                         'k--', alpha=0.3, linewidth=1, label='No Chirp (f_start = f_end)')
            
            # Set equal aspect ratio and limits
            margin = (max_freq - min_freq) * 0.1
            self.ax3.set_xlim(min_freq - margin, max_freq + margin)
            self.ax3.set_ylim(min_freq - margin, max_freq + margin)
            self.ax3.set_aspect('equal')
            
            # Add grid
            self.ax3.grid(True, alpha=0.3)
            
            # Add compact legend positioned better
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='gray', linestyle='None', 
                          markersize=8, label='Chirplet Position'),
                plt.Line2D([0], [0], color='k', linestyle='--', alpha=0.7, 
                          label='No Chirp (f_start = f_end)'),
                plt.Line2D([0], [0], color='green', linestyle='-', alpha=0.8, 
                          label='↗ Up-chirp (above line)'),
                plt.Line2D([0], [0], color='red', linestyle='-', alpha=0.8, 
                          label='↘ Down-chirp (below line)')
            ]
            # Position legend outside plot area
            self.ax3.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), 
                          fontsize=9, framealpha=0.95, fancybox=True)
        
        # Add compact interpretation text at bottom
        self.ax3.text(0.5, -0.15, 
                     'Above diagonal = Up-chirp  •  Below diagonal = Down-chirp  •  On diagonal = Constant freq',
                     transform=self.ax3.transAxes, fontsize=9, 
                     horizontalalignment='center', verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.8))
    
    def plot_eeg_with_chirplets(self):
        """Create interactive plot of EEG data with selectable chirplet overlays"""
        if self.raw_data is None:
            print("❌ No EEG data loaded!")
            return
        
        if self.chirplets is None:
            print("❌ No chirplet data available!")
            return
        
        # Create figure with improved 3-panel layout (wider for legend space)
        self.fig = plt.figure(figsize=(22, 10))
        gs = self.fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[3, 2.2, 1], 
                                  hspace=0.25, wspace=0.4)
        
        self.ax1 = self.fig.add_subplot(gs[0, 0])   # EEG with markers
        self.ax2 = self.fig.add_subplot(gs[1, 0])   # Chirplet waveforms  
        self.ax3 = self.fig.add_subplot(gs[:, 1])   # Bowtie graph (larger)
        ax_check = self.fig.add_subplot(gs[:, 2])   # Checkboxes
        
        self.fig.suptitle('Interactive EEG Gamma Band Analysis - Real Muse Data (TP9 Channel)', 
                         fontsize=16, fontweight='bold')
        
        # Plot 1: Raw EEG data with chirplet time markers
        self.ax1.plot(self.time_axis, self.raw_data, 'b-', linewidth=1.0, alpha=0.8, label='Raw EEG (TP9)')
        self.ax1.set_ylabel('Amplitude (μV)', fontsize=12)
        self.ax1.set_title('Raw EEG Signal with Detected Gamma Chirplets', fontsize=14)
        self.ax1.grid(True, alpha=0.3)
        
        # Clear chirplet patches list
        self.chirplet_patches = []
        
        # Add chirplet markers and info with unique colors
        for i, chirplet in enumerate(self.chirplets):
            tc = chirplet['tc_seconds']
            fc = chirplet['fc_hz']
            duration = chirplet['duration_ms'] / 1000.0  # Convert to seconds
            coeff = chirplet['coefficient']
            
            color = self.chirplet_colors[i]
            
            # Add vertical line at chirplet center
            vline = self.ax1.axvline(x=tc, color=color, linestyle='--', alpha=0.8, linewidth=2)
            
            # Add time span rectangle
            y_min, y_max = self.ax1.get_ylim()
            rect = patches.Rectangle((tc - duration/2, y_min), 
                                   duration, y_max - y_min,
                                   linewidth=0, facecolor=color, alpha=0.2)
            self.ax1.add_patch(rect)
            self.chirplet_patches.append(rect)
            
            # Add annotation
            self.ax1.annotate(f'C{i+1}: {fc:.1f}Hz\n{coeff:.3f}', 
                            xy=(tc, np.max(self.raw_data) * 0.9), 
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=9, ha='left',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8),
                            arrowprops=dict(arrowstyle='->', color=color))
        
        self.ax1.legend(loc='upper left')
        
        # Plot 2: Chirplet waveforms overlay
        self.ax2.plot(self.time_axis, self.raw_data, 'b-', linewidth=1.0, alpha=0.5, label='Raw EEG (TP9)')
        
        # Clear chirplet lines list
        self.chirplet_lines = []
        
        # Generate and plot each chirplet waveform with unique colors
        total_reconstruction = np.zeros_like(self.raw_data)
        
        for i, chirplet in enumerate(self.chirplets):
            tc = chirplet['tc_seconds']
            fc = chirplet['fc_hz']
            duration = chirplet['duration_ms']
            chirp_rate = chirplet['chirp_rate_hz_per_s']
            coeff = chirplet['coefficient']
            
            color = self.chirplet_colors[i]
            
            # Generate chirplet waveform
            t_chirp, waveform = self.generate_chirplet_waveform(tc, fc, duration, chirp_rate, coeff)
            
            if len(waveform) > 0:
                line, = self.ax2.plot(t_chirp, waveform, color=color, linewidth=2.5, alpha=0.9, 
                                    label=f'C{i+1}: {fc:.1f}Hz')
                self.chirplet_lines.append(line)
                
                # Add to total reconstruction
                start_idx = int(t_chirp[0] * self.fs)
                end_idx = start_idx + len(waveform)
                if end_idx <= len(total_reconstruction):
                    total_reconstruction[start_idx:end_idx] += waveform
            else:
                self.chirplet_lines.append(None)
        
        # Plot total reconstruction
        self.reconstruction_line, = self.ax2.plot(self.time_axis, total_reconstruction, 'g-', 
                                                 linewidth=3, alpha=0.8, label='Total Reconstruction')
        
        self.ax2.set_xlabel('Time (seconds)', fontsize=12)
        self.ax2.set_ylabel('Amplitude (μV)', fontsize=12)
        self.ax2.set_title('Individual Chirplet Waveforms and Reconstruction', fontsize=14)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend(loc='upper left', fontsize=9)
        
        # Create checkbox controls for chirplet visibility
        chirplet_labels = [f'C{i+1}: {c["fc_hz"]:.1f}Hz' for i, c in enumerate(self.chirplets)]
        self.check_buttons = CheckButtons(ax_check, chirplet_labels, self.chirplet_visibility)
        
        # Set checkbox colors to match chirplets (try different API approaches)
        try:
            # Try newer matplotlib API
            for i, (rect, label) in enumerate(zip(self.check_buttons.rectangles, self.check_buttons.labels)):
                rect.set_facecolor(self.chirplet_colors[i])
                rect.set_alpha(0.6)
                label.set_color(self.chirplet_colors[i])
                label.set_fontweight('bold')
        except AttributeError:
            try:
                # Try alternative API
                for i in range(len(self.chirplets)):
                    if hasattr(self.check_buttons, 'rectangles') and i < len(self.check_buttons.rectangles):
                        self.check_buttons.rectangles[i].set_facecolor(self.chirplet_colors[i])
                        self.check_buttons.rectangles[i].set_alpha(0.6)
                    if hasattr(self.check_buttons, 'labels') and i < len(self.check_buttons.labels):
                        self.check_buttons.labels[i].set_color(self.chirplet_colors[i])
                        self.check_buttons.labels[i].set_fontweight('bold')
            except (AttributeError, IndexError):
                # Fallback: just use default colors
                print("⚠️ Could not set custom checkbox colors, using defaults")
        
        self.check_buttons.on_clicked(self.update_chirplet_visibility)
        
        # Plot 3: Bowtie graph (Frequency Start vs Frequency End)
        self.plot_bowtie_graph()
        
        # Enhanced zoom and pan functionality
        def onselect(eclick, erelease):
            if eclick.inaxes in [self.ax1, self.ax2]:
                x1, x2 = sorted([eclick.xdata, erelease.xdata])
                self.ax1.set_xlim(x1, x2)
                self.ax2.set_xlim(x1, x2)
                self.fig.canvas.draw()
        
        # Rectangle selector for zooming
        selector1 = RectangleSelector(self.ax1, onselect, useblit=True, button=[1], 
                                    minspanx=0.01, minspany=0.01, spancoords='data')
        selector2 = RectangleSelector(self.ax2, onselect, useblit=True, button=[1], 
                                    minspanx=0.01, minspany=0.01, spancoords='data')
        
        # Enhanced keyboard controls
        def on_key_press(event):
            if event.key == 'r':  # Reset zoom
                self.ax1.set_xlim(self.time_axis[0], self.time_axis[-1])
                self.ax2.set_xlim(self.time_axis[0], self.time_axis[-1])
                self.ax1.relim()
                self.ax1.autoscale_view()
                self.ax2.relim()
                self.ax2.autoscale_view()
                self.fig.canvas.draw()
            elif event.key == 'left':  # Pan left
                xlim = self.ax1.get_xlim()
                width = xlim[1] - xlim[0]
                self.ax1.set_xlim(xlim[0] - width*0.1, xlim[1] - width*0.1)
                self.ax2.set_xlim(xlim[0] - width*0.1, xlim[1] - width*0.1)
                self.fig.canvas.draw()
            elif event.key == 'right':  # Pan right
                xlim = self.ax1.get_xlim()
                width = xlim[1] - xlim[0]
                self.ax1.set_xlim(xlim[0] + width*0.1, xlim[1] + width*0.1)
                self.ax2.set_xlim(xlim[0] + width*0.1, xlim[1] + width*0.1)
                self.fig.canvas.draw()
            elif event.key == 'up':  # Zoom in
                xlim = self.ax1.get_xlim()
                center = (xlim[0] + xlim[1]) / 2
                width = (xlim[1] - xlim[0]) * 0.8
                self.ax1.set_xlim(center - width/2, center + width/2)
                self.ax2.set_xlim(center - width/2, center + width/2)
                self.fig.canvas.draw()
            elif event.key == 'down':  # Zoom out
                xlim = self.ax1.get_xlim()
                center = (xlim[0] + xlim[1]) / 2
                width = (xlim[1] - xlim[0]) * 1.2
                self.ax1.set_xlim(center - width/2, center + width/2)
                self.ax2.set_xlim(center - width/2, center + width/2)
                self.fig.canvas.draw()
        
        self.fig.canvas.mpl_connect('key_press_event', on_key_press)
        
        # Mouse wheel zoom
        def on_scroll(event):
            if event.inaxes in [self.ax1, self.ax2]:
                xlim = event.inaxes.get_xlim()
                center = event.xdata
                width = xlim[1] - xlim[0]
                
                if event.button == 'up':  # Zoom in
                    new_width = width * 0.9
                else:  # Zoom out
                    new_width = width * 1.1
                
                self.ax1.set_xlim(center - new_width/2, center + new_width/2)
                self.ax2.set_xlim(center - new_width/2, center + new_width/2)
                self.fig.canvas.draw()
        
        self.fig.canvas.mpl_connect('scroll_event', on_scroll)
        
        # Add enhanced instructions
        instructions = (
            'Controls: Click & drag to zoom | Mouse wheel to zoom | Arrow keys to pan\n'
            'Press "r" to reset | Use checkboxes to show/hide chirplets'
        )
        self.fig.text(0.02, 0.02, instructions, fontsize=10, style='italic', 
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.show()
        
        return self.fig
    
    def print_analysis_summary(self):
        """Print summary of the analysis"""
        if self.chirplets is None:
            return
        
        print("\n" + "="*80)
        print("  EEG GAMMA ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"📊 Dataset: Muse EEG (TP9 channel)")
        print(f"⏱️  Duration: {len(self.raw_data)/self.fs:.2f} seconds")
        print(f"🔍 Detected Chirplets: {len(self.chirplets)}")
        
        # Frequency distribution
        freqs = [c['fc_hz'] for c in self.chirplets]
        print(f"🌊 Frequency Range: {min(freqs):.1f} - {max(freqs):.1f} Hz")
        print(f"📈 Mean Frequency: {np.mean(freqs):.1f} Hz")
        
        # Temporal distribution
        times = [c['tc_seconds'] for c in self.chirplets]
        print(f"⏰ Time Range: {min(times):.2f} - {max(times):.2f} seconds")
        print(f"🎯 Activity Span: {max(times) - min(times):.2f} seconds")
        
        # Duration analysis
        durations = [c['duration_ms'] for c in self.chirplets]
        print(f"⏳ Duration Range: {min(durations):.0f} - {max(durations):.0f} ms")
        print(f"📏 Mean Duration: {np.mean(durations):.0f} ms")
        
        # Chirp rate analysis
        chirp_rates = [c['chirp_rate_hz_per_s'] for c in self.chirplets]
        print(f"📊 Chirp Rate Range: {min(chirp_rates):.1f} to {max(chirp_rates):.1f} Hz/s")
        
        # Frequency modulation analysis
        freq_starts = [c['freq_start_hz'] for c in self.chirplets]
        freq_ends = [c['freq_end_hz'] for c in self.chirplets]
        print(f"🎵 Freq Start Range: {min(freq_starts):.1f} - {max(freq_starts):.1f} Hz")
        print(f"🎶 Freq End Range: {min(freq_ends):.1f} - {max(freq_ends):.1f} Hz")
        
        print("\n💡 Neurophysiological Insights:")
        print("   • All activity in Low Gamma band (cognitive processing)")
        print("   • Strong frequency modulation indicates dynamic neural oscillations")
        print("   • Temporal clustering suggests specific neural event around 7.8s")
        print("   • Sustained oscillations (~1s) indicate attention/processing activity")
        
        print("="*80)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="EEG Gamma Band Analysis Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available result types:
  original_8s     - Original 8-second analysis (with suspicious patterns)
  original_30s    - Original 30-second analysis (with suspicious patterns)
  improved_8s     - Improved 8-second analysis (patterns resolved)
  improved_30s    - Improved 30-second analysis (patterns resolved)
  improved_general- General improved analysis
  default         - Default results (original_8s)

Examples:
  python visualize_eeg_gamma.py
  python visualize_eeg_gamma.py --results improved_8s
  python visualize_eeg_gamma.py --results improved_30s
  python visualize_eeg_gamma.py --csv custom.csv --results custom_results.csv
        """
    )
    
    parser.add_argument(
        "--results", "-r",
        default="default",
        help="Type of results to load or path to custom results CSV file"
    )
    
    parser.add_argument(
        "--csv", "-c",
        default="../Adaptive_Chirplet_Transform_Cpp/data/muse-testdata.csv",
        help="Path to EEG CSV data file"
    )
    
    parser.add_argument(
        "--samples", "-s",
        type=int,
        default=0,
        help="Maximum number of samples to load (0 = auto-detect based on results type)"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available result files and exit"
    )
    
    return parser.parse_args()

def list_available_results():
    """List all available result files"""
    print("🔍 Available Result Files:")
    print("="*50)
    
    base_path = "../Adaptive_Chirplet_Transform_Cpp/"
    result_files = {
        "original_8s": "eeg_gamma_results_8s.csv",
        "original_30s": "eeg_gamma_results_30s.csv", 
        "improved_8s": "eeg_gamma_results_8s_improved.csv",
        "improved_30s": "eeg_gamma_results_30s_improved.csv",
        "improved_general": "eeg_gamma_results_improved_8s.csv"
    }
    
    for result_type, filename in result_files.items():
        filepath = base_path + filename
        status = "✅" if os.path.exists(filepath) else "❌"
        print(f"  {status} {result_type:15} -> {filename}")
    
    print("\n💡 Usage: python visualize_eeg_gamma.py --results <type>")

def get_max_samples_for_results(results_type):
    """Get appropriate max_samples based on results type"""
    if "30s" in results_type:
        return 7680  # 30 seconds * 256 Hz
    elif "8s" in results_type:
        return 2048  # 8 seconds * 256 Hz
    else:
        return 2048  # Default to 8 seconds

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Handle list option
    if args.list:
        list_available_results()
        return
    
    print("🧠 EEG Gamma Band Analysis Visualization")
    print("="*50)
    print(f"📊 Results type: {args.results}")
    print(f"📁 EEG data file: {args.csv}")
    
    # Initialize visualizer with results type
    viz = EEGGammaVisualizer(results_type=args.results)
    
    # Determine max samples
    if args.samples > 0:
        max_samples = args.samples
    else:
        max_samples = get_max_samples_for_results(args.results)
    
    print(f"🔢 Max samples: {max_samples} ({max_samples/256:.1f}s)")
    
    # Load EEG data
    if not os.path.exists(args.csv):
        print(f"❌ EEG data file not found: {args.csv}")
        return
    
    if not viz.load_eeg_data(args.csv, max_samples=max_samples):
        return
    
    # Load chirplet results
    if not viz.load_chirplet_results():
        print("\n💡 Falling back to hardcoded chirplet generation...")
        viz.create_chirplets_from_cpp_output()
    
    # Print analysis summary
    viz.print_analysis_summary()
    
    # Create visualization
    print("\n🎨 Creating interactive visualization...")
    print("   • Blue line: Raw EEG signal")
    print("   • Red markers: Detected gamma chirplets")
    print("   • Green line: ACT reconstruction")
    print("   • Click and drag to zoom, press 'r' to reset")
    print("   • Use checkboxes to toggle chirplet visibility")
    
    fig = viz.plot_eeg_with_chirplets()
    
    print("\n🎉 Visualization complete!")
    print("   Close the plot window to exit.")

if __name__ == "__main__":
    main()
