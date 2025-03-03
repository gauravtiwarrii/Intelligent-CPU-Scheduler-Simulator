import tkinter as tk
from tkinter import ttk, messagebox
from ttkthemes import ThemedTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import csv
from datetime import datetime

# --- Module 1: Core Scheduler Engine (Unchanged) ---
class Process:
    def __init__(self, pid, arrival_time, burst_time, priority=0):
        self.pid = pid
        self.arrival_time = arrival_time
        self.burst_time = burst_time 
        self.priority = priority
        self.remaining_time = burst_time
        self.start_time = 0
        self.end_time = 0
        self.waiting_time = 0
        self.turnaround_time = 0

def fcfs_scheduler(processes):
    processes.sort(key=lambda x: x.arrival_time)
    current_time = 0
    timeline = []
    for p in processes:
        p.start_time = max(current_time, p.arrival_time)
        p.end_time = p.start_time + p.burst_time
        p.waiting_time = p.start_time - p.arrival_time
        p.turnaround_time = p.end_time - p.arrival_time
        timeline.append((p.pid, p.start_time, p.end_time))
        current_time = p.end_time
    return processes, "FCFS (Non-Preemptive)", timeline

def sjf_non_preemptive(processes):
    processes.sort(key=lambda x: (x.arrival_time, x.burst_time))
    current_time = 0
    completed = []
    remaining = processes.copy()
    timeline = []
    while remaining:
        available = [p for p in remaining if p.arrival_time <= current_time]
        if not available:
            current_time += 1
            continue
        p = min(available, key=lambda x: x.burst_time)
        p.start_time = current_time
        p.end_time = p.start_time + p.burst_time
        p.waiting_time = p.start_time - p.arrival_time
        p.turnaround_time = p.end_time - p.arrival_time
        current_time = p.end_time
        timeline.append((p.pid, p.start_time, p.end_time))
        completed.append(p)
        remaining.remove(p)
    return completed, "SJF (Non-Preemptive)", timeline

def sjf_preemptive(processes):
    processes.sort(key=lambda x: x.arrival_time)
    current_time = processes[0].arrival_time if processes else 0
    completed = []
    timeline = []
    remaining = processes.copy()
    while remaining or completed:
        available = [p for p in remaining if p.arrival_time <= current_time]
        if not available and not completed:
            current_time += 1
            continue
        if available:
            p = min(available, key=lambda x: x.remaining_time)
            if p.start_time == 0:
                p.start_time = current_time
            current_time += 1
            p.remaining_time -= 1
            timeline.append((p.pid, current_time-1, current_time))
            if p.remaining_time == 0:
                p.end_time = current_time
                p.turnaround_time = p.end_time - p.arrival_time
                p.waiting_time = p.turnaround_time - p.burst_time
                completed.append(p)
                remaining.remove(p)
        else:
            current_time += 1
    return completed, "SJF (Preemptive - SRTF)", timeline

def rr_scheduler(processes, quantum):
    queue = processes.copy()
    queue.sort(key=lambda x: x.arrival_time)
    current_time = queue[0].arrival_time if queue else 0
    timeline = []
    while queue:
        p = queue.pop(0)
        if p.start_time == 0:
            p.start_time = current_time
        exec_time = min(quantum, p.remaining_time)
        timeline.append((p.pid, current_time, current_time + exec_time))
        current_time += exec_time
        p.remaining_time -= exec_time
        arrived = [proc for proc in processes if proc.arrival_time <= current_time and proc not in queue and proc.remaining_time > 0]
        queue.extend(arrived)
        if p.remaining_time > 0:
            queue.append(p)
        else:
            p.end_time = current_time
            p.turnaround_time = p.end_time - p.arrival_time
            p.waiting_time = p.turnaround_time - p.burst_time
    return processes, "Round Robin", timeline

def priority_non_preemptive(processes):
    processes.sort(key=lambda x: (x.arrival_time, x.priority))
    current_time = 0
    completed = []
    remaining = processes.copy()
    timeline = []
    while remaining:
        available = [p for p in remaining if p.arrival_time <= current_time]
        if not available:
            current_time += 1
            continue
        p = min(available, key=lambda x: x.priority)
        p.start_time = current_time
        p.end_time = p.start_time + p.burst_time
        p.waiting_time = p.start_time - p.arrival_time
        p.turnaround_time = p.end_time - p.arrival_time
        current_time = p.end_time
        timeline.append((p.pid, p.start_time, p.end_time))
        completed.append(p)
        remaining.remove(p)
    return completed, "Priority (Non-Preemptive)", timeline

def priority_preemptive(processes):
    processes.sort(key=lambda x: x.arrival_time)
    current_time = processes[0].arrival_time if processes else 0
    completed = []
    timeline = []
    remaining = processes.copy()
    while remaining or completed:
        available = [p for p in remaining if p.arrival_time <= current_time]
        if not available and not completed:
            current_time += 1
            continue
        if available:
            p = min(available, key=lambda x: x.priority)
            if p.start_time == 0:
                p.start_time = current_time
            current_time += 1
            p.remaining_time -= 1
            timeline.append((p.pid, current_time-1, current_time))
            if p.remaining_time == 0:
                p.end_time = current_time
                p.turnaround_time = p.end_time - p.arrival_time
                p.waiting_time = p.turnaround_time - p.burst_time
                completed.append(p)
                remaining.remove(p)
        else:
            current_time += 1
    return completed, "Priority (Preemptive)", timeline

def intelligent_scheduler(processes, quantum=2):
    avg_burst = sum(p.burst_time for p in processes) / len(processes)
    processes_copy = [Process(p.pid, p.arrival_time, p.burst_time, p.priority) for p in processes]
    if avg_burst < 5:
        return sjf_non_preemptive(processes_copy)
    else:
        return rr_scheduler(processes_copy, quantum)

def calculate_metrics(processes):
    if not processes or not all(hasattr(p, 'waiting_time') for p in processes):
        raise ValueError("Invalid process list passed to calculate_metrics")
    avg_waiting = sum(p.waiting_time for p in processes) / len(processes)
    avg_turnaround = sum(p.turnaround_time for p in processes) / len(processes)
    return avg_waiting, avg_turnaround

# --- Module 2: Enhanced Interactive GUI with Separate Gantt Window ---
class SchedulerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Intelligent CPU Scheduler Simulator")
        self.root.geometry("1200x900")
        self.processes = []
        self.process_entries = []
        self.canvas_widget = None
        self.anim_running = False
        self.anim = None
        self.timeline = None
        self.gantt_window = None
        
        self.root.set_theme("radiance")
        
        main_frame = tk.Frame(root, bg="#f0f8ff")
        main_frame.pack(fill="both", expand=True)
        
        input_frame = ttk.LabelFrame(main_frame, text="Process Configuration", padding=15)
        input_frame.pack(fill="x", pady=10, padx=10)
        
        self.canvas = tk.Canvas(input_frame, height=250, bg="#f0f8ff")
        scrollbar = ttk.Scrollbar(input_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        headers = ["PID", "Arrival", "Burst", "Priority", "Actions"]
        for i, header in enumerate(headers):
            label = ttk.Label(self.scrollable_frame, text=header, font=("Helvetica", 12, "bold"))
            label.grid(row=0, column=i, padx=10, pady=5)
            self.add_tooltip(label, f"Enter {header.lower()} for each process")
        
        self.add_process_row()
        
        ttk.Button(input_frame, text="Add Process", command=self.add_process_row).pack(pady=10)
        
        algo_frame = ttk.LabelFrame(main_frame, text="Algorithm & Settings", padding=15)
        algo_frame.pack(fill="x", pady=10, padx=10)
        self.algo_var = tk.StringVar(value="Intelligent")
        algos = [
            ("FCFS", "FCFS"), ("SJF (NP)", "SJF-NP"), ("SJF (P)", "SJF-P"),
            ("Round Robin", "RR"), ("Priority (NP)", "PR-NP"), ("Priority (P)", "PR-P"),
            ("Intelligent", "Intelligent")
        ]
        for i, (text, value) in enumerate(algos):
            rb = ttk.Radiobutton(algo_frame, text=text, variable=self.algo_var, value=value)
            rb.grid(row=0, column=i, padx=10, pady=5)
            self.add_tooltip(rb, f"Select {text} scheduling algorithm")
        
        ttk.Label(algo_frame, text="RR Quantum:").grid(row=1, column=0, padx=10, pady=5)
        self.quantum_var = tk.StringVar(value="2")
        quantum_entry = ttk.Entry(algo_frame, textvariable=self.quantum_var, width=5)
        quantum_entry.grid(row=1, column=1, padx=10, pady=5)
        self.add_tooltip(quantum_entry, "Set quantum for Round Robin (default: 2)")
        
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(pady=15, padx=10)
        buttons = [
            ("Run Simulation", self.run_simulation, "Run the selected scheduling algorithm"),
            ("View Gantt Chart", self.view_gantt_chart, "View Gantt chart in a new window"),
            ("Start Animation", self.start_animation, "Start real-time animation in new window"),
            ("Stop Animation", self.stop_animation, "Stop real-time animation"),
            ("Compare All", self.compare_all, "Compare all algorithms' performance"),
            ("Export Results", self.export_results, "Export results to CSV"),
            ("Reset", self.reset, "Reset all inputs and results")
        ]
        for text, cmd, tip in buttons:
            btn = ttk.Button(control_frame, text=text, command=cmd)
            btn.pack(side="left", padx=10)
            self.add_tooltip(btn, tip)
        
        self.result_frame = ttk.LabelFrame(main_frame, text="Simulation Results", padding=15)
        self.result_frame.pack(fill="both", expand=True, pady=10, padx=10)
        self.result_text = tk.Text(self.result_frame, height=10, width=100, font=("Courier", 11), bg="#f0f8ff")
        self.result_text.pack(padx=10, pady=5)

    def add_process_row(self):
        row = len(self.process_entries) + 1
        pid_entry = ttk.Entry(self.scrollable_frame, width=10)
        arrival_entry = ttk.Entry(self.scrollable_frame, width=10)
        burst_entry = ttk.Entry(self.scrollable_frame, width=10)
        priority_entry = ttk.Entry(self.scrollable_frame, width=10)
        pid_entry.insert(0, f"P{row}")
        arrival_entry.insert(0, "0")
        burst_entry.insert(0, "0")
        priority_entry.insert(0, "0")
        
        pid_entry.grid(row=row, column=0, padx=10, pady=5)
        arrival_entry.grid(row=row, column=1, padx=10, pady=5)
        burst_entry.grid(row=row, column=2, padx=10, pady=5)
        priority_entry.grid(row=row, column=3, padx=10, pady=5)
        
        remove_btn = ttk.Button(self.scrollable_frame, text="Remove", command=lambda: self.remove_process_row(row-1))
        remove_btn.grid(row=row, column=4, padx=10, pady=5)
        self.add_tooltip(remove_btn, "Remove this process")
        
        self.process_entries.append((pid_entry, arrival_entry, burst_entry, priority_entry, remove_btn))
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def remove_process_row(self, index):
        if len(self.process_entries) > 1:
            for widget in self.process_entries.pop(index):
                widget.destroy()
            for i, (pid_e, *_) in enumerate(self.process_entries):
                pid_e.delete(0, tk.END)
                pid_e.insert(0, f"P{i+1}")
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def get_processes(self):
        self.processes.clear()
        for pid_entry, arrival_entry, burst_entry, priority_entry, _ in self.process_entries:
            pid = pid_entry.get()
            arrival = int(arrival_entry.get())
            burst = int(burst_entry.get())
            priority = int(priority_entry.get())
            if arrival < 0 or burst <= 0:
                raise ValueError("Invalid input: Arrival >= 0, Burst > 0")
            self.processes.append(Process(pid, arrival, burst, priority))

    def run_simulation(self):
        try:
            self.get_processes()
            algo = self.algo_var.get()
            quantum = int(self.quantum_var.get())
            if algo == "FCFS":
                processes, algo_name, timeline = fcfs_scheduler(self.processes[:])
            elif algo == "SJF-NP":
                processes, algo_name, timeline = sjf_non_preemptive(self.processes[:])
            elif algo == "SJF-P":
                processes, algo_name, timeline = sjf_preemptive(self.processes[:])
            elif algo == "RR":
                processes, algo_name, timeline = rr_scheduler(self.processes[:], quantum)
            elif algo == "PR-NP":
                processes, algo_name, timeline = priority_non_preemptive(self.processes[:])
            elif algo == "PR-P":
                processes, algo_name, timeline = priority_preemptive(self.processes[:])
            else:  # Intelligent
                result = intelligent_scheduler(self.processes[:], quantum)
                processes, algo_name, timeline = result if len(result) == 3 else (result[0], result[1], None)
            
            self.timeline = timeline
            self.current_processes = processes
            self.current_algo_name = algo_name
            avg_wait, avg_turn = calculate_metrics(processes)
            self.display_results(processes, algo_name, avg_wait, avg_turn)
            # Remove embedded Gantt chart from main window
            if self.canvas_widget:
                self.canvas_widget.get_tk_widget().destroy()
                self.canvas_widget = None
        
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def view_gantt_chart(self):
        if not hasattr(self, 'timeline') or not self.timeline:
            messagebox.showwarning("Warning", "Run a simulation first!")
            return
        self.plot_gantt(self.current_processes, self.current_algo_name, self.timeline, animate=False)

    def start_animation(self):
        if not hasattr(self, 'timeline') or not self.timeline:
            messagebox.showwarning("Warning", "Run a simulation first!")
            return
        if self.anim_running:
            self.stop_animation()
        self.plot_gantt(self.current_processes, self.current_algo_name, self.timeline, animate=True)

    def stop_animation(self):
        if self.anim and self.anim_running:
            self.anim.event_source.stop()
            self.anim_running = False
            if self.gantt_window:
                self.gantt_window.destroy()
                self.gantt_window = None

    def compare_all(self):
        try:
            self.get_processes()
            quantum = int(self.quantum_var.get())
            algorithms = [
                fcfs_scheduler, sjf_non_preemptive, sjf_preemptive,
                lambda p: rr_scheduler(p, quantum), priority_non_preemptive, priority_preemptive
            ]
            results = {}
            for algo in algorithms:
                proc_copy = [Process(p.pid, p.arrival_time, p.burst_time, p.priority) for p in self.processes]
                proc_result, algo_name, _ = algo(proc_copy)
                avg_wait, _ = calculate_metrics(proc_result)
                results[algo_name] = avg_wait
            self.plot_comparison(results)
        
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def display_results(self, processes, algo_name, avg_wait, avg_turn):
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Algorithm: {algo_name}\n")
        self.result_text.insert(tk.END, f"Avg Waiting Time: {avg_wait:.2f}\n")
        self.result_text.insert(tk.END, f"Avg Turnaround Time: {avg_turn:.2f}\n")
        self.result_text.insert(tk.END, "-" * 60 + "\n")
        self.result_text.insert(tk.END, "PID | Start | End | Waiting | Turnaround\n")
        for p in processes:
            self.result_text.insert(tk.END, f"{p.pid:<5} | {p.start_time:<6} | {p.end_time:<4} | {p.waiting_time:<7} | {p.turnaround_time}\n")

    def plot_gantt(self, processes, algo_name, timeline, animate=False):
        if self.gantt_window:
            self.gantt_window.destroy()
        
        self.gantt_window = tk.Toplevel(self.root)
        self.gantt_window.title(f"Gantt Chart - {algo_name}")
        self.gantt_window.geometry("1000x600")
        
        fig, ax = plt.subplots(figsize=(14, 6), facecolor="#f0f8ff")  # Increased figure size
        colors = plt.cm.Set3(np.linspace(0, 1, len(processes)))
        
        ax.set_ylim(0, 1.5)
        max_time = max(t[2] for t in timeline) + 1 if timeline else max(p.end_time for p in processes) + 1
        ax.set_xlim(0, max_time)
        ax.set_xlabel("Time (ms)", fontsize=16, color="#333333", fontweight='bold')
        ax.set_yticks([])
        ax.set_title(f"Gantt Chart - {algo_name}", fontsize=20, color="#333333", pad=20, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5, color='gray')
        ax.set_facecolor("#e6f0ff")  # Lighter background for contrast
        
        legend_elements = [plt.Line2D([0], [0], marker='s', color=colors[i], label=p.pid, 
                                     markersize=15, markerfacecolor=colors[i], markeredgecolor='black') 
                          for i, p in enumerate(processes)]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1.15), ncol=len(processes), 
                  frameon=True, fontsize=12, title="Processes", title_fontsize=14, facecolor="#f0f8ff", edgecolor='black')

        if animate:
            self.anim_running = True
            
            def update(frame):
                ax.clear()
                ax.set_ylim(0, 1.5)
                ax.set_xlim(0, max_time)
                ax.set_xlabel("Time (ms)", fontsize=16, color="#333333", fontweight='bold')
                ax.set_yticks([])
                ax.set_title(f"Gantt Chart - {algo_name}", fontsize=20, color="#333333", pad=20, fontweight='bold')
                ax.grid(True, linestyle='--', alpha=0.5, color='gray')
                ax.set_facecolor("#e6f0ff")
                ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1.15), ncol=len(processes), 
                          frameon=True, fontsize=12, title="Processes", title_fontsize=14, facecolor="#f0f8ff", edgecolor='black')
                
                for pid, start, end in timeline[:frame + 1]:
                    idx = [p.pid for p in processes].index(pid)
                    color = colors[idx]
                    ax.broken_barh([(start, end - start)], (0, 1), facecolors=color, edgecolors='black', linewidth=2, alpha=0.9)
                    ax.text(start + (end - start) / 2, 0.5, pid, ha='center', va='center', fontsize=14, color='white', fontweight='bold')

            self.anim = FuncAnimation(fig, update, frames=len(timeline), interval=500, repeat=False)
        else:
            for pid, start, end in timeline:
                idx = [p.pid for p in processes].index(pid)
                color = colors[idx]
                ax.broken_barh([(start, end - start)], (0, 1), facecolors=color, edgecolors='black', linewidth=2, alpha=0.9)
                ax.text(start + (end - start) / 2, 0.5, pid, ha='center', va='center', fontsize=14, color='white', fontweight='bold')

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.gantt_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def plot_comparison(self, results):
        fig, ax = plt.subplots(figsize=(12, 6), facecolor="#f0f8ff")
        algos = list(results.keys())
        waits = list(results.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(algos)))
        bars = ax.bar(algos, waits, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_ylabel("Average Waiting Time (ms)", fontsize=14, color="#333333")
        ax.set_title("Algorithm Performance Comparison", fontsize=16, color="#333333", pad=15)
        plt.xticks(rotation=45, ha="right", fontsize=12, color="#333333")
        plt.yticks(fontsize=12, color="#333333")
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', 
                    ha='center', va='bottom', fontsize=12, color='black', fontweight='bold')
        
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax.set_facecolor("#f0f8ff")
        plt.tight_layout()
        plt.show()

    def export_results(self):
        if not self.processes:
            messagebox.showwarning("Warning", "Run a simulation first!")
            return
        filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["PID", "Arrival", "Burst", "Priority", "Start", "End", "Waiting", "Turnaround"])
            for p in self.processes:
                writer.writerow([p.pid, p.arrival_time, p.burst_time, p.priority, p.start_time, p.end_time, p.waiting_time, p.turnaround_time])
        messagebox.showinfo("Success", f"Results exported to {filename}")

    def reset(self):
        while len(self.process_entries) > 1:
            self.remove_process_row(0)
        pid_entry, arrival_entry, burst_entry, priority_entry, _ = self.process_entries[0]
        arrival_entry.delete(0, tk.END)
        burst_entry.delete(0, tk.END)
        priority_entry.delete(0, tk.END)
        arrival_entry.insert(0, "0")
        burst_entry.insert(0, "0")
        priority_entry.insert(0, "0")
        self.result_text.delete(1.0, tk.END)
        if self.canvas_widget:
            self.canvas_widget.get_tk_widget().destroy()
            self.canvas_widget = None
        if self.gantt_window:
            self.gantt_window.destroy()
            self.gantt_window = None
        self.anim_running = False
        self.anim = None
        self.timeline = None

    def add_tooltip(self, widget, text):
        def enter(event):
            tooltip = tk.Toplevel(widget)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{widget.winfo_rootx() + 10}+{widget.winfo_rooty() + 20}")
            label = tk.Label(tooltip, text=text, background="lightyellow", relief="solid", borderwidth=1, font=("Helvetica", 9))
            label.pack(padx=5, pady=2)
            widget.tooltip = tooltip
        
        def leave(event):
            if hasattr(widget, 'tooltip') and widget.tooltip:
                widget.tooltip.destroy()
                del widget.tooltip
        
        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

if __name__ == "__main__":
    root = ThemedTk(theme="radiance")
    app = SchedulerGUI(root)
    root.mainloop()
