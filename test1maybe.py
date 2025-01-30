# -*- coding: utf-8 -*-
#!/usr/bin/env python3

##############################
# IMPORTS
##############################
import os
import sys
import json
import math
import time
import random
import logging
import threading
import queue
import glob
import pickle
import sqlite3
import traceback
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Audio
try:
    import sounddevice as sd
except ImportError:
    sd = None
    logging.warning("sounddevice is not installed. Audio processing will not be available.")

# Plotting / UI
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Performance acceleration and spatial tree
from numba import njit, prange
from scipy.spatial import cKDTree

# Extra modules for offline AI functionality
import tkinter as tk
from tkinter import scrolledtext
import pygame
import speech_recognition as sr
import pyttsx3
from PIL import Image
import collections

# Optional: Camera capture via OpenCV
try:
    import cv2
except ImportError:
    cv2 = None
    logging.warning("OpenCV is not installed. Camera functionality will not be available.")

# Optional: Mayavi for volumetric heatmaps
try:
    from mayavi import mlab
except ImportError:
    mlab = None
    logging.warning("Mayavi is not installed. Volumetric heatmap functionality will not be available.")

##############################
# GLOBAL CONSTANTS & LOGGING SETUP
##############################
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("CosmicSynapse")

# Simulation / Physics Constants
G = 6.67430e-11         # Gravitational constant (m^3 kg^-1 s^-2)
c = 3.0e8               # Speed of light (m/s)
h = 6.626e-34           # Planck's constant (J s)
k_B = 1.381e-23         # Boltzmann's constant (J/K)
E_0 = 1.0e3             # Reference energy (J)
alpha_initial = 1.0e-10   # Scaling factor
a_0 = 9.81              # Characteristic acceleration (m/s^2)
lambda_evo_initial = 1.0  # Evolution factor
E_replicate = 1e50      # Replication threshold (J)
MAX_PARTICLES = 1000    # Maximum number of particles

DENSITY_PROFILE_PARAMS = {
    'rho0': 1e-24,        # Central density (kg/m^3)
    'rs': 1e21            # Scale radius (m)
}

# Audio settings
AUDIO_SAMPLERATE = 44100  
AUDIO_DURATION = 0.1     # seconds
FREQ_BINS = [500, 2000, 4000, 8000]

# Data directory (use raw string to avoid escape issues)
DATA_DIRECTORY = r"C:\Users\phera\Desktop\test\data"

# Cosmic Brain File
COSMIC_BRAIN_FILE = "cosmic_brain.json"

# Ensure Streamlit session state variables
if "stop_threads" not in st.session_state:
    st.session_state["stop_threads"] = False
if "plot_queue" not in st.session_state:
    st.session_state["plot_queue"] = queue.Queue()

# Global constants for extra mode (Pygame window)
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600

##############################
# MATH SYSTEM CLASS
##############################
class MathSystem:
    """
    Provides mathematical functions for evolving particle properties.
    """
    def __init__(self):
        self.parameters = {
            "base": 2,
            "offset": 1,
            "multiplier": 1,
            "chaos_factor": 0.5
        }
    
    def calculate(self, x):
        p = self.parameters
        chaos = math.sin(p["chaos_factor"] * x)
        return p["base"] ** (p["multiplier"] * x + p["offset"] + chaos)
    
    def evolve(self, dominant_freq, feedback=None):
        if feedback and "math_update" in feedback:
            self.parameters.update(feedback["math_update"])
        self.parameters["multiplier"] = 1 + (dominant_freq % 10) / 10
        self.parameters["offset"] = math.sin(dominant_freq / 100)
        self.parameters["base"] = max(1.1, 2 + (dominant_freq / 200))
        self.parameters["chaos_factor"] = 0.5 + math.cos(dominant_freq / 50)

math_system = MathSystem()

##############################
# NEURAL NETWORK DEFINITION
##############################
class ParticleNeuralNet(nn.Module):
    """
    Neural network for particle adaptation.
    """
    def __init__(self, input_size=15, hidden_size=64, output_size=2):
        super(ParticleNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

GLOBAL_MODEL = ParticleNeuralNet()
GLOBAL_OPTIMIZER = optim.SGD(GLOBAL_MODEL.parameters(), lr=0.001)

##############################
# FILE I/O FUNCTIONS
##############################
def scan_directory(directory_path: str) -> List[str]:
    """
    Scans the given directory recursively for JSON, DB, and BA2 files.
    """
    try:
        json_files = glob.glob(os.path.join(directory_path, '**', '*.json'), recursive=True)
        db_files = glob.glob(os.path.join(directory_path, '**', '*.db'), recursive=True)
        ba2_files = glob.glob(os.path.join(directory_path, '**', '*.ba2'), recursive=True)
        all_files = json_files + db_files + ba2_files
        logger.info(f"Found {len(json_files)} JSON, {len(db_files)} DB, and {len(ba2_files)} BA2 files in {directory_path}.")
        return all_files
    except Exception as e:
        logger.error(f"Error scanning directory {directory_path}: {e}")
        return []

def load_json_files(json_files: List[str]) -> List:
    """
    Loads and parses JSON files.
    """
    data = []
    for file in json_files:
        try:
            with open(file, 'r', encoding='utf-8', errors="replace") as f:
                data.append(json.load(f))
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {file}: {e}. Skipping this file.")
        except Exception as e:
            logger.error(f"Unexpected error loading {file}: {e}. Skipping this file.")
    logger.info(f"Loaded data from {len(data)} of {len(json_files)} JSON files.")
    return data

def load_db_files(db_files: List[str]) -> List:
    """
    Loads data from DB files by querying the 'data' table.
    """
    data = []
    for file in db_files:
        try:
            conn = sqlite3.connect(file)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='data';")
            if cursor.fetchone():
                try:
                    cursor.execute("SELECT * FROM data;")
                    rows = cursor.fetchall()
                    data.extend(rows)
                    logger.info(f"Loaded {len(rows)} records from {file}.")
                except Exception as e:
                    logger.error(f"Error querying 'data' table in {file}: {e}. Skipping this file.")
            else:
                logger.warning(f"No 'data' table found in {file}. Skipping.")
            conn.close()
        except Exception as e:
            logger.error(f"Error loading DB file {file}: {e}")
    logger.info(f"Loaded data from {len(db_files)} DB files.")
    return data

def load_ba2_file(file_path: str):
    """
    Loads BA2 files as binary data.
    """
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        logger.info(f"Loaded BA2 file: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading BA2 file {file_path}: {e}")
        return None

def scan_data_for_all() -> List:
    """
    Scans the data directory and loads all relevant data.
    """
    all_files = scan_directory(DATA_DIRECTORY)
    json_data = load_json_files([f for f in all_files if f.endswith('.json')])
    db_data = load_db_files([f for f in all_files if f.endswith('.db')])
    ba2_data = [load_ba2_file(f) for f in all_files if f.endswith('.ba2') and load_ba2_file(f) is not None]
    return json_data + db_data + ba2_data

def load_simulation_data(filename: str) -> dict:
    """
    Loads simulation data from a pickle file.
    """
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Loaded simulation data from {filename}.")
        return data
    except Exception as e:
        logger.error(f"Error loading simulation data from {filename}: {e}")
        return {}

def save_cosmic_brain(frequencies: dict):
    """
    Saves frequency data to the cosmic_brain.json file.
    """
    try:
        if os.path.exists(COSMIC_BRAIN_FILE):
            with open(COSMIC_BRAIN_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {}
        timestamp = time.time()
        data[str(timestamp)] = frequencies
        with open(COSMIC_BRAIN_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Saved frequencies to {COSMIC_BRAIN_FILE} at timestamp {timestamp}.")
    except Exception as e:
        logger.error(f"Error saving to {COSMIC_BRAIN_FILE}: {e}")

##############################
# AUDIO CAPTURE FUNCTIONS
##############################
def audio_capture_thread(simulator, audio_queue, duration, sample_rate):
    """
    Captures audio data and places it in the audio queue.
    """
    if sd is None:
        logger.error("sounddevice is not available. Cannot capture audio.")
        return
    try:
        while simulator.audio_running:
            audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, blocking=True)
            audio_queue.put(audio_data.flatten())
    except Exception as e:
        logger.error(f"Error in audio_capture_thread: {e}")

def process_audio(simulator, audio_queue, processed_audio_queue, sample_rate):
    """
    Processes audio data by performing FFT and extracting frequency magnitudes.
    """
    while simulator.audio_running or not audio_queue.empty():
        try:
            data = audio_queue.get()
            if data is None:
                break
            # Perform FFT
            fft_vals = np.fft.rfft(data)
            freqs = np.fft.rfftfreq(len(data), 1.0 / sample_rate)
            mags = np.abs(fft_vals)
            processed_audio_queue.put({'freqs': freqs.tolist(), 'mags': mags.tolist()})
        except Exception as e:
            logger.error(f"Error processing audio data: {e}")

##############################
# ADVANCED MATH FUNCTIONS (NUMBA)
##############################
@njit(parallel=True)
def compute_connectivity_numba(positions, masses, G, a0, N):
    """
    Computes the connectivity (Omega) between particles based on gravitational interactions.
    """
    Omega = np.zeros(N)
    for i in prange(N):
        for j in range(N):
            if i != j:
                dx = positions[j,0] - positions[i,0]
                dy = positions[j,1] - positions[i,1]
                dz = positions[j,2] - positions[i,2]
                r_sq = dx*dx + dy*dy + dz*dz + 1e-10  # Avoid division by zero
                r = math.sqrt(r_sq)
                Omega[i] += G * masses[j] / r_sq
    # Normalize Omega if needed
    Omega /= a0
    return Omega

@njit
def nfw_profile_density(r: float) -> float:
    """
    Computes the Navarro-Frenk-White (NFW) density profile.
    """
    EPSILON_DM = 1e-7
    if r < EPSILON_DM:
        r = EPSILON_DM
    rs = DENSITY_PROFILE_PARAMS['rs']
    rho0 = DENSITY_PROFILE_PARAMS['rho0']
    return rho0 * (rs / r) / ((1.0 + (r / rs)) ** 2)

@njit
def advanced_dark_matter_potential(mass: float, x: float, y: float, z: float) -> float:
    """
    Computes the dark matter potential based on the NFW profile.
    """
    r = math.sqrt(x*x + y*y + z*z)
    rho_val = nfw_profile_density(r)
    return -G * mass * rho_val * (4.0 * math.pi * r * r)

@njit(parallel=True)
def update_particle_positions_numba(num_particles: int,
                                    masses: np.ndarray,
                                    positions: np.ndarray,
                                    velocities: np.ndarray,
                                    dt: float):
    """
    Updates the positions and velocities of particles based on computed forces.
    """
    for i in prange(num_particles):
        fx = 0.0
        fy = 0.0
        fz = 0.0
        m1 = masses[i]
        x1, y1, z1 = positions[i, 0], positions[i, 1], positions[i, 2]
        for j in range(num_particles):
            if i == j:
                continue
            m2 = masses[j]
            x2, y2, z2 = positions[j, 0], positions[j, 1], positions[j, 2]
            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1
            dist_sq = dx*dx + dy*dy + dz*dz + 1e-10
            dist = math.sqrt(dist_sq)
            f_mag = (G * m1 * m2) / dist_sq
            fx += f_mag * (dx / dist)
            fy += f_mag * (dy / dist)
            fz += f_mag * (dz / dist)
        ax = fx / m1
        ay = fy / m1
        az = fz / m1
        velocities[i, 0] += ax * dt
        velocities[i, 1] += ay * dt
        velocities[i, 2] += az * dt
    for i in prange(num_particles):
        positions[i, 0] += velocities[i, 0] * dt
        positions[i, 1] += velocities[i, 1] * dt
        positions[i, 2] += velocities[i, 2] * dt

##############################
# PARTICLE CLASS
##############################
class Particle:
    """
    Represents a cosmic particle with mass, position, velocity, energy, frequency, memory, entropy, and unique ID.
    """
    _id_counter = 0  # Class variable for unique IDs
    
    def __init__(self, mass, position, velocity, memory_size=10):
        self.id = Particle._id_counter
        Particle._id_counter += 1
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.Ec = 0.5 * self.mass * np.linalg.norm(self.velocity)**2  # Kinetic Energy
        self.Uc = 0.0  # Potential Energy (to be updated)
        self.nu = (self.Ec + self.Uc) / h if h != 0 else 0.0
        self.memory = np.zeros(memory_size)
        self.S = self.compute_entropy()
        self.frequency = 0.0  # Assigned frequency
        self.tokens = []      # List to hold tokens
    
    def update_position(self, force, dt):
        """
        Updates the particle's position and velocity based on the applied force.
        """
        try:
            acceleration = force / self.mass
            self.velocity += acceleration * dt
            self.position += self.velocity * dt
            logger.debug(f"Particle {self.id}: Position updated.")
        except Exception as e:
            logger.error(f"Error updating position for particle {self.id}: {e}")
    
    def update_energy(self, potential_energy):
        """
        Updates the particle's total energy by adding potential energy.
        """
        try:
            kinetic_energy = 0.5 * self.mass * np.linalg.norm(self.velocity)**2
            self.Ec = kinetic_energy if kinetic_energy > 0 else 0.0
            self.Uc = potential_energy
            self.nu = (self.Ec + self.Uc) / h if h != 0 else 0.0
            self.update_entropy()
            logger.debug(f"Particle {self.id}: Energy updated. Kinetic: {self.Ec}, Potential: {self.Uc}")
        except Exception as e:
            logger.error(f"Error updating energy for particle {self.id}: {e}")
    
    def compute_entropy(self):
        """
        Computes the entropy of the particle based on its energy.
        """
        try:
            Ec_safe = self.Ec if self.Ec > 0 else E_0
            return k_B * math.log2(Ec_safe / E_0)
        except Exception as e:
            logger.error(f"Error computing entropy for particle {self.id}: {e}")
            return 0.0
    
    def update_entropy(self):
        """
        Updates the entropy of the particle.
        """
        try:
            self.S = self.compute_entropy()
            logger.debug(f"Particle {self.id}: Entropy updated.")
        except Exception as e:
            logger.error(f"Error updating entropy for particle {self.id}: {e}")
    
    def assign_frequency(self, frequency):
        """
        Assigns a frequency to the particle and generates a corresponding token.
        """
        try:
            self.frequency = frequency
            token = f"Freq_{frequency:.2f}_Particle_{self.id}"
            self.tokens.append(token)
            logger.info(f"Particle {self.id}: Assigned frequency {frequency:.2f} Hz and created token '{token}'.")
        except Exception as e:
            logger.error(f"Error assigning frequency to particle {self.id}: {e}")
    
    def form_face_structure(self, target_position):
        """
        Adjusts the particle's position to form part of a face structure.
        """
        try:
            self.position = np.array(target_position)
            logger.debug(f"Particle {self.id}: Formed face structure at {self.position}.")
        except Exception as e:
            logger.error(f"Error forming face structure for particle {self.id}: {e}")

##############################
# COSMIC NETWORK CLASS
##############################
class CosmicNetwork:
    """
    Calculates cosmic connectivity based on gravitational interactions.
    """
    def __init__(self, particles, a0=a_0):
        self.particles = particles
        self.a0 = a0
    
    def compute_connectivity(self):
        """
        Computes the connectivity (Omega) for each particle.
        """
        try:
            positions = np.array([p.position for p in self.particles])
            masses = np.array([p.mass for p in self.particles])
            N = len(self.particles)
            Omega = compute_connectivity_numba(positions, masses, G, self.a0, N)
            logger.debug(f"Computed connectivity for {N} particles.")
            return Omega
        except Exception as e:
            logger.error(f"Error computing connectivity: {e}")
            return np.zeros(len(self.particles))

##############################
# DYNAMICS CLASS
##############################
class Dynamics:
    """
    Handles dynamics computations.
    """
    def __init__(self, alpha=alpha_initial, lambda_evo_initial=lambda_evo_initial, gamma=0.01):
        self.alpha = alpha
        self.lambda_evo_initial = lambda_evo_initial
        self.gamma = gamma  # Damping factor
    
    def get_lambda_evo(self, t):
        """
        Computes the current lambda evolution factor based on time.
        """
        return self.lambda_evo_initial * math.exp(-self.gamma * t)
    
    def compute_Psi(self, Ec, Omega, positions, masses):
        """
        Computes Psi based on energy and connectivity.
        """
        return self.alpha * Omega * Ec
    
    def compute_forces(self, Psi, positions):
        """
        Computes the forces acting on each particle.
        """
        try:
            N = len(Psi)
            forces = np.zeros((N, 3))
            for i in prange(N):
                for j in range(N):
                    if i != j:
                        delta = positions[j] - positions[i]
                        distance_sq = np.dot(delta, delta) + 1e-10
                        distance = math.sqrt(distance_sq)
                        force_magnitude = G * Psi[i] * Psi[j] / distance_sq
                        forces[i] += force_magnitude * (delta / distance)
            return forces
        except Exception as e:
            logger.error(f"Error computing forces: {e}")
            return np.zeros((len(Psi), 3))

##############################
# DARK MATTER INFLUENCE CLASS
##############################
class DarkMatterInfluence:
    """
    Models dark matter's influence on particle energy.
    """
    def __init__(self, rho_d_func, params=DENSITY_PROFILE_PARAMS):
        self.rho_d_func = rho_d_func
        self.params = params
    
    def compute_delta_E_dark(self, particle_position, mass):
        """
        Computes the change in energy due to dark matter influence.
        """
        try:
            rho_d = self.rho_d_func(particle_position, self.params)
            distance = np.linalg.norm(particle_position) + 1e-10
            U_dark = -G * mass * rho_d / distance
            logger.debug(f"DarkMatterInfluence: U_dark computed for {particle_position}.")
            return U_dark
        except Exception as e:
            logger.error(f"Error computing dark matter influence: {e}")
            return 0.0

##############################
# REPLICATION CLASS
##############################
class Replication:
    """
    Handles particle replication.
    """
    def __init__(self, E_replicate=E_replicate):
        self.E_replicate = E_replicate
    
    def check_and_replicate(self, particles: List[Particle]) -> List[Particle]:
        """
        Checks particles for replication based on energy thresholds and replicates them if necessary.
        """
        new_particles = []
        try:
            for particle in particles:
                if particle.Ec > self.E_replicate and (len(particles) + len(new_particles) < MAX_PARTICLES):
                    new_mass = particle.mass * random.uniform(0.95, 1.05)
                    new_position = particle.position + np.random.normal(0, 1e9, 3)
                    new_velocity = particle.velocity * random.uniform(0.95, 1.05)
                    new_particle = Particle(new_mass, new_position.tolist(), new_velocity.tolist())
                    new_particle.Ec = particle.Ec * 0.5
                    new_particles.append(new_particle)
                    particle.Ec *= 0.5
                    logger.info(f"Particle {particle.id} replicated into {new_particle.id}.")
            return new_particles
        except Exception as e:
            logger.error(f"Error during replication: {e}")
            return new_particles

##############################
# LEARNING MECHANISM CLASS
##############################
class LearningMechanism:
    """
    Updates particle memory based on neighbor data and frequency features.
    """
    def __init__(self, memory_size=10, learning_rate=0.01):
        self.memory_size = memory_size
        self.learning_rate = learning_rate
    
    def update_memory(self, particle: Particle, neighbors: List[Particle], frequency_data=None):
        """
        Updates the memory of a particle based on its neighbors and frequency data.
        """
        try:
            average_Ec = np.mean([n.Ec for n in neighbors]) if neighbors else 0.0
            if frequency_data:
                freqs = frequency_data.get('freqs', [])
                mags = frequency_data.get('mags', [])
                top_indices = np.argsort(mags)[-5:][::-1]
                top_freqs = np.array(freqs)[top_indices]
                top_mags = np.array(mags)[top_indices]
                if np.max(top_mags) > 0:
                    top_mags = top_mags / np.max(top_mags)
                frequency_features = top_mags[:5]
            else:
                frequency_features = np.zeros(5)
            particle.memory = np.roll(particle.memory, -1)
            particle.memory[-5:] = frequency_features
            logger.debug(f"Memory updated for particle {particle.id}.")
        except Exception as e:
            logger.error(f"Error updating memory for particle {particle.id}: {e}")

##############################
# ADAPTIVE BEHAVIOR CLASS
##############################
class AdaptiveBehavior:
    """
    Enables particles to adapt their energy based on memory and neural network output.
    """
    def __init__(self, sensitivity=0.1):
        self.sensitivity = sensitivity
    
    def adapt(self, particle: Particle, neural_net, combined_input):
        """
        Adapts the particle's energy based on neural network output and combined input.
        """
        try:
            recent_input = combined_input.reshape(1, -1)
            neural_input = torch.tensor(recent_input, dtype=torch.float32)
            with torch.no_grad():
                output = neural_net(neural_input).numpy().flatten()
            delta_alpha = output[0]
            particle.Ec += self.sensitivity * delta_alpha * (recent_input.flatten()[-5:].mean() - particle.Ec)
            particle.Ec = max(particle.Ec, 0.0)
            particle.update_entropy()
            logger.debug(f"Particle {particle.id} energy adapted.")
        except Exception as e:
            logger.error(f"Error adapting behavior for particle {particle.id}: {e}")

##############################
# EXTRA FEATURES: FLOWER FORMATION & TEXT-TO-PARTICLE
##############################
def generate_flower_formation(simulator: 'Simulator'):
    """
    Rearranges particles into a flower-like formation.
    """
    num_petals = 6
    num_to_rearrange = min(100, len(simulator.particles))
    center = np.array([0.0, 0.0, 0.0])
    for i in range(num_to_rearrange):
        petal = i % num_petals
        radius = 1e10 + (i // num_petals) * 5e9
        angle = petal * (2 * math.pi / num_petals)
        x = center[0] + radius * math.cos(angle) + random.uniform(-1e9, 1e9)
        y = center[1] + radius * math.sin(angle) + random.uniform(-1e9, 1e9)
        z = center[2] + random.uniform(-1e9, 1e9)
        simulator.particles[i].position = np.array([x, y, z])
    logger.info("Particles rearranged into a flower formation.")

def generate_particles_from_text(simulator: 'Simulator', text: str):
    """
    Generates new particles based on the provided text input.
    """
    try:
        for char in text:
            freq = ord(char) * 10
            mass = np.random.uniform(1e20, 1e25)
            position = np.random.uniform(-1e11, 1e11, 3)
            velocity = np.random.uniform(-1e3, 1e3, 3)
            new_particle = Particle(mass, position, velocity)
            new_particle.nu = freq
            new_particle.Ec = 0.5 * mass * np.linalg.norm(velocity)**2
            new_particle.assign_frequency(freq)
            simulator.particles.append(new_particle)
            simulator.history[-1] = np.vstack([simulator.history[-1], new_particle.position])
        logger.info(f"Generated particles from text: {text}")
    except Exception as e:
        logger.error(f"Error generating particles from text: {e}")

##############################
# VISUAL PARTICLE CLASS (Extra Mode)
##############################
class VisualParticleExtra:
    """
    Represents a visual particle for extra (Pygame) mode.
    """
    def __init__(self, x, y, z, color, life, energy):
        self.x, self.y, self.z = x, y, z
        self.color = color
        self.life = life
        self.energy = energy
        self.radius = 1
    
    def update(self, dx, dy, dz, dominant_freq, target_position=None, motion_modifier=1.0):
        """
        Updates the particle's position and state.
        """
        freq_effect = math_system.calculate(self.energy)
        chaos = math.sin(dominant_freq / 50)
        if target_position:
            self.x += (target_position[0] - self.x) * 0.01
            self.y += (target_position[1] - self.y) * 0.01
            self.z += (target_position[2] - self.z) * 0.01
        else:
            self.x += dx * freq_effect * chaos * motion_modifier
            self.y += dy * freq_effect * motion_modifier
            self.z += dz * freq_effect * chaos * motion_modifier
        self.life -= 1
        self.radius += math.sin(self.energy) * 0.1
    
    def render(self, surface, zoom, camera_pos, rotation):
        """
        Renders the particle on the Pygame surface.
        """
        cx, cy, _ = camera_pos
        x = (self.x - cx) * zoom
        y = (self.y - cy) * zoom
        screen_x = int(WINDOW_WIDTH / 2 + x)
        screen_y = int(WINDOW_HEIGHT / 2 - y)
        pygame.draw.circle(surface, self.color, (screen_x, screen_y), max(1, int(self.radius)))

##############################
# VISUALIZER CLASS (Streamlit & Mayavi)
##############################
class Visualizer:
    """
    Visualizes the simulation using Plotly and Mayavi.
    """
    def __init__(self, simulator: 'Simulator'):
        self.simulator = simulator
    
    def plot_particles(self, step, frequency_data=None):
        """
        Plots the positions of particles at a given simulation step.
        """
        try:
            positions = self.simulator.history[step]
            Ec = np.array([p.Ec for p in self.simulator.particles])
            masses = np.array([p.mass for p in self.simulator.particles])
            norm_energy = Ec / np.max(Ec) if np.max(Ec) > 0 else Ec
            sizes = 2 + (masses / np.max(masses)) * 8 if np.max(masses) > 0 else np.full(masses.shape, 2)
            if frequency_data:
                freqs = frequency_data.get('freqs', [])
                mags = frequency_data.get('mags', [])
                if len(freqs) > 0:
                    dominant_freq = freqs[np.argmax(mags)]
                    hue = (dominant_freq / 20000) * 360
                    # Convert hue to RGB using matplotlib's hsv colormap
                    color = plt.cm.hsv(hue / 360)
                    colors = np.tile(color[:3], (len(self.simulator.particles), 1))
                else:
                    colors = plt.cm.viridis(norm_energy)
            else:
                colors = plt.cm.viridis(norm_energy)
            camera1 = dict(eye=dict(x=1.25, y=1.25, z=1.25))
            camera2 = dict(eye=dict(x=-1.25, y=1.25, z=1.25))
            fig = go.Figure(data=[go.Scatter3d(
                x=positions[:,0],
                y=positions[:,1],
                z=positions[:,2],
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=colors[:,0] if colors.ndim == 2 else colors,
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(title='Normalized Energy')
                )
            )])
            fig.update_layout(
                title=f"Time Step: {step}",
                scene=dict(
                    xaxis_title='X Position (m)',
                    yaxis_title='Y Position (m)',
                    zaxis_title='Z Position (m)',
                    bgcolor="black",
                    xaxis=dict(backgroundcolor="black", showbackground=True),
                    yaxis=dict(backgroundcolor="black", showbackground=True),
                    zaxis=dict(backgroundcolor="black", showbackground=True),
                    camera=camera1
                ),
                paper_bgcolor='black',
                font=dict(color='white')
            )
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig.update_layout(scene_camera=camera2)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error plotting particles at step {step}: {e}")
    
    def animate_simulation(self):
        """
        Creates an animation of the simulation over time.
        """
        try:
            frames = []
            for step in range(len(self.simulator.history)):
                positions = self.simulator.history[step]
                Ec = np.array([p.Ec for p in self.simulator.particles])
                norm_energy = Ec / np.max(Ec) if np.max(Ec) > 0 else Ec
                masses = np.array([p.mass for p in self.simulator.particles])
                sizes = 2 + (masses / np.max(masses)) * 8 if np.max(masses) > 0 else np.full(masses.shape, 2)
                freqs = []  # Placeholder for frequency data if needed
                mags = []
                if step < len(self.simulator.audio_history):
                    freqs = self.simulator.audio_history[step].get('freqs', [])
                    mags = self.simulator.audio_history[step].get('mags', [])
                if len(freqs) > 0:
                    dominant_freq = freqs[np.argmax(mags)]
                    hue = (dominant_freq / 20000) * 360
                    color = plt.cm.hsv(hue / 360)
                    colors = np.tile(color[:3], (len(self.simulator.particles), 1))
                else:
                    colors = plt.cm.viridis(norm_energy)
                camera_angle = dict(eye=dict(x=1.25*math.cos(0.1*step), y=1.25*math.sin(0.1*step), z=1.25))
                frame = go.Frame(data=[go.Scatter3d(
                    x=positions[:,0],
                    y=positions[:,1],
                    z=positions[:,2],
                    mode='markers',
                    marker=dict(
                        size=sizes,
                        color=colors[:,0] if colors.ndim == 2 else colors,
                        colorscale='Viridis',
                        opacity=0.8,
                        showscale=False
                    )
                )],
                layout=go.Layout(scene_camera=camera_angle),
                name=str(step))
                frames.append(frame)
            fig = go.Figure(
                data=[go.Scatter3d(
                    x=self.simulator.history[0][:,0],
                    y=self.simulator.history[0][:,1],
                    z=self.simulator.history[0][:,2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=np.zeros(len(self.simulator.particles)),
                        colorscale='Viridis',
                        opacity=0.8,
                        colorbar=dict(title='Normalized Energy')
                    )
                )],
                layout=go.Layout(
                    title="Cosmic Synapse Simulation Animation",
                    updatemenus=[dict(
                        type="buttons",
                        buttons=[dict(label="Play",
                                      method="animate",
                                      args=[None, {"frame": {"duration": 50, "redraw": True},
                                               "fromcurrent": True,
                                               "transition": {"duration": 0}}])])],
                    scene=dict(
                        xaxis_title='X Position (m)',
                        yaxis_title='Y Position (m)',
                        zaxis_title='Z Position (m)',
                        bgcolor="black",
                        xaxis=dict(backgroundcolor="black", showbackground=True),
                        yaxis=dict(backgroundcolor="black", showbackground=True),
                        zaxis=dict(backgroundcolor="black", showbackground=True),
                        camera=dict(eye=dict(x=1.25, y=1.25, z=1.25))
                    ),
                    paper_bgcolor='black',
                    font=dict(color='white')
                ),
                frames=frames
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error creating animation: {e}")
    
    def plot_volumetric_heatmap(self, step):
        """
        Plots a volumetric heatmap of particle energies using Mayavi.
        """
        try:
            if mlab is None:
                st.error("Mayavi is not installed. Cannot render volumetric heatmap.")
                return
            positions = self.simulator.history[step]
            Ec = np.array([p.Ec for p in self.simulator.particles])
            grid_size = 100
            x = np.linspace(np.min(positions[:,0]), np.max(positions[:,0]), grid_size)
            y = np.linspace(np.min(positions[:,1]), np.max(positions[:,1]), grid_size)
            z = np.linspace(np.min(positions[:,2]), np.max(positions[:,2]), grid_size)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            density = np.zeros_like(X)
            for pos, energy in zip(positions, Ec):
                ix = np.searchsorted(x, pos[0]) - 1
                iy = np.searchsorted(y, pos[1]) - 1
                iz = np.searchsorted(z, pos[2]) - 1
                if 0 <= ix < grid_size and 0 <= iy < grid_size and 0 <= iz < grid_size:
                    density[ix, iy, iz] += energy
            density /= np.max(density) if np.max(density) > 0 else 1
            mlab.figure(bgcolor=(0, 0, 0))
            src = mlab.pipeline.scalar_field(X, Y, Z, density)
            mlab.pipeline.volume(src, vmin=0.1, vmax=1.0, opacity='auto')
            mlab.title(f"Volumetric Heatmap - Step {step}", color=(1,1,1), size=0.5)
            mlab.show()
        except Exception as e:
            logger.error(f"Error creating volumetric heatmap at step {step}: {e}")
    
    def plot_metrics(self):
        """
        Plots various simulation metrics over time.
        """
        try:
            steps_arr = np.arange(1, len(self.simulator.total_energy_history) + 1) * self.simulator.dt
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))
            axs[0,0].plot(steps_arr, self.simulator.total_energy_history, label='Total Energy', color='blue')
            axs[0,0].set_xlabel('Time (s)')
            axs[0,0].set_ylabel('Energy (J)')
            axs[0,0].set_title('Total Energy Over Time')
            axs[0,0].legend()
            axs[0,0].grid(True)
            axs[0,1].plot(steps_arr, self.simulator.kinetic_energy_history, label='Kinetic Energy', color='green')
            axs[0,1].plot(steps_arr, self.simulator.potential_energy_history, label='Potential Energy', color='red')
            axs[0,1].set_xlabel('Time (s)')
            axs[0,1].set_ylabel('Energy (J)')
            axs[0,1].set_title('Kinetic and Potential Energy Over Time')
            axs[0,1].legend()
            axs[0,1].grid(True)
            axs[1,0].plot(steps_arr, self.simulator.entropy_history, label='Average Entropy', color='purple')
            axs[1,0].set_xlabel('Time (s)')
            axs[1,0].set_ylabel('Entropy (J)')
            axs[1,0].set_title('Average Entropy Over Time')
            axs[1,0].legend()
            axs[1,0].grid(True)
            particle_counts = [len(p) for p in self.simulator.history]
            axs[1,1].plot(steps_arr, particle_counts, label='Number of Particles', color='orange')
            axs[1,1].set_xlabel('Time (s)')
            axs[1,1].set_ylabel('Count')
            axs[1,1].set_title('Number of Particles Over Time')
            axs[1,1].legend()
            axs[1,1].grid(True)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            logger.error(f"Error plotting metrics: {e}")

##############################
# SIMULATOR CLASS (Main Simulation)
##############################
class Simulator:
    """
    Orchestrates the simulation, integrating all components.
    """
    """
    Notes:
    - Particles will react to audio frequencies and camera brightness.
    - Specific particles will form a mouth structure based on audio intensity.
    - The Tkinter logs window will display audio frequencies and camera brightness data.
    """
    def __init__(self, num_particles=100, steps=1000, dt=1.0, data_directory=DATA_DIRECTORY):
        self.particles = self.initialize_particles(num_particles)
        self.network = CosmicNetwork(self.particles)
        self.dynamics = Dynamics()
        self.dark_matter = DarkMatterInfluence(self.density_function)
        self.replication = Replication()
        self.learning = LearningMechanism()
        self.adaptive = AdaptiveBehavior()
        self.visualizer = Visualizer(self)
        self.steps = steps
        self.dt = dt
        self.time = 0
        self.history = [np.array([p.position for p in self.particles])]
        self.total_energy_history = []
        self.kinetic_energy_history = []
        self.potential_energy_history = []
        self.entropy_history = []
        self.octree = None
        self.octree_size = 0
        self.audio_queue = queue.Queue()
        self.processed_audio_queue = queue.Queue()
        self.audio_thread = None
        self.processing_thread = None
        self.audio_running = False
        # Camera integration variables
        self.camera_running = False
        self.camera_queue = queue.Queue()
        self.processed_camera_queue = queue.Queue()
        self.camera_thread = None
        self.data_directory = data_directory
        # Cosmic Brain Frequency Data
        self.cosmic_brain = self.load_cosmic_brain()
        # Audio History for Visualization
        self.audio_history = []
        logger.info(f"Initialized {num_particles} particles.")
    
    def initialize_particles(self, N) -> List[Particle]:
        """
        Initializes particles with random masses, positions, and velocities.
        """
        try:
            masses = np.random.uniform(1e20, 1e25, N)
            positions = np.random.uniform(-1e11, 1e11, (N, 3))
            velocities = np.random.uniform(-1e3, 1e3, (N, 3))
            particles = [Particle(m, pos, vel) for m, pos, vel in zip(masses, positions, velocities)]
            return particles
        except Exception as e:
            logger.error(f"Error initializing particles: {e}")
            return []
    
    def density_function(self, position, params) -> float:
        """
        Computes the dark matter density at a given position using the NFW profile.
        """
        try:
            rho0 = params.get('rho0', 1e-24)
            rs = params.get('rs', 1e21)
            r = np.linalg.norm(position) + 1e-10
            return rho0 / ((r/rs) * (1 + r/rs)**2)
        except Exception as e:
            logger.error(f"Error in density function: {e}")
            return 0.0
    
    def compute_total_energy(self) -> float:
        """
        Computes the total energy of the system, including kinetic and potential energies.
        """
        try:
            kinetic_energy = np.sum([0.5 * p.mass * np.linalg.norm(p.velocity)**2 for p in self.particles])
            potential_energy = 0.0
            N = len(self.particles)
            positions = np.array([p.position for p in self.particles])
            masses = np.array([p.mass for p in self.particles])
            for i in range(N):
                r_i = positions[i]
                delta = positions[i+1:] - r_i
                distances = np.linalg.norm(delta, axis=1) + 1e-10
                potential_energy -= np.sum(G * masses[i] * masses[i+1:] / distances)
            total_Ec = np.sum([p.Ec for p in self.particles])
            total_energy = kinetic_energy + potential_energy + total_Ec
            self.total_energy_history.append(total_energy)
            self.kinetic_energy_history.append(kinetic_energy)
            self.potential_energy_history.append(potential_energy)
            avg_entropy = np.mean([p.S for p in self.particles])
            self.entropy_history.append(avg_entropy)
            logger.debug(f"Computed total energy: {total_energy} J")
            return total_energy
        except Exception as e:
            logger.error(f"Error computing total energy: {e}")
            return 0.0
    
    def run_step(self, neural_net, frequency_data=None, camera_data=None):
        """
        Executes a single simulation step, updating particle states and processing interactions.
        """
        try:
            # Scan directory for data files
            all_files = scan_directory(self.data_directory)
            json_files = [f for f in all_files if f.endswith('.json')]
            db_files = [f for f in all_files if f.endswith('.db')]
            json_data = load_json_files(json_files)
            db_data = load_db_files(db_files)
            combined_data = json_data + db_data  # (For potential future use)
            
            # Process frequency data
            if frequency_data:
                freqs = frequency_data.get('freqs', [])
                mags = frequency_data.get('mags', [])
                top_indices = np.argsort(mags)[-len(FREQ_BINS):][::-1]
                top_freqs = np.array(freqs)[top_indices]
                top_mags = np.array(mags)[top_indices]
                if len(top_freqs) < len(FREQ_BINS):
                    # Pad with zeros if not enough frequencies
                    pad_length = len(FREQ_BINS) - len(top_freqs)
                    top_freqs = np.pad(top_freqs, (0, pad_length), 'constant')
                    top_mags = np.pad(top_mags, (0, pad_length), 'constant')
                else:
                    top_freqs = top_freqs[:len(FREQ_BINS)]
                    top_mags = top_mags[:len(FREQ_BINS)]
                # Normalize magnitudes
                if np.max(top_mags) > 0:
                    top_mags = top_mags / np.max(top_mags)
                else:
                    top_mags = top_mags
                # Save frequencies to cosmic_brain
                frequencies = {"freqs": top_freqs.tolist(), "mags": top_mags.tolist()}
                save_cosmic_brain(frequencies)
                self.audio_history.append(frequencies)
            else:
                top_freqs = np.zeros(len(FREQ_BINS))
                top_mags = np.zeros(len(FREQ_BINS))
            
            # Assign frequencies to particles
            frequency_particle_map = {}
            for i, freq in enumerate(top_freqs):
                if i < len(self.particles):
                    particle = self.particles[i]
                    particle.assign_frequency(freq)
                    frequency_particle_map[freq] = particle
                else:
                    break  # No more particles to assign
            
            # Generate tokens based on frequencies
            tokens = [f"Freq_{freq:.2f}_Particle_{frequency_particle_map[freq].id}" for freq in top_freqs if freq > 0]
            
            # Prepare NN input: memory (10) + top_mags (4)
            # Adjusted input size based on memory_size (10) + len(FREQ_BINS) (4)
            input_vectors = [np.concatenate((p.memory, top_mags)) for p in self.particles]
            neural_input = np.array(input_vectors)
            neural_input_tensor = torch.tensor(neural_input, dtype=torch.float32)
            
            # Compute connectivity
            Omega = self.network.compute_connectivity()
            Ec = np.array([p.Ec for p in self.particles])
            positions = np.array([p.position for p in self.particles])
            masses = np.array([p.mass for p in self.particles])
            
            # Compute Psi
            Psi = self.dynamics.compute_Psi(Ec, Omega, positions, masses)
            
            # Compute dark matter influence
            delta_E_dark = np.array([self.dark_matter.compute_delta_E_dark(p.position, p.mass) for p in self.particles])
            for i, particle in enumerate(self.particles):
                particle.Ec += delta_E_dark[i]
                if particle.Ec < 0:
                    particle.Ec = 0.0
            
            # Get current lambda value
            current_lambda = self.dynamics.get_lambda_evo(self.time)
            self.dynamics.alpha = current_lambda  # Example usage
            
            # Compute forces
            forces = self.dynamics.compute_forces(Psi, positions)
            
            # Update particle positions and energies
            for i, particle in enumerate(self.particles):
                particle.update_position(forces[i], self.dt)
                # Calculate potential energy (simplified example)
                potential_energy = -G * np.sum([
                    other_p.mass / (np.linalg.norm(particle.position - other_p.position) + 1e-10)
                    for j, other_p in enumerate(self.particles) if j != i
                ])
                particle.update_energy(potential_energy)
    
            # Learning and adaptation...
            for particle in self.particles:
                neighbors = self.find_neighbors(particle)
                self.learning.update_memory(particle, neighbors, frequency_data=frequency_data)
                combined_input = np.concatenate((particle.memory, top_mags))
                self.adaptive.adapt(particle, neural_net, combined_input)
            
            # Replication and data token processing...
            new_particles = self.replication.check_and_replicate(self.particles)
            self.particles.extend(new_particles)
            
            # Cross-reference frequency data to generate particles from tokens
            if tokens:
                for token in tokens:
                    word = self.map_token_to_word(token)
                    if word:
                        data_tokens = self.scan_data_for_word(word)
                        for data_token in data_tokens:
                            self.generate_particles_from_token(data_token)
                        # Form specific structures based on word
                        if word.lower() == "flower":
                            generate_flower_formation(self)
                        generate_particles_from_text(self, word)
                        if word.lower() in ["hello", "world", "face", "mouth"]:
                            self.form_face()
            
            # Handle camera data for real-time particle formation
            if camera_data:
                self.process_camera_data(camera_data)
            
            # Update history and compute total energy
            self.history.append(np.array([p.position for p in self.particles]))
            self.compute_total_energy()
            self.time += self.dt
            logger.info(f"Completed step {self.time} with {len(self.particles)} particles.")
        except Exception as e:
            logger.error(f"Error during simulation step: {e}")
    
    def find_neighbors(self, particle, radius=1e10) -> List[Particle]:
        """
        Finds neighboring particles within a specified radius using a spatial tree.
        """
        try:
            if not self.octree or self.octree_size != len(self.particles):
                positions = np.array([p.position for p in self.particles])
                self.octree = cKDTree(positions)
                self.octree_size = len(self.particles)
            indices = self.octree.query_ball_point(particle.position, r=radius)
            return [self.particles[i] for i in indices if self.particles[i] != particle]
        except Exception as e:
            logger.error(f"Error finding neighbors for particle {particle.id}: {e}")
            return []
    
    def map_token_to_word(self, token: str) -> str:
        """
        Maps a token to a word based on predefined rules.
        """
        # Example mapping based on frequency or token content
        if "flower" in token.lower():
            return "flower"
        elif "hello" in token.lower():
            return "hello"
        elif "world" in token.lower():
            return "world"
        elif "face" in token.lower():
            return "face"
        elif "mouth" in token.lower():
            return "mouth"
        else:
            return "unknown"
    
    def scan_data_for_word(self, word: str) -> List:
        """
        Scans data files for the given word and returns relevant tokens.
        """
        data_tokens = []
        try:
            all_files = scan_directory(self.data_directory)
            json_files = [f for f in all_files if f.endswith('.json')]
            db_files = [f for f in all_files if f.endswith('.db')]
            for file in json_files:
                try:
                    with open(file, 'r', encoding='utf-8', errors="replace") as f:
                        content = f.read()
                        if word.lower() in content.lower():
                            data_tokens.append({'file': file, 'content': content})
                except Exception as e:
                    logger.error(f"Error scanning JSON file {file}: {e}")
            for file in db_files:
                try:
                    conn = sqlite3.connect(file)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='data';")
                    if cursor.fetchone():
                        cursor.execute("SELECT * FROM data;")
                        rows = cursor.fetchall()
                        for row in rows:
                            for item in row:
                                if isinstance(item, str) and word.lower() in item.lower():
                                    data_tokens.append({'file': file, 'content': item})
                    conn.close()
                except Exception as e:
                    logger.error(f"Error scanning DB file {file}: {e}")
            logger.info(f"Found {len(data_tokens)} tokens for word '{word}'.")
        except Exception as e:
            logger.error(f"Error scanning data for word '{word}': {e}")
        return data_tokens
    
    def generate_particles_from_token(self, token):
        """
        Generates new particles based on the content of a token.
        """
        try:
            content = token.get('content', '') if isinstance(token, dict) else str(token)
            for char in content:
                freq = ord(char) * 10
                mass = np.random.uniform(1e20, 1e25)
                pos = np.random.uniform(-1e11, 1e11, 3)
                vel = np.random.uniform(-1e3, 1e3, 3)
                new_particle = Particle(mass, pos, vel)
                new_particle.nu = freq
                new_particle.Ec = 0.5 * mass * np.linalg.norm(vel)**2
                new_particle.assign_frequency(freq)
                self.particles.append(new_particle)
                self.history[-1] = np.vstack([self.history[-1], new_particle.position])
            logger.info("Generated new particle(s) from token.")
        except Exception as e:
            logger.error(f"Error generating particle from token: {e}")
    
    def form_face(self):
        """
        Forms a facial structure with particles based on learned frequencies.
        """
        try:
            # Simple face structure: eyes, nose, mouth
            face_positions = [
                # Eyes
                (-1e10, 2e10, 0),
                (1e10, 2e10, 0),
                # Nose
                (0, 0, 0),
                # Mouth corners
                (-1e10, -2e10, 0),
                (1e10, -2e10, 0)
            ]
            for i, pos in enumerate(face_positions):
                if i < len(self.particles):
                    self.particles[i].form_face_structure(pos)
            logger.info("Particles arranged to form a face structure.")
        except Exception as e:
            logger.error(f"Error forming face structure: {e}")
    
    def process_camera_data(self, camera_data):
        """
        Processes camera data to influence particle behaviors.
        """
        try:
            # Example: Use average brightness to adjust particle colors or positions
            avg_brightness = camera_data.get('brightness', 0.0)
            # Map brightness to some parameter, e.g., movement speed
            movement_speed = avg_brightness * 0.1
            # Influence specific particles (e.g., mouth particles) based on brightness
            mouth_particle_indices = [3, 4]  # Assuming these indices correspond to mouth corners
            for idx in mouth_particle_indices:
                if idx < len(self.particles):
                    particle = self.particles[idx]
                    # Adjust velocity based on brightness
                    particle.velocity += np.random.uniform(-movement_speed, movement_speed, 3)
                    # Optionally, adjust position to simulate opening/closing mouth
                    particle.position[1] += movement_speed * 1e9  # Example adjustment
            logger.info(f"Processed camera data: Avg Brightness = {avg_brightness}")
        except Exception as e:
            logger.error(f"Error processing camera data: {e}")
    
    def load_cosmic_brain(self) -> dict:
        """
        Loads the cosmic_brain.json file containing frequency data.
        """
        try:
            if os.path.exists(COSMIC_BRAIN_FILE):
                with open(COSMIC_BRAIN_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Loaded cosmic brain data from {COSMIC_BRAIN_FILE}.")
                return data
            else:
                logger.info(f"No existing cosmic brain data found. Starting fresh.")
                return {}
        except Exception as e:
            logger.error(f"Error loading cosmic brain data: {e}")
            return {}
    
    def start_audio_capture(self, duration=1.0, sample_rate=44100):
        """
        Starts the audio capture and processing threads.
        """
        if sd is None:
            logger.error("sounddevice is not installed. Cannot start audio capture.")
            return
        if not self.audio_running:
            self.audio_running = True
            self.audio_thread = threading.Thread(target=audio_capture_thread,
                                                 args=(self, self.audio_queue, duration, sample_rate),
                                                 daemon=True)
            self.audio_thread.start()
            self.processing_thread = threading.Thread(target=process_audio,
                                                      args=(self, self.audio_queue, self.processed_audio_queue, sample_rate),
                                                      daemon=True)
            self.processing_thread.start()
            logger.info("Started audio capture and processing threads.")
    
    def stop_audio_capture(self):
        """
        Stops the audio capture and processing threads.
        """
        if self.audio_running:
            self.audio_running = False
            self.audio_queue.put(None)
            self.processed_audio_queue.put(None)
            logger.info("Stopped audio capture.")
    
    def get_latest_frequency_data(self):
        """
        Retrieves the latest processed frequency data from the queue.
        """
        try:
            if not self.processed_audio_queue.empty():
                frequency_data = self.processed_audio_queue.get()
                return frequency_data
            else:
                return None
        except Exception as e:
            logger.error(f"Error retrieving frequency data: {e}")
            return None
    
    # Camera integration methods
    def start_camera_capture(self, device=0):
        """
        Starts the camera capture and processing threads.
        """
        if cv2 is None:
            logger.error("OpenCV is not installed. Camera capture is unavailable.")
            return
        if not self.camera_running:
            self.camera_running = True
            self.camera_queue = queue.Queue()
            self.processed_camera_queue = queue.Queue()
            self.camera_thread = threading.Thread(target=camera_capture_thread, args=(self, self.camera_queue, device), daemon=True)
            self.camera_thread.start()
            threading.Thread(target=process_camera, args=(self, self.camera_queue, self.processed_camera_queue), daemon=True).start()
            logger.info("Started camera capture and processing threads.")
    
    def stop_camera_capture(self):
        """
        Stops the camera capture and processing threads.
        """
        if self.camera_running:
            self.camera_running = False
    
    def get_latest_camera_data(self):
        """
        Retrieves the latest processed camera data from the queue.
        """
        try:
            if not self.processed_camera_queue.empty():
                return self.processed_camera_queue.get()
            else:
                return None
        except Exception as e:
            logger.error(f"Error retrieving camera data: {e}")
            return None

##############################
# CAMERA CAPTURE FUNCTIONS
##############################
def camera_capture_thread(simulator, camera_queue, device=0):
    """
    Captures frames from the camera and places them in the queue.
    """
    if cv2 is None:
        logger.error("OpenCV is not available.")
        return
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        logger.error("Failed to open camera.")
        return
    while simulator.camera_running:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to capture camera frame.")
            continue
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert image to grayscale to simplify brightness extraction
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Compute average brightness as a proxy for light data
        avg_brightness = np.mean(gray_frame)
        # Map brightness to a parameter (e.g., influencing particle behavior)
        brightness_param = avg_brightness / 255.0
        frequency_data = {'brightness': brightness_param}
        camera_queue.put(frequency_data)
        # Save frequency data to cosmic brain
        simulator.save_cosmic_brain(frequency_data)
        time.sleep(0.1)  # Adjust as needed for real-time responsiveness
    cap.release()

def process_camera(simulator, camera_queue, processed_camera_queue):
    """
    Processes camera data and places the processed data in the queue.
    """
    while simulator.camera_running or not camera_queue.empty():
        try:
            data = camera_queue.get()
            processed_camera_queue.put(data)
        except Exception as e:
            logger.error(f"Error processing camera data: {e}")

##############################
# EXTRA MODE: Tkinter/Pygame/Speech/TTS
##############################
# Initialize TTS engine
tts_engine = pyttsx3.init()

def create_log_window(log_queue) -> tk.Tk:
    """
    Creates a Tkinter window to display system logs.
    """
    root = tk.Tk()
    root.title("System Logs")
    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=30)
    text_area.pack(expand=True, fill=tk.BOTH)
    
    def update_log():
        while not log_queue.empty():
            log_message = log_queue.get()
            text_area.insert(tk.END, log_message + "\n")
            text_area.see(tk.END)
        root.after(100, update_log)
    
    root.after(100, update_log)
    return root

def visualization_loop(simulator):
    """
    Runs the Pygame visualization loop, rendering particles and evolving the visual representation.
    """
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Math-Driven Creative System")
    clock = pygame.time.Clock()
    running = True
    fullscreen = False
    # Optionally load a sample image
    ba2_files = glob.glob(os.path.join(DATA_DIRECTORY, '**', '*.ba2'), recursive=True)
    image_data = None
    sample_image_path = os.path.join(DATA_DIRECTORY, "sample_image.jpg")
    if ba2_files and os.path.exists(sample_image_path):
        try:
            image = Image.open(sample_image_path).resize((WINDOW_WIDTH, WINDOW_HEIGHT))
            image_data = np.array(image)
            logger.info("Loaded sample image for visualization.")
        except Exception as e:
            logger.error(f"Error loading sample image: {e}")
    while running:
        screen.fill((0, 0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11:
                    fullscreen = not fullscreen
                    if fullscreen:
                        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                    else:
                        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        if not audio_queue_extra.empty():
            dominant_freq = audio_queue_extra.get()
        else:
            dominant_freq = 440  # Default frequency
        
        update_particles_extra(simulator, dominant_freq, image_data)
        
        for p in particles_extra:
            p.render(screen, 1.0, [0, 0, 10], [0, 0])
        
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()

def audio_capture_thread_extra(simulator, audio_queue, duration, sample_rate):
    """
    Captures audio data and places it in the audio queue (Extra Mode).
    """
    try:
        while simulator.audio_running:
            if sd is None:
                logger.error("sounddevice is not available. Cannot capture audio.")
                break
            audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, blocking=True)
            audio_queue.put(audio_data.flatten())
    except Exception as e:
        logger.error(f"Error in audio_capture_thread_extra: {e}")

def audio_callback_extra(indata, frames, time_info, status):
    """
    Callback function for audio input stream in extra mode.
    """
    if status:
        log_queue_extra.put(f"Audio error: {status}")
        return
    fft_result = np.fft.rfft(indata[:, 0])
    freqs = np.fft.rfftfreq(len(indata[:, 0]), 1 / AUDIO_SAMPLERATE)
    magnitudes = np.abs(fft_result)
    if len(freqs) > 0:
        dominant_freq = freqs[np.argmax(magnitudes)]
        audio_queue_extra.put(dominant_freq)
        log_queue_extra.put(f"Dominant Frequency: {dominant_freq:.2f} Hz")
        frequency_history.append(dominant_freq)

def update_particles_extra(simulator, dominant_freq, image_data=None, feedback=None):
    """
    Updates particles in the extra visualization mode based on dominant frequency and image data.
    """
    if feedback is None:
        feedback = {
            "particle_updates": {
                "spawn_multiplier": 1.0,
                "color_adjustments": [0, 0, 0],
                "lifespan_multiplier": 1.0,
                "motion_modifier": 1.0
            }
        }
    spawn_multiplier = feedback["particle_updates"].get("spawn_multiplier", 1.0)
    particle_limit = MAX_PARTICLES - len(particles_extra)
    for _ in range(min(particle_limit, int(50 * spawn_multiplier))):
        x, y, z = np.random.uniform(-10, 10, 3)
        color = [random.randint(0, 255) for _ in range(3)]
        particles_extra.append(VisualParticleExtra(x, y, z, color, random.randint(50, 100), random.uniform(0.5, 1.5)))
    if image_data is not None:
        for i, p in enumerate(particles_extra):
            if i < image_data.shape[0] * image_data.shape[1]:
                x, y = divmod(i, image_data.shape[1])
                color = image_data[x % image_data.shape[0], y % image_data.shape[1]]
                p.color = color.tolist()
                p.update(0, 0, 0, dominant_freq, target_position=(x, y, 0))
            else:
                dx, dy, dz = np.random.uniform(-0.1, 0.1, 3)
                p.update(dx, dy, dz, dominant_freq)
    else:
        for p in particles_extra:
            dx, dy, dz = np.random.uniform(-0.1, 0.1, 3)
            p.update(dx, dy, dz, dominant_freq)
    particles_extra[:] = [p for p in particles_extra if p.life > 0]
    math_system.evolve(dominant_freq, feedback)

def recognize_speech_extra():
    """
    Recognizes speech from the microphone and processes it in extra mode.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            log_queue_extra.put(f"Recognized Speech: {text}")
            learn_from_text_extra(text)
            return text
        except sr.UnknownValueError:
            log_queue_extra.put("Speech Recognition could not understand audio")
        except sr.RequestError as e:
            log_queue_extra.put(f"Speech Recognition error: {e}")
    return ""

def speak_text_extra(text: str):
    """
    Uses TTS to speak the provided text in extra mode.
    """
    tts_engine.say(text)
    tts_engine.runAndWait()

def learn_from_text_extra(text: str):
    """
    Processes learned text by updating the learning data in extra mode.
    """
    try:
        learning_data[text] = f"Learned response to: {text}"
        save_learning_data_extra()
        speak_text_extra(f"I have learned {text}")
    except Exception as e:
        log_queue_extra.put(f"Learning error: {e}")

def save_learning_data_extra():
    """
    Saves the learning data to a JSON file in extra mode.
    """
    try:
        with open("learning_data.json", "w", encoding="utf-8") as f:
            json.dump(learning_data, f, indent=4)
        logger.info("Learning data saved.")
    except Exception as e:
        logger.error(f"Error saving learning data: {e}")

def load_learning_data_extra():
    """
    Loads learning data from a JSON file in extra mode.
    """
    global learning_data
    try:
        with open("learning_data.json", "r", encoding="utf-8") as f:
            learning_data = json.load(f)
        logger.info("Learning data loaded.")
    except Exception:
        learning_data = {}
        logger.warning("No existing learning data found. Starting fresh.")

##############################
# VISUALIZER CLASS (Streamlit & Mayavi)
##############################
# Note: The class VisualParticleExtra is defined above and used in extra mode.

# Globals for extra mode
audio_queue_extra = queue.Queue()
log_queue_extra = queue.Queue()
particles_extra: List[VisualParticleExtra] = []
frequency_history = collections.deque(maxlen=30)
learning_data = {}
XAI_API_KEY = "your-api-key-here"  # Placeholder for any future AI integrations

##############################
# STREAMLIT INTERFACE
##############################
def run_streamlit_ui():
    """
    Runs the Streamlit user interface for the simulation.
    """
    st.set_page_config(page_title="Cosmic Synapse Theory Simulation", layout="wide")
    st.title("Cosmic Synapse Theory (Madsen's Theory) Simulation")
    st.sidebar.header("Simulation Controls")
    
    # Simulation Parameters
    num_particles = st.sidebar.slider("Number of Particles", 100, 1000, 100, step=100)
    steps = st.sidebar.slider("Number of Steps", 100, 5000, 1000, step=100)
    dt = st.sidebar.slider("Time Step (s)", 0.1, 10.0, 1.0, step=0.1)
    E_replicate_slider = st.sidebar.slider("Replication Energy Threshold (J)", 1e40, 1e60, E_replicate, step=1e40, format="%.0e")
    alpha = st.sidebar.slider("Alpha (J/m)", 1e-12, 1e-8, alpha_initial, step=1e-10, format="%.1e")
    lambda_evo = st.sidebar.slider("Lambda Evolution (J)", 0.1, 10.0, lambda_evo_initial, step=0.1)
    data_directory = st.sidebar.text_input("Data Directory Path", DATA_DIRECTORY)
    audio_duration = st.sidebar.slider("Audio Capture Duration (s)", 0.5, 5.0, 1.0, step=0.5)
    sample_rate = st.sidebar.slider("Audio Sample Rate (Hz)", 8000, 48000, 44100, step=1000)
    
    global MAX_PARTICLES
    MAX_PARTICLES = num_particles * 10  # Adjust maximum particles based on user input
    
    # Initialize Simulator and Neural Network
    global SimulatorInstance
    if "simulator" not in st.session_state:
        SimulatorInstance = Simulator(num_particles=num_particles, steps=steps, dt=dt, data_directory=data_directory)
        st.session_state.simulator = SimulatorInstance
        st.session_state.neural_net = ParticleNeuralNet()
        st.session_state.neural_net.eval()
        for param in st.session_state.neural_net.parameters():
            nn.init.uniform_(param, -0.1, 0.1)
        st.session_state.running = False
        st.session_state.step = 0
        st.session_state.audio_running = False
        logger.info("Simulation and neural network initialized.")
    else:
        SimulatorInstance = st.session_state.simulator
    
    neural_net = st.session_state.neural_net
    
    # Update Simulation Parameters based on user input
    SimulatorInstance.replication.E_replicate = E_replicate_slider
    SimulatorInstance.dynamics.alpha = alpha
    SimulatorInstance.dynamics.lambda_evo_initial = lambda_evo  # Update initial lambda
    SimulatorInstance.data_directory = data_directory
    
    # Control Buttons
    start_button = st.sidebar.button("Start Simulation")
    pause_button = st.sidebar.button("Pause Simulation")
    reset_button = st.sidebar.button("Reset Simulation")
    run_full_button = st.sidebar.button("Run Full Simulation")
    load_data_button = st.sidebar.button("Load Simulation Data")
    visualize_metrics_button = st.sidebar.button("Visualize Metrics")
    animate_simulation_button = st.sidebar.button("Animate Simulation")
    volumetric_heatmap_button = st.sidebar.button("Volumetric Heatmap")
    start_audio_button = st.sidebar.button("Start Audio Capture")
    stop_audio_button = st.sidebar.button("Stop Audio Capture")
    
    # Handle Control Buttons
    if start_button:
        st.session_state.running = True
        st.sidebar.write("Simulation Started.")
    if pause_button:
        st.session_state.running = False
        st.sidebar.write("Simulation Paused.")
    if reset_button:
        simulator = Simulator(num_particles=num_particles, steps=steps, dt=dt, data_directory=data_directory)
        st.session_state.simulator = simulator
        SimulatorInstance = simulator
        st.session_state.neural_net = ParticleNeuralNet()
        st.session_state.neural_net.eval()
        for param in st.session_state.neural_net.parameters():
            nn.init.uniform_(param, -0.1, 0.1)
        st.session_state.running = False
        st.session_state.step = 0
        st.sidebar.write("Simulation Reset.")
        logger.info("Simulation reset.")
    if start_audio_button:
        if not SimulatorInstance.audio_running:
            SimulatorInstance.start_audio_capture(duration=audio_duration, sample_rate=sample_rate)
            st.sidebar.write("Audio Capture Started.")
            st.session_state.audio_running = True
        else:
            st.sidebar.write("Audio Capture is already running.")
    if stop_audio_button:
        if SimulatorInstance.audio_running:
            SimulatorInstance.stop_audio_capture()
            st.sidebar.write("Audio Capture Stopped.")
            st.session_state.audio_running = False
        else:
            st.sidebar.write("Audio Capture is not running.")
    
    # Run Simulation Step
    if st.session_state.running and st.session_state.step < SimulatorInstance.steps:
        frequency_data = SimulatorInstance.get_latest_frequency_data()
        camera_data = SimulatorInstance.get_latest_camera_data()
        SimulatorInstance.run_step(neural_net, frequency_data, camera_data)
        st.session_state.step += 1
        st.write(f"**Simulation Step:** {st.session_state.step}/{SimulatorInstance.steps}")
        st.write(f"**Number of Particles:** {len(SimulatorInstance.particles)}")
        SimulatorInstance.visualizer.plot_particles(st.session_state.step, frequency_data=frequency_data)
        st.subheader("Real-Time Metrics")
        if SimulatorInstance.total_energy_history:
            st.write(f"**Total Energy:** {SimulatorInstance.total_energy_history[-1]:.2e} J")
        if SimulatorInstance.kinetic_energy_history:
            st.write(f"**Kinetic Energy:** {SimulatorInstance.kinetic_energy_history[-1]:.2e} J")
        if SimulatorInstance.potential_energy_history:
            st.write(f"**Potential Energy:** {SimulatorInstance.potential_energy_history[-1]:.2e} J")
        if SimulatorInstance.entropy_history:
            st.write(f"**Average Entropy:** {SimulatorInstance.entropy_history[-1]:.2e} J")
    
    # Run Full Simulation
    if run_full_button:
        with st.spinner("Running full simulation..."):
            for _ in range(steps):
                frequency_data = SimulatorInstance.get_latest_frequency_data()
                camera_data = SimulatorInstance.get_latest_camera_data()
                SimulatorInstance.run_step(neural_net, frequency_data, camera_data)
                st.session_state.step += 1
            st.success("Simulation completed.")
            SimulatorInstance.visualizer.plot_metrics()
            SimulatorInstance.visualizer.animate_simulation()
    
    # Load Simulation Data
    if load_data_button:
        filename = st.sidebar.text_input("Enter filename to load", "simulation_data.pkl")
        if st.sidebar.button("Load"):
            if os.path.exists(filename):
                data = load_simulation_data(filename)
                SimulatorInstance.history = data.get('history', [])
                SimulatorInstance.total_energy_history = data.get('total_energy_history', [])
                SimulatorInstance.kinetic_energy_history = data.get('kinetic_energy_history', [])
                SimulatorInstance.potential_energy_history = data.get('potential_energy_history', [])
                SimulatorInstance.entropy_history = data.get('entropy_history', [])
                st.success("Simulation data loaded.")
                SimulatorInstance.visualizer.plot_metrics()
                SimulatorInstance.visualizer.animate_simulation()
            else:
                st.error("File not found.")
    
    # Visualize Metrics
    if visualize_metrics_button:
        SimulatorInstance.visualizer.plot_metrics()
    
    # Animate Simulation
    if animate_simulation_button:
        SimulatorInstance.visualizer.animate_simulation()
    
    # Plot Volumetric Heatmap
    if volumetric_heatmap_button:
        step_to_plot = st.number_input("Enter step number for volumetric heatmap", min_value=1, max_value=SimulatorInstance.steps, value=st.session_state.step)
        if st.button("Plot Volumetric Heatmap"):
            SimulatorInstance.visualizer.plot_volumetric_heatmap(step_to_plot)

##############################
# EXTRA MODE: Tkinter/Pygame/Speech/TTS
##############################
def run_extra_mode():
    """
    Runs the standalone mode with Tkinter and Pygame for enhanced visualization and interaction.
    """
    load_learning_data_extra()
    tk_log = create_log_window(log_queue_extra)
    # Initialize SimulatorInstance for extra mode
    global SimulatorInstance
    SimulatorInstance = Simulator()
    threading.Thread(target=visualization_loop, args=(SimulatorInstance,), daemon=True).start()
    # Optionally start camera capture if available
    if cv2 is not None:
        SimulatorInstance.start_camera_capture(device=0)
    # Start speech recognition in a separate thread
    threading.Thread(target=speech_recognition_thread, args=(SimulatorInstance,), daemon=True).start()
    tk_log.mainloop()

def speech_recognition_thread(simulator):
    """
    Continuously listens for speech and processes it.
    """
    while True:
        text = recognize_speech_extra()
        if text:
            logger.info(f"Recognized and processed speech: {text}")

##############################
# MAIN EXECUTION
##############################
def main_entry():
    """
    Entry point of the program. Determines the mode to run based on command-line arguments.
    """
    if len(sys.argv) > 1 and sys.argv[1] == "tk":
        run_extra_mode()
    else:
        run_streamlit_ui()

if __name__ == "__main__":
    main_entry()
