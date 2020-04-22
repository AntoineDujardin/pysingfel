"""
This is an MPI script aiming at generating SPI images.

The current target is a single node on a hybrid system with 1 GPU.
Rank 0 coordinates.
Rank 1 uses the GPU.
Ranks 2+ use the CPU.
"""

from mpi4py import MPI
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.size

MASTER_RANK = 0
GPU_RANKS = (1,)

import os
# Only rank 1 uses the GPU/cupy.
os.environ["USE_CUPY"] = '1' if RANK in GPU_RANKS else '0'
# Unlock parallel but non-MPI HDF5
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

import sys
ROOT_DIR = "../.."
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
sys.path.append(ROOT_DIR+"/../lcls2/psana")

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import h5py as h5
import time
import datetime
from matplotlib.colors import LogNorm
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

import pysingfel as ps
import pysingfel.gpu as pg
from pysingfel.util import asnumpy, xp


if RANK == 0:
    assert SIZE > 1, "This script is planed for at least 2 ranks."

beam = ps.Beam(ROOT_DIR+'/examples/input/beam/amo86615.beam')
#beam.set_photons_per_pulse(beam.get_photons_per_pulse()*100)

# Load and initialize the detector
det = ps.PnccdDetector(geom=ROOT_DIR+'/examples/lcls/amo86615/PNCCD::CalibV1/Camp.0:pnCCD.1/geometry/0-end.data',
                       beam=beam)

N_images_per_batch = 20
N_batches = 500
N_images_tot = N_images_per_batch * N_batches
FNAME = "/reg/neh/home/dujardin/scratch/2CEX-new.h5"

if RANK == MASTER_RANK:
    f = h5.File(FNAME, "w")
    f.create_dataset("pixel_position_reciprocal",
                     data=det.pixel_position_reciprocal)
    f.create_dataset("slices", (N_images_tot, 4, 512, 512), np.float)
    f.close()

# Create a particle object
if RANK == GPU_RANKS[0]:
    particle = ps.Particle()
    particle.read_pdb(ROOT_DIR+'/examples/input/pdb/2CEX.pdb', ff='WK')
    experiment = ps.SPIExperiment(det, beam, particle)
else:
    experiment = ps.SPIExperiment(det, beam, None)
buffer = asnumpy(experiment.volumes[0])

sys.stdout.flush()

# Exchange
COMM.Bcast(buffer, root=1)

if RANK in GPU_RANKS[1:]:
    experiment.volumes[0] = xp.asarray(experiment.volumes[0])

N_images_processed = 0
start = time.perf_counter()

if RANK == MASTER_RANK:
    for batch_n in tqdm(range(N_batches)):
        # Send batch numbers to ranks
        i_rank = COMM.recv(source=MPI.ANY_SOURCE)
        COMM.send(batch_n, dest=i_rank)
    for _ in range(SIZE-1):
        # Send one "None" to each rank as final flag
        i_rank = COMM.recv(source=MPI.ANY_SOURCE)
        COMM.send(None, dest=i_rank)
else:
    f = h5.File(FNAME, "r+")
    h5_slices = f["slices"]

    while True:
        # Ask for more data
        COMM.send(RANK, dest=MASTER_RANK)
        batch_n = COMM.recv(source=MASTER_RANK)
        if batch_n is None:
            break
        for i in range(N_images_per_batch):
            idx = batch_n*N_images_per_batch + i
            img = experiment.generate_image_stack()
            h5_slices[idx] = asnumpy(img)
            N_images_processed += 1

    f.close()

stop = time.perf_counter()

sys.stdout.flush()
COMM.barrier()
if RANK in GPU_RANKS:
    print(f"GPU rank {RANK} generated {N_images_processed} images in "
          f"{datetime.timedelta(seconds=round(stop-start))}s.")
elif RANK > 1:
    print(f"CPU rank {RANK} generated {N_images_processed} images in "
          f"{datetime.timedelta(seconds=round(stop-start))}s.")

sys.stdout.flush()
COMM.barrier()
if RANK == MASTER_RANK:
    print(f"Total script generated {N_images_tot} images in "
          f"{datetime.timedelta(seconds=round(stop-start))}s.")
