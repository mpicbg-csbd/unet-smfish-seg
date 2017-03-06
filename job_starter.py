from __future__ import print_function, division
import subprocess
# import pickle as pkl
import os
import platform
import sys
import numpy as np
import shutil


def setup_new_dir_and_return_dirname():
    number = int(util.sglob('results/*')[-1][-4:])
    saveDir = 'results/trial{:04d}/'.format(number+1)
    util.safe_makedirs(saveDir)
    # make directory
    return saveDir

def main(direct):
    home = os.getcwdu()
    # params = makeParams()
    # repeats = 1
    # N = params["N"]
    # D = params["D"]
    print(direct)

    # for i in range(repeats):
    #   # initial conditions
    #   positions = ic.pos_spacing2d(1, 1, 20, 20, N, D)
    #   positions = positions + (np.random.rand(N,D)-0.5)*0.2
    #   velocities = np.random.normal(scale=1.0/np.sqrt(2), size=(N,D))

    os.makedirs(direct)
    filesToMove = ["ipy.py", "models/unet.py", "main.py"]
    for f in filesToMove:
        shutil.copy(f, direct)
    # START JOBS FROM MAIN DIR
    # os.chdir(direct)

    # this is what we submit to qsub

    if platform.uname()[1].startswith('myers-mac-10'):
      print("Normal Sub.")
      job = r'python main.py {}'.format(direct)
      subprocess.call(job, shell=True)
    elif platform.uname()[1].startswith('falcon1'):
      print("On Furiosa. Trying SLURM.")
      subprocess.call("module load hdf5", shell=True)
      subprocess.call("module load gcc/5.3.0", shell=True)
      subprocess.call("module load cuda/8.0.44", shell=True)
      job = "srun -n 1 -c 4 -p gpu --time=24:00:00 --mem-per-cpu=4096 -e {0}/stderr -o {0}/stdout time python main.py {0}".format(direct)
      subprocess.call(job, shell=True)
      print("Running job:", job)
    elif platform.uname()[1].startswith('falcon'):
      print("On Madmax. Trying bsub. TODO...")
    else:
      print("ERROR: Couldn't detect platform!")

    # os.chdir(home)

# ---- Main entry point
if __name__ == '__main__':
  try:
    direct = sys.argv[1]
  except:
    print('You need to include the folder name silly.')
  main(direct)
