import subprocess
import os
import platform
import sys
import shutil

def setup_new_dir_and_return_dirname():
    number = int(util.sglob('results/*')[-1][-4:])
    saveDir = 'results/trial{:04d}/'.format(number+1)
    util.safe_makedirs(saveDir)
    return saveDir

def main(script, direct):
    print(script, direct)
    os.makedirs(direct)
    filesToMove = ["unet.py", script, "warping.py"]
    for f in filesToMove:
        shutil.copy(f, direct)

    if platform.uname()[1].startswith('myers-mac-10'):
      print("Normal Sub.")
      job = r'python3 {} {}'.format(script, direct)
    elif platform.uname()[1].startswith('falcon1'):
      print("On Furiosa. Trying SLURM.")
      job = "srun -J {2} -n 1 -p gpu --time=48:00:00 -e {1}/stderr -o {1}/stdout time python3 {0} {1} &".format(script, direct, os.path.basename(direct)[-8:])
    elif platform.uname()[1].startswith('falcon'):
      print("On Madmax. Trying bsub.")
      job = "bsub -J {2} -n 1 -q gpu -W 48:00 -e {1}/stderr -o {1}/stdout time python3 {0} {1} &".format(script, direct, os.path.basename(direct)[-8:])
    else:
      print("ERROR: Couldn't detect platform!")
      sys.exit(1)

    subprocess.call(job, shell=True)

if __name__ == '__main__':
  print("System Args: ", sys.argv)
  script = sys.argv[1]
  direct = sys.argv[2]
  main(script, direct)
