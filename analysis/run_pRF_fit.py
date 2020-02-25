import re
import os
import glob
import json
import sys

with open('analysis_params.json', 'r') as json_file:
    analysis_params = json.load(json_file)

subjects = ['12']#['2','11','12','13']#['1','2','3','4','5','8','9','11','12','13'] #['11']#['7']

total_chunks = analysis_params['total_chunks']


batch_string = """#!/bin/bash
#SBATCH -t 100:00:00
#SBATCH -N 1
#SBATCH -v
#SBATCH -c 24

# call the programs
echo "Job $SLURM_JOBID started at `date`" | mail $USER -s "Job $SLURM_JOBID"

conda activate i36

python /home/inesv/SB-ref/scripts/pRF_fitmodel.py $SJ_NR cartesius $CHUNK_NR

wait          # wait until programs are finished

echo "Job $SLURM_JOBID finished at `date`" | mail $USER -s "Job $SLURM_JOBID"
"""

basedir = '/home/inesv/batch/'

os.chdir(basedir)


for subject in subjects:

    for _,chu in enumerate(range(total_chunks)): # submit job for each chunk

        working_string = batch_string.replace('$SJ_NR', str(subject).zfill(2))
        working_string = working_string.replace('$CHUNK_NR', str(chu+1).zfill(3))

        js_name = os.path.join(basedir, 'pRF_' + str(subject).zfill(2) + '_chunk-%s_of_%s'%(str(chu+1).zfill(3),str(total_chunks).zfill(3)) + '_iterative.sh')
        of = open(js_name, 'w')
        of.write(working_string)
        of.close()

        print('submitting ' + js_name + ' to queue')
        print(working_string)
        os.system('sbatch ' + js_name)

