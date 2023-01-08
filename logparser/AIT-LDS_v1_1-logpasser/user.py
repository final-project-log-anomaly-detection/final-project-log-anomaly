import sys
import os

if __name__ == '__main__':
    sys.path.append(os.path.abspath('logparser'))

import Drain

sys.path.append('../')

inputs_dir = ['AIT-LDS-v1_1/data/mail.cup.com',
              'AIT-LDS-v1_1/data/mail.insect.com',
              'AIT-LDS-v1_1/data/mail.onion.com',
              'AIT-LDS-v1_1/data/mail.spiral.com'
              ]
outputs_dir = 'Drain_result/AIT_user'

st = 0.5  # Similarity threshold
depth = 4  # Depth of all leaf nodes

log_file = 'user.log'

log_format = '<Month> <Day> <Time> <Type> <something_1>: <Content>'

regex = []

parser = Drain.LogParser(log_format, indir=inputs_dir[0], outdir=outputs_dir,  depth=depth, st=st, rex=regex)
parser.parse(log_file)
