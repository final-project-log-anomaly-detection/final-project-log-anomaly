import sys
import os

if __name__ == '__main__':
    sys.path.append(os.path.abspath('logparser'))

import Drain

sys.path.append('../')

inputs_dir = ['AIT-LDS-v1_1/data/mail.cup.com/apache2',
              'AIT-LDS-v1_1/data/mail.insect.com/apache2',
              'AIT-LDS-v1_1/data/mail.onion.com/apache2',
              'AIT-LDS-v1_1/data/mail.spiral.com/apache2'
              ]
outputs_dir = 'Drain_result/apache2'

st = 0.5  # Similarity threshold
depth = 4  # Depth of all leaf nodes

log_file = ['mail.cup.com-access.log', 'mail.insect.com-access.log',
            'mail.onion.com-access.log', 'mail.spiral.com-access.log']

log_format = '<IP> - - \[<DateTime>\] "<Content>"'

regex = []

parser = Drain.LogParser(log_format, indir=inputs_dir[1], outdir=outputs_dir,  depth=depth, st=st, rex=regex)
parser.parse(log_file[1])
