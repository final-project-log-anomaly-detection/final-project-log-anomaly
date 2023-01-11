import Drain
import sys
sys.path.append('../')


input_dir = 'ThunderBird/'
output_dir = 'Drain_result/'  # The output directory of parsing results


log_file = 'thunderbird_10M.log'

log_format = '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>'

regex = [
    {'regex': r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', 'name': 'IP'},  # IP
    {'regex': r'\\_SB_.+_PRT', 'name': 'RT'},  # routing table
    {'regex': r'(\/[\w.-]+)+(\.\w+)?', 'name': 'Path'},  # path
    {'regex': r'0[xX][0-9a-fA-F]+', 'name': 'Hexadecimal'},
    {'regex': r'[0-9]+', 'name': 'Numbers'},  # Numbers
]

st = 0.5  # Similarity threshold
depth = 4  # Depth of all leaf nodes

parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir,  depth=depth, st=st, rex=regex)
parser.parse(log_file)
