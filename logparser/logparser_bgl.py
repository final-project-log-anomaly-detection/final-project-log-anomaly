import Drain
import sys
sys.path.append('../')

input_dir = 'BGL/'
output_dir = 'Drain_result/'

log_file = 'BGL.log'

log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'

regex = [
    {'regex': r'0[xX][0-9a-fA-F]+', 'name': 'Hexadecimal'},
    {'regex': r'core.+', 'name': 'Core'},  # core
    {'regex': r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', 'name': 'IP'},  # IP
    # {'regex': r'(\/.*?\/)((?:[^\/]|\\\/)+?)(?:(?<!\\):|$)', 'name': 'Path'},  # path
    {'regex': r'(\/[\w.-]+)+(\.\w+)?', 'name': 'Path'},  # path
    {'regex': r'[0-9]+', 'name': 'Numbers'},  # Numbers
]

st = 0.5  # Similarity threshold
depth = 4  # Depth of all leaf nodes

parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir,  depth=depth, st=st, rex=regex)
parser.parse(log_file)
