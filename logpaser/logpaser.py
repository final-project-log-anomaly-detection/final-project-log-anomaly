#!/usr/bin/env python
import Drain
import sys
sys.path.append('../')

# The input directory of log file
# input_dir = '/Users/thanadonlamsan/Documents/research project จบ/log_anomaly_detection_code/Hadoop/application_1445062781478_0011/'
input_dir = 'BGL/'
output_dir = 'Drain_result/'  # The output directory of parsing results
# log_file = 'container_1445062781478_0011_01_000001.log'  # The input log file name
log_file = 'BGL_2.log'
# HDFS log format   Hadoop >>>>>  Date,Time,Level,Process,Component,Content,EventId,EventTemplate
# log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>' # HDFS
log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'  # BGL
# Regular expression list for optional preprocessing (default: [])
regex = [
    # {'regex': r'blk_(|-)[0-9]+', 'name': 'block_id'},  # block id
    # {'regex': r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', 'name': 'IP'},  # IP
    # {'regex': r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$', 'name': 'Numbers'},  # Numbers
    # {'regex': r'0x.{8},?', 'name': 'Memory_address'},  # memory address
    {'regex': r'0[xX][0-9a-fA-F]+', 'name': 'Hexadecimal'},
    {'regex': r'core.+', 'name': 'Core'},  # core
    {'regex': r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', 'name': 'IP'},  # IP
    # {'regex': r'(\/.*?\/)((?:[^\/]|\\\/)+?)(?:(?<!\\):|$)', 'name': 'Path'},  # path
    {'regex': r'(\/[\w-]+)+(\.\w+)?', 'name': 'Path'},  # path
    {'regex': r'[0-9]+', 'name': 'Numbers'},  # Numbers
]
st = 0.5  # Similarity threshold
depth = 4  # Depth of all leaf nodes

parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir,  depth=depth, st=st, rex=regex)
parser.parse(log_file)
