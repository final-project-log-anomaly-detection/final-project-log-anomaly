"""
Description : This file implements the Drain algorithm for log parsing
Author      : LogPAI team
License     : MIT
"""
import re
import os
import numpy as np
import pandas as pd
import hashlib
from datetime import datetime
import logging
from tqdm import tqdm

_logger = logging.getLogger(__name__)


class Logcluster:
    def __init__(self, logTemplate="", logTemplateIdent="", logIDL=None):
        self.logTemplate = logTemplate
        self.logTemplateIdent = logTemplateIdent
        # print(self.logTemplate)
        if logIDL is None:
            logIDL = []
        self.logIDL = logIDL


class Node:
    def __init__(self, childD=None, depth=0, digitOrtoken=None):
        if childD is None:
            childD = dict()
        self.childD = childD
        self.depth = depth
        self.digitOrtoken = digitOrtoken


class LogParser:
    def __init__(
        self, log_format, indir="./", outdir="./result/", depth=4, st=0.4, maxChild=100, rex=[], keep_para=True
    ):
        """
        Attributes
        ----------
            rex : regular expressions used in preprocessing (step1)
            path : the input path stores the input log file name
            depth : depth of all leaf nodes
            st : similarity threshold
            maxChild : max number of children of an internal node
            logName : the name of the input file containing raw log messages
            savePath : the output path stores the file containing structured logs
        """
        self.path = indir
        self.depth = depth - 2
        self.st = st
        self.maxChild = maxChild
        self.logName = None
        self.savePath = outdir
        self.df_log = None
        self.log_format = log_format
        self.rex = rex
        self.keep_para = keep_para

    def hasNumbers(self, s):
        return any(char.isdigit() for char in s)

    def treeSearch(self, rn, seq):  # รอบแรกที่เข้ามา return null seq = LogmessageL
        retLogClust = None

        seqLen = len(seq)
        if seqLen not in rn.childD:
            return retLogClust

        parentn = rn.childD[seqLen]

        currentDepth = 1
        for token in seq:
            if currentDepth >= self.depth or currentDepth > seqLen:
                break

            if token in parentn.childD:
                parentn = parentn.childD[token]
            elif "<*>" in parentn.childD:
                parentn = parentn.childD["<*>"]
            else:
                return retLogClust
            currentDepth += 1

        logClustL = parentn.childD

        retLogClust = self.fastMatch(logClustL, seq)

        return retLogClust

    def addSeqToPrefixTree(self, rn, logClust):  # หา pattern ของจำนวนการแบ่งคำที่ตรงกันแล้วทำเป็น tree
        seqLen = len(logClust.logTemplate)
        if seqLen not in rn.childD:
            firtLayerNode = Node(depth=1, digitOrtoken=seqLen)
            rn.childD[seqLen] = firtLayerNode
        else:
            firtLayerNode = rn.childD[seqLen]

        parentn = firtLayerNode

        currentDepth = 1
        for token in logClust.logTemplate:

            # Add current log cluster to the leaf node
            # print(self.depth) # มันเท่ากับ 2 ตอนตัวอย่างในตอนเริ่มต้นขึ้นอยู่กับตอนสร้าง class log passer
            if currentDepth >= self.depth or currentDepth > seqLen:
                if len(parentn.childD) == 0:
                    parentn.childD = [logClust]
                else:
                    parentn.childD.append(logClust)
                break

            # If token not matched in this layer of existing tree.
            if token not in parentn.childD:

                if not self.hasNumbers(token):
                    if "<*>" in parentn.childD:
                        if len(parentn.childD) < self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                        else:
                            parentn = parentn.childD["<*>"]
                    else:
                        if len(parentn.childD) + 1 < self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                        elif len(parentn.childD) + 1 == self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken="<*>")
                            parentn.childD["<*>"] = newNode
                            parentn = newNode
                        else:
                            parentn = parentn.childD["<*>"]

                else:
                    if "<*>" not in parentn.childD:
                        newNode = Node(depth=currentDepth + 1, digitOrtoken="<*>")
                        parentn.childD["<*>"] = newNode
                        parentn = newNode
                    else:
                        parentn = parentn.childD["<*>"]

            # If the token is matched
            else:
                parentn = parentn.childD[token]

            currentDepth += 1

    # seq1 is template
    def seqDist(self, seq1, seq2):
        assert len(seq1) == len(seq2)
        simTokens = 0
        numOfPar = 0

        for token1, token2 in zip(seq1, seq2):
            if token1 == "<*>":
                numOfPar += 1
                continue
            if token1 == token2:
                simTokens += 1

        retVal = float(simTokens) / len(seq1)

        return retVal, numOfPar

    def fastMatch(self, logClustL, seq):
        retLogClust = None

        maxSim = -1
        maxNumOfPara = -1
        maxClust = None

        for logClust in logClustL:
            # print(logClust)
            curSim, curNumOfPara = self.seqDist(logClust.logTemplateIdent, seq)
            if curSim > maxSim or (curSim == maxSim and curNumOfPara > maxNumOfPara):
                maxSim = curSim
                maxNumOfPara = curNumOfPara
                maxClust = logClust

        if maxSim >= self.st:
            retLogClust = maxClust

        return retLogClust

    def getTemplate(self, seq1, seq1_prev, ident, ident_prev):
        assert len(ident) == len(ident_prev)
        retVal = []
        retVal_sys_tag = []
        i = 0
        for word in seq1:
            if word == seq1_prev[i]:
                retVal.append(word)
            else:
                retVal.append("<*>")

            i += 1
        i = 0
        for word in ident:
            if word == ident_prev[i]:
                retVal_sys_tag.append(word)
            else:
                retVal_sys_tag.append("<similar_prev_pattern>")

            i += 1

        return retVal, retVal_sys_tag

    def outputResult(self, logClustL):  # logClustL คือ list ของ Logcluster
        log_templates = [0] * self.df_log.shape[0]
        log_templateids = [0] * self.df_log.shape[0]

        df_events = []
        event_ident = 1
        for logClust in logClustL:
            template_str = " ".join(logClust.logTemplate)  # ต่อ template จากที่แยกเป็น list กลับเป็น string
            occurrence = len(logClust.logIDL)
            # template_id = 'E' + str(event_ident)  # เข้ารหัสเพื่อใช้เป็น event_ID
            template_id = hashlib.md5(template_str.encode("utf-8")).hexdigest()[0:8]
            for logID in logClust.logIDL:
                logID -= 1
                log_templates[logID] = template_str
                log_templateids[logID] = template_id
            # df_events.append([template_id, template_str, templateIdent_str, occurrence])  # data for log_templates.csv
            df_events.append([template_id, template_str, occurrence])  # data for log_templates.csv
            event_ident += 1

        df_event = pd.DataFrame(df_events, columns=["EventId", "EventTemplate", "Occurrences"])
        # convert list of event to dataFrame

        self.df_log["EventId"] = log_templateids
        self.df_log["EventTemplate"] = log_templates

        # add 2 more column for EventId and EventTemplate
        if self.keep_para:
            _logger.warning(self.get_parameter_list)

            self.df_log["ParameterList"] = self.df_log.apply(self.get_parameter_list, axis=1)

        self.df_log.to_csv(os.path.join(self.savePath, self.logName + "_structured.csv"), index=False)

        occ_dict = dict(self.df_log["EventTemplate"].value_counts())
        df_event = pd.DataFrame()

        df_event["EventTemplate"] = self.df_log["EventTemplate"].unique()
        df_event["EventId"] = df_event["EventTemplate"].map(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()[0:8])
        df_event["Occurrences"] = df_event["EventTemplate"].map(occ_dict)
        df_event.to_csv(
            os.path.join(self.savePath, self.logName + "_templates.csv"),
            index=False,
            columns=["EventId", "EventTemplate", "Occurrences"],
        )

    def outputResultIdent(self, logClustL):
        log_templates = [0] * self.df_log.shape[0]
        log_templateids = [0] * self.df_log.shape[0]
        log_templateIdents = [0] * self.df_log.shape[0]

        df_events = []
        event_ident = 1
        for logClust in logClustL:
            template_str = " ".join(logClust.logTemplate)  # ต่อ template จากที่แยกเป็น list กลับเป็น string
            templateIdent_str = " ".join(logClust.logTemplateIdent)
            occurrence = len(logClust.logIDL)
            template_id = hashlib.md5(templateIdent_str.encode("utf-8")).hexdigest()[0:8]
            for logID in logClust.logIDL:
                logID -= 1
                log_templates[logID] = template_str
                log_templateids[logID] = template_id
                log_templateIdents[logID] = templateIdent_str
            df_events.append([template_id, templateIdent_str, occurrence])  # data for log_templates.csv
            event_ident += 1

        df_event = pd.DataFrame(df_events, columns=["EventId", "EventTemplateIdent", "Occurrences"])

        self.df_log["EventId"] = log_templateids
        self.df_log["EventTemplate"] = log_templates
        self.df_log["EventTemplateIdent"] = log_templateIdents
        if self.keep_para:
            _logger.warning(self.get_parameter_list)

            self.df_log["ParameterList"] = self.df_log.apply(self.get_parameter_list, axis=1)

        self.df_log.drop(["EventTemplate"], axis=1)
        self.df_log.to_csv(os.path.join(self.savePath, self.logName + "_ident_structured.csv"), index=False)

        occ_dict = dict(self.df_log["EventTemplateIdent"].value_counts())
        df_event = pd.DataFrame()

        df_event["EventTemplateIdent"] = self.df_log["EventTemplateIdent"].unique()
        df_event["EventId"] = df_event["EventTemplateIdent"].map(
            lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()[0:8]
        )
        df_event["Occurrences"] = df_event["EventTemplateIdent"].map(occ_dict)
        df_event.to_csv(
            os.path.join(self.savePath, self.logName + "_ident_templates.csv"),
            index=False,
            columns=["EventId", "EventTemplateIdent", "Occurrences"],
        )

    def printTree(self, node, dep):
        pStr = ""
        for i in range(dep):
            pStr += "\t"

        if node.depth == 0:
            pStr += "Root"
        elif node.depth == 1:
            pStr += "<" + str(node.digitOrtoken) + ">"
        else:
            pStr += node.digitOrtoken

        print(pStr)

        if node.depth == self.depth:
            return 1
        for child in node.childD:
            self.printTree(node.childD[child], dep + 1)

    def parse(self, logName):
        print("Parsing file: " + os.path.join(self.path, logName))
        start_time = datetime.now()
        self.logName = logName
        rootNode = Node()
        logCluL = []

        self.load_data()  # return self.df_log dataframe
        # print(self.df_log)

        for idx, line in tqdm(self.df_log.iterrows(), total=len(self.df_log), desc="Parsing log :"):
            logID = line["LineId"]
            logmessageL, logmessageIdent = self.preprocess(line["Content"])
            logmessageL = logmessageL.strip().split()  # fill <*> in content then split
            logmessageIdent = logmessageIdent.strip().split()
            # logmessageL = filter(lambda x: x != '', re.split('[\s=:,]', self.preprocess(line['Content'])))
            matchCluster = self.treeSearch(rootNode, logmessageIdent)

            # search หา logmessage ที่ match กับ logmessageL ที่ split ออกมา

            # Match no existing log cluster
            if matchCluster is None:
                # create Logcluster class give list logmessageL and logID to this class
                newCluster = Logcluster(logTemplate=logmessageL, logTemplateIdent=logmessageIdent, logIDL=[logID])
                logCluL.append(newCluster)
                self.addSeqToPrefixTree(rootNode, newCluster)

            # Add the new log message to the existing cluster
            else:
                newTemplate, newTemplateIdent = self.getTemplate(
                    logmessageL, matchCluster.logTemplate, logmessageIdent, matchCluster.logTemplateIdent
                )

                # getTemplate มันจะเช็ค logmessage ใหม่ที่แบ่งมาแล้วไปเทียบกับ template ที่เหมือนกัน
                # มีโอกาสที่ มันอาจจะเช็คกันแล้ว template มันไม่เหมือนกันก็จะยึดตามตัวใหม่ โดยเช็คอีกทีตอน if ด้านล่าง
                # ถ้า match กับ template ที่มีอยู่ก็จะ add logID เข้าไปเพื่อบอกว่าบรรทัดไหน
                matchCluster.logIDL.append(logID)
                # print(matchCluster.logIDL)
                if " ".join(newTemplateIdent) != " ".join(matchCluster.logTemplateIdent):
                    matchCluster.logTemplate = newTemplate
                    matchCluster.logTemplateIdent = newTemplateIdent

        # direction to save file
        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)
        self.outputResultIdent(logCluL)

        print("Parsing done. [Time taken: {!s}]".format(datetime.now() - start_time))

    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        # print(self.log_format)  # <Date> <Time> <Pid> <Level> <Component>: <Content>
        # print(headers)  # ['Date', 'Time', 'Pid', 'Level', 'Component', 'Content']
        # print(regex)  # re.compile('^(?P<Date>.*?)\\s+(?P<Time>.*?)\\s+(?P<Pid>.*?)\\s+(?P<Level>.*?)\\s+(?P<Component>.*?):\\s+(?P<Content>.*?)$')
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logName), regex, headers, self.log_format)
        # print(self.df_log)

    def preprocess(self, line):  # ใส่ <*> ให้กับ ข้อมูลที่เป็น content
        line2 = line
        for currentRex in self.rex:
            line = re.sub(currentRex["regex"], "<*>", line)
            line2 = re.sub(currentRex["regex"], "<" + currentRex["name"] + ">", line2)

        return line, line2

    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """Function to transform log file to dataframe"""
        log_messages = []
        linecount = 0
        # print(regex)
        with open(log_file, "r") as fin:
            for line in fin.readlines():
                try:
                    match = regex.search(line.strip())  # ตัดอักขระพิเศษหลังข้อความออกเช่น \n
                    # group ข้อมูลออกมาแล้วใส่เป็น list [081109,203615,148,INFO,dfs.DataNode$PacketResponder,PacketResponder 1 for block blk_38865049064139660 terminating]
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    pass
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, "LineId", None)
        logdf["LineId"] = [i + 1 for i in range(linecount)]
        return logdf

    def generate_logformat_regex(self, logformat):
        """Function to generate regular expression to split log messages"""

        headers = []
        splitters = re.split(r"(<[^<>]+>)", logformat)
        # ['', '<Date>', ' ', '<Time>', ' ', '<Pid>', ' ', '<Level>', ' ', '<Component>', ': ', '<Content>', '']
        # print(splitters)
        regex = ""

        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(" +", "\\\s+", splitters[k])
                regex += splitter

            else:
                header = splitters[k].strip("<").strip(">")
                regex += "(?P<%s>.*?)" % header  # such as (?P<Date>.*?)
                headers.append(header)

        regex = re.compile("^" + regex + "$")
        return headers, regex

    def get_parameter_list(self, row):  # list parameter ออกมาจาก <*> แล้วเก็บใน column ParameterList
        # template_regex = re.sub(r"<.{1,5}>", "<*>", row["EventTemplate"])
        template_regex = row["EventTemplate"]
        if "<*>" not in template_regex:
            return []

        template_regex = re.sub(r"([^A-Za-z0-9])", r"\\\1", template_regex)
        template_regex = re.sub(r"\\ +", r"\s+", template_regex)
        template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"

        parameter_list = re.findall(template_regex, row["Content"])
        # print(parameter_list)
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
        return parameter_list
