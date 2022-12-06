import os
import pandas as pd


class Tokenize:
    def __init__(self, dir_file, file_name, out_dir, file_name_out) -> None:
        self.dir_file = dir_file
        self.file_name = file_name
        self.out_dir = out_dir
        self.file_name_out = file_name_out

    def splitText(self):
        EventTemplateIdentSplit = []
        EventTemplateSplit = []
        csv_file = pd.read_csv(os.path.join(self.dir_file, self.file_name))
        for index, line in csv_file.iterrows():
            EventTemplateIdentSplit.append(line['EventTemplateIdent'].split())
            EventTemplateSplit.append(line['EventTemplate'].split())

        csv_file['EventTemplateIdent'] = EventTemplateIdentSplit
        csv_file['EventTemplate'] = EventTemplateSplit

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        csv_file.to_csv(os.path.join(self.out_dir, self.file_name_out+"_tokenize.csv"), index=False)
