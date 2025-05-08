import json
import pickle
import csv
from collections import defaultdict
import pandas as pd
def load_jsonl(file_name):
    data = []
    with open(file_name, "rb") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

class DataStore:
    def __init__(self,) -> None:
        self.length = 0
        
    def prepare_for_slurm(self):
        return self._prepare_for_slurm()


        

# QuaRel
class QuaRelDataStore(DataStore):
    def __init__(self,in_path) -> None:
        super().__init__()
        self.load(in_path)
        self.data = [self.data.iloc[i] for i in range(len(self.data))]
        self.knoweledge_dict = {}

    def _prepare_for_slurm(self):
        new_data = []
        for d in self.data:
            for w in d["world_literals"]:
                new_data.append({"entity":d["world_literals"][w], "question":d["question"]})
        return new_data
    def save(self,path):
        with open(path, "wb") as f:
            pickle.dump(self.data, f)
    def load(self,path):
        with open(path, "rb") as f:
            self.data = pickle.load(f)    

    def __getitem__(self, i):
        return self.data[i]
    def __len__(self):
        return len(self.data)
    def __iter__(self):
        self.i = 0
        self.max = len(self.data)
        return self
    def __next__(self):
        if self.i < self.max:
            result = self.data[self.i]
            self.i += 1
            return result
        else:
            raise StopIteration

    def _turn_df_2_dict(self,df):
        '''
        Turn generated knowledges pandas dataframe to a ditionary. More convinient for later use to combine with original data
        '''
        from collections import defaultdict
        dict = defaultdict(lambda : defaultdict(list))
        cur_entity = None
        cur_consr = None
        for i in range(len(df)):
            # print(df.iloc[i]["knowledge 0"])
            if type(df.iloc[i]["original query"]) == str:
                cur_entity = df.iloc[i]["original query"]
            if type(df.iloc[i]["positive constriants"]) == str:
                cur_consr = df.iloc[i]["positive constriants"]
            if type(df.iloc[i]["knowledge 0"]) == str:
                knowledge = df.iloc[i]["knowledge 0"]
                # knowledge = knowledge.split("\n")[1].strip()
                if "Context:" in knowledge:
                    knowledge = knowledge.split("Context:")[1].strip()
                dict[cur_entity][cur_consr].append(knowledge)
        self.knowledge_dict = dict
        return dict
    def add_generated_knowledge(self, df):
        self._turn_df_2_dict(df)
        for dp in self.data:
            world1 = dp["world_literals"]["world1"]; world2 = dp["world_literals"]["world2"]
            world1_knowledges = [x for xs in list(self.knowledge_dict[world1].values()) for x in xs] 
            world2_knowledges = [x for xs in list(self.knowledge_dict[world2].values()) for x in xs]
            dp["generated_knowledges1"] = world1_knowledges
            dp["generated_knowledges2"] = world2_knowledges
    

        
    def temp_add_generated_knowledge(self,df):
        '''
        for the wrong format, where the original query is the question rather than the entity
        '''
        from collections import defaultdict
        cur_idx = 0
        for i in range(len(df)):
            for j in range(len(self)):
                if df.iloc[i]["original query"] == self.data[j]["question"]:
                    cur_idx = j
                    break
            if "generated_knowledges1" not in self.data[cur_idx]:
                self.data[cur_idx]["generated_knowledges1"] = []
                self.data[cur_idx]["generated_knowledges2"] = []
            if type(df.iloc[i]["knowledge 0"]) == str:
                knowledge = df.iloc[i]["knowledge 0"].split("\n")[1].strip()
                if "Context:" in knowledge:
                    knowledge = knowledge.split("Context:")[1].strip()
                self.data[cur_idx]["generated_knowledges1"].append(knowledge)
            

# Com2Sense

def prepare_com2sense(filename,pair_file):
    data = json.load(open(filename))
    pair_data = json.load(open(pair_file))
    return data,pair_data

# Lions
class LionsDataStore(DataStore):
    def __init__(self,in_path,header,header_names=False) -> None:
        super().__init__()
        self.load(in_path,header,header_names)


    def _prepare_for_slurm(self):
        new_data = []
        for d in self.data:
            new_data.append([""] + d)
        return new_data
    def save(self,path):
        pandas.DataFrame(self.data).to_csv(path)
    def load(self,path,header=None, header_names = False):
        data = pd.read_csv(path,header=header)
        data_dict = defaultdict(lambda: defaultdict(dict))
        new_data = []
        for i in range(len(data)):
            if header_names:
                new_ans = data.iloc[i]["new_ans"]
            else:
                new_ans = data.iloc[i][3]
            if new_ans == -42:
                continue
            if header_names:
                obj1 = data.iloc[i]["obj1"]; obj2 = data.iloc[i]["obj2"]; dim = data.iloc[i]["dimension"]
            else:
                obj1 = data.iloc[i][0]; obj2 = data.iloc[i][1]; dim = data.iloc[i][2]
            data_dict[obj1][obj2][dim] = new_ans
            if len(new_data) >0:
                last = new_data[-1]
                if obj1 != last[0] or obj2 != last[1] or dim != last[2]:
                    new_data.append([obj1,obj2,dim,new_ans])
            else:
                new_data.append([obj1,obj2,dim,new_ans])
        self.data = new_data
        self.data_dict = data_dict

    def __getitem__(self, i):
        return self.data[i]
    def __len__(self):
        return len(self.data)
    def __iter__(self):
        self.i = 0
        self.max = len(self.data)
        return self
    def __next__(self):
        if self.i < self.max:
            result = self.data[self.i]
            self.i += 1
            return result
        else:
            raise StopIteration
    def _turn_df_2_dict(self,df):
        '''
        Turn generated knowledges pandas dataframe to a ditionary. More convinient for later use to combine with original data
        '''
        from collections import defaultdict
        dict = defaultdict(lambda : defaultdict(lambda : defaultdict(list)))
        cur_entity = None
        cur_2nd_entity = None
        cur_consr = None
        for i in range(len(df)):
            # print(df.iloc[i]["knowledge 0"])
            if type(df.iloc[i]["original query"]) == str:
                cur_entity = df.iloc[i]["original query"]
            if type(df.iloc[i]["2nd entity"]) == str:
                cur_2nd_entity = df.iloc[i]["2nd entity"]
            if type(df.iloc[i]["positive constriants"]) == str:
                cur_consr = df.iloc[i]["positive constriants"]
            if type(df.iloc[i]["knowledge 0"]) == str:
                knowledge = df.iloc[i]["knowledge 0"]
                # knowledge = knowledge.split("\n")[1].strip()
                if "Context:" in knowledge:
                    knowledge = knowledge.split("Context:")[1].strip()
                dict[cur_entity][cur_2nd_entity][cur_consr].append(knowledge)
        self.knowledge_dict = dict
        return dict
    def add_generated_knowledge(self, df):
        self._turn_df_2_dict(df)
        for dp in self.data:
            world1 = dp["world_literals"]["world1"]; world2 = dp["world_literals"]["world2"]
            world1_knowledges = [x for xs in list(self.knowledge_dict[world1].values()) for x in xs] 
            world2_knowledges = [x for xs in list(self.knowledge_dict[world2].values()) for x in xs]
            dp["generated_knowledges1"] = world1_knowledges
            dp["generated_knowledges2"] = world2_knowledges
