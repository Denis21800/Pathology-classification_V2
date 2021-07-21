from config import PipelineConfig
from pymongo import MongoClient
import numpy as np


class DBManager(object):
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.dataset = {}
        self.dataset = {}

    def connect(self):
        pass

    def get_data(self, labels=None):
        pass

    def upload_data(self, data):
        pass


class MongoDBManager(DBManager):
    def __init__(self, config: PipelineConfig):
        super(MongoDBManager, self).__init__(config)
        self.db_name = self.config.mongo_db_name
        self.host = self.config.mongo_host
        self.port = self.config.mongo_port
        self.col_name = self.config.mongo_col_name
        self.client = None
        self.db = None
        self.data_col = None

    def connect(self):
        self.client = MongoClient(self.host, self.port)
        self.db = self.client[self.db_name]
        self.data_col = self.db[self.col_name]

    def upload_data(self, data):
        for key in data:
            key_rec = data.get(key)
            data_rec = key_rec.get('data')
            intensity_arr = data_rec.get('intensity')
            p_mass_arr = data_rec.get('pm')
            label = data_rec.get('label')
            is_test = data_rec.get('is_test')
            file = data_rec.get('file')
            if type(intensity_arr) == np.ndarray:
                intensity_arr = intensity_arr.tolist()
            if type(p_mass_arr) == np.ndarray:
                p_mass_arr = p_mass_arr.tolist()

            self.data_col.insert_one({'file': file,
                                      'label': label,
                                      'is_test': is_test,
                                      'intensity_data': intensity_arr,
                                      'p_mass_data': p_mass_arr})

    def get_data(self, labels=None):
        if labels:
            cursor = self.data_col.find({'label': {'$in': labels}})
        else:
            cursor = self.data_col.find()

        for index, rec in enumerate(cursor):
            file_ = rec.get('file')
            intensity_arr = rec.get('intensity_data')
            label_ = rec.get('label')
            is_test = rec.get('is_test')
            pm_arr = rec.get('p_mass_data')
            data_rec = {'label': label_,
                        'intensity': intensity_arr,
                        'pm': pm_arr,
                        'file': file_,
                        'is_test': is_test,
                        'metadata': None}
            self.dataset.update({index: {'data': data_rec}})
