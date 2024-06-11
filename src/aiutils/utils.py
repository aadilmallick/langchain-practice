import os 
import sys
import json

class JSONer:
    @staticmethod
    def stringify(data: dict):
        return json.dumps(data)
    
    @staticmethod
    def parse(json_data: str):
        return json.loads(json_data)
    
    def __init__(self, filepath: str, data: dict):
        pass

class OSUtils:
    @staticmethod
    def run_unix_command(command: str):
        try: 
            os.system(command)
        except Exception as e:
            print(f"Are you running in the correct environment? Error: {e}")         

    
    @staticmethod
    def find_how_many_files_are_in_a_folder(folder_path: str):
        return len(os.listdir(folder_path))
    
    @staticmethod
    def read_file(file_path: str):
        with open(file_path, 'r') as file:
            return file.read()
    
    @staticmethod
    def isWindows():
        return sys.platform.startswith('win')
    
    @staticmethod
    def isLinux():
        return sys.platform.startswith('linux')