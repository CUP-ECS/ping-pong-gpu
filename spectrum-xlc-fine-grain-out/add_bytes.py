import pandas as pd
import sys, getopt

def csv_to_dataframe(csv_file):
        df = pd.read_csv(csv_file)
        df['dim'] = df['dim'].map(lambda x: (x ** 2) * 120)
        df.to_csv(csv_file)

def main():
        file1 = sys.argv[1]
        file2 = sys.argv[2]
        file3 = sys.argv[3]
        csv_to_dataframe(file1)
        csv_to_dataframe(file2)
        csv_to_dataframe(file3)
        
if __name__=="__main__":
        main()
