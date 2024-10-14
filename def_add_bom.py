from pandas import read_excel


file_path=input("Enter path to excel file:")

def excell_extraction(file_path):
   data = read_excel(file_path)
   return data

my_data=excell_extraction(file_path)
print(my_data)