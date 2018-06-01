import pickle
import os
import pandas as pd

def pickle_it(obj, name_str):
    '''

    Create a pickle file in the current working directory
    and store a pickle file by the name '<name_str>.pickle'
    with the object serialized in it

    '''
    file_name = name_str + '.pickle'

    with open(file_name, 'wb') as pickle_out:
        pickle.dump(obj, pickle_out)
        pickle_out.close()
    

def read_pickle(name_str):
    '''

    Read the pickle file named '<name_str>.pickle' to
    deserialize the previously serialized object and return
    it

    '''
    file_name = name_str + '.pickle'
    obj  = None

    with open(file_name, 'rb') as pickle_in:
        obj = pickle.load(pickle_in)
        pickle_in.close()

    return obj


def main():
	os.chdir('C:\\Users\\praty\\Desktop\\python urllib')
	
	x = pd.DataFrame([[1, 2, 3], [4, 5, 6]])

	pickle_it(x, 'dataframe')

	y = read_pickle('dataframe')


	for i in range(0, 4):
   		print(i, type(y))
	print(y)
	print(type(y))

if __name__ == '__main__':
	main()
