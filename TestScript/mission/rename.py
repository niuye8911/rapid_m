import os 
  
# Function to rename multiple files 
def main(): 
    for filename in os.listdir("./"): 
        if 'P_M' in filename:
            dst = filename.replace('P_M','P_M_RUSH')
            os.rename(filename, dst)
        else:
            continue
  
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main() 
