import gzip
import shutil

def extract_and_peek(gz_filename, txt_filename):
    print(f"Extracting '{gz_filename}' into '{txt_filename}'...")
    
    try:
        # 1. Extract the compressed file to a standard text file
        with gzip.open(gz_filename, 'rt', encoding='utf-8') as f_in:
            with open(txt_filename, 'w', encoding='utf-8') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # 2. Print the first 10 lines so we can see the format
        print(f"\n--- First 5 lines of {txt_filename} ---")
        with open(txt_filename, 'r', encoding='utf-8') as f:
            for i in range(5):
                line = f.readline()
                if not line: break
                # Using repr() highlights spaces (' ') vs tabs ('\t')
                print(repr(line)) 
        print("------------------------------------------\n")
        
    except FileNotFoundError:
        print(f"ERROR: Could not find '{gz_filename}'. Check the spelling!\n")

if __name__ == '__main__':
    # File 1: The Graph
    extract_and_peek('actual graph.gz', 'actual_graph.txt')
    
    # File 2: The Communities
    extract_and_peek('top 5000 communities.gz', 'top_5000_communities.txt')