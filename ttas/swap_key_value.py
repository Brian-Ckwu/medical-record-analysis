import json

def main():
    infile = "./ttas_name_to_code.json"
    with open(infile, mode="rt", encoding="utf-8") as f:
        name_to_code = json.load(f)
    
    code_to_name = {}
    for name, code in name_to_code.items():
        code_to_name[code] = name
    
    outfile = "./ttas_code_to_name.json"
    with open(outfile, mode="wt", encoding="utf-8") as f:
        json.dump(code_to_name, f)

if __name__ == "__main__":
    main()