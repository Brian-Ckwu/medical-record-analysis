import json

def main():
    # Read from raw icdcodes file
    infile = "./raw_allcodes.json"
    with open(infile, mode="rt", encoding="utf-8") as f:
        raw_icdcodes = json.load(f)
    
    # Process and store in new icdcodes dictionary
    icdcodes = {
        1: {},
        2: {},
        3: {},
        4: {},
        5: {}
    }

    # Test depth
    for code_group in raw_icdcodes:
        for depth in range(1, len(code_group) + 1):
            assert(code_group[depth - 1]["depth"] == depth) # check if the depth is compatible

            try:
                code = code_group[depth - 1]["code"]
                desc = code_group[depth - 1]["descr"]
            except KeyError:
                continue

            if (code):
                icdcodes[depth][code] = desc
    
    # Write to new file
    outfile = "./allcodes.json"
    with open(outfile, mode="wt", encoding="utf-8") as f:
        json.dump(icdcodes, f)

    return 0

if __name__ == "__main__":
    main()