import csv

tree = {}
with open("board_list.csv", newline="") as csvfile:
    board_reader = csv.reader(csvfile, delimiter=',')
    next(board_reader)
    for row in board_reader:
        cls = row[0]
        board = row[1]

        clses = cls.split(">")
        N = len(clses)
        curr = tree
        for i, cls in enumerate(clses):
            if cls not in curr.keys():
                curr[cls] = {}
            curr = curr[cls]
            if(i+1 == N):
                if board not in curr.keys():
                    curr[board] = {}

def print_tree(t, lev):
    for k, v in t.items():
        print("\t"*lev, k)
        if v:
            print_tree(v, lev+1)

print_tree(tree, 0)
