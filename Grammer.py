import re
from typing import Tuple
from TokenStream import TokenStream


class LL1Grammer:
    def __init__(self, grammer_path):
        NoTerminals, Terminals = set(), set()
        rules, S = [], None
        with open(grammer_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                ts = re.findall(r"([a-zA-Z]+) *::=(.*)", line)
                rules.append((ts[0][0], ts[0][1].strip()))
        # test=[]
        for r in rules:
            if S is None:
                S = r[0]
            NoTerminals.add(r[0])
            # test.append(r[0])
        # self.test=test
        for r in rules:
            V = r[1].replace("|", "").split()
            for v in V:
                if v not in NoTerminals:
                    Terminals.add(v)
        G = {}
        for r in rules:
            if G.get(r[0], None) is None:
                G[r[0]] = []
            for g in r[1].split("|"):
                G[r[0]].append(g.strip().split())

        self.NoTerminals = NoTerminals
        self.Terminals = Terminals
        self.G = G
        self.S = S
        self.epsilon = "ε"
        self.first_set = None
        self.follow_set = None
        self.compute_first()
        self.compute_follow()
        self.compute_predict()
        self.build_LL1_map()

    def compute_first(self):
        if self.first_set is not None:
            return self.first_set
        change_detector = makechanging()
        first_set = {v: set() for v in self.NoTerminals | self.Terminals}
        for t in self.Terminals:
            first_set[t].add(t)
        while change_detector(first_set):
            for nt, rs in self.G.items():
                for r in rs:
                    flag = 0
                    for E in r:
                        if self.epsilon in first_set[E]:
                            first_set[nt] |= first_set[E] - set(self.epsilon)
                        else:
                            first_set[nt] |= first_set[E]
                            flag = 1
                            break
                    if flag == 0:
                        first_set[nt].add(self.epsilon)
        self.first_set = first_set
        return first_set

    def compute_follow(self):
        change_detector = makechanging()
        follow_set = {v: set() for v in self.NoTerminals | self.Terminals}
        follow_set[self.S].add("$")
        while change_detector(follow_set):
            for nt, rs in self.G.items():
                for r in rs:
                    reverse_r, flag = r[::-1], 0
                    follow_set[reverse_r[0]] |= follow_set[nt]
                    for idx in range(1, len(reverse_r)):
                        Fol = reverse_r[idx - 1]
                        Cur = reverse_r[idx]
                        if Fol in self.Terminals:
                            follow_set[Cur] |= self.first_set[Fol]
                            flag = 1
                        elif Fol in self.NoTerminals:
                            if flag == 0 and self.epsilon in self.first_set[Fol]:
                                follow_set[Cur] |= (
                                    self.first_set[Fol] - set(self.epsilon)
                                ) | follow_set[nt]
                            else:
                                flag = 1
                                follow_set[Cur] |= self.first_set[Fol] - set(
                                    self.epsilon
                                )
                        else:
                            assert False
        self.follow_set = follow_set
        return follow_set

    def compute_predict(self):
        predict_set = {v: [] for v in self.NoTerminals}
        for nt, rs in self.G.items():
            for r in rs:
                flag = False
                selected = set()
                for E in r:
                    if self.epsilon in self.first_set[E]:
                        selected |= self.first_set[E] - set(self.epsilon)
                    else:
                        flag |= True
                        selected |= self.first_set[E]
                        break
                if flag is False:
                    selected |= self.follow_set[nt]
                predict_set[nt].append(selected)
        self.predict_set = predict_set
        return predict_set

    def build_LL1_map(self):
        if not check_LL1(self.predict_set):
            raise ValueError("Wrong LL1 grammer")
        LL1_map = {}
        for nt, selected in self.predict_set.items():
            rules = self.G[nt]
            assert len(rules) == len(selected)

            for sel, r in zip(selected, rules):
                for s in sel:

                    assert LL1_map.get((nt, s), None) is None
                    LL1_map[(nt, s)] = r
        self.LL1_map = LL1_map
        return LL1_map

    def parse(self, token_stream: TokenStream):
        stack = ["$", self.S]
        token = token_stream.get_next_token()
        root = AST(self.S)
        AST_stack = [root]

        while len(stack):
            E = stack.pop()
            if E == "$":
                break
            node = AST_stack.pop()
            if E in self.NoTerminals:
                if token[1] is None:
                    tok = next(
                        filter(lambda x: x.lower() == token[0].lower(), self.Terminals)
                    )
                else:
                    tok = token[1]

                next_rule = self.LL1_map.get((E, tok), None)
                stack.extend(next_rule[::-1])

                children = [AST(ne, None, node) for ne in next_rule]
                node.set_children(children)
                AST_stack.extend(children[::-1])

            elif E in self.Terminals:
                if E == self.epsilon:
                    continue
                if token[1] is not None:
                    assert token[1].lower() == E.lower()
                    node.set_name(token[0])
                else:
                    assert token[0].lower() == E.lower()

                token = token_stream.get_next_token()
            else:
                raise ValueError("snl error")
        return root


def check_LL1(predicts):
    for _, value in predicts.items():
        for i in range(len(value) - 1):
            for j in range(i + 1, len(value)):
                if value[i] & value[j]:
                    return False
    return True


class RecursiveDecentGrammer:
    def __init__(self, grammer_path):
        NoTerminals, Terminals = set(), set()
        rules, S = [], None
        with open(grammer_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                ts = re.findall(r"([a-zA-Z]+) *::=(.*)", line)
                rules.append((ts[0][0], ts[0][1].strip()))
        for r in rules:
            if S is None:
                S = r[0]
            NoTerminals.add(r[0])
        for r in rules:
            V = r[1].replace("|", "").split()
            for v in V:
                if v not in NoTerminals:
                    Terminals.add(v)
        G = {}
        for r in rules:
            if G.get(r[0], None) is None:
                G[r[0]] = []
            for g in r[1].split("|"):
                G[r[0]].append(g.strip().split())

        self.NoTerminals = NoTerminals
        self.Terminals = Terminals
        self.G = G
        self.S = S
        self.epsilon = "ε"

    def parse(self, token_stream: list[Tuple[str, str]]):
        self.init_state()
        token_idx = 0
        back_trace = False
        while self.stack:
            if back_trace:
                self.restore_state()
                E, r_idx, token_idx, node = self.stack.pop()
                back_trace = False
            else:
                E, r_idx, _, node = self.stack.pop()

            if E == "$":
                break
            if r_idx is None:
                tok = self.get_token(token_stream[token_idx])
                if E == self.epsilon:
                    node.set_name(self.epsilon)
                    continue
                elif E != tok[1]:
                    back_trace = True
                    continue
                node.set_name(tok[0])
                token_idx += 1
            else:
                if r_idx + 1 < len(self.G[E]):
                    self.save_state(E, r_idx + 1, token_idx, node)
                rule = self.G[E][r_idx]
                children = []
                for r in rule[::-1]:
                    child = AST(r, None, node)
                    if r in self.Terminals:
                        self.stack.append((r, None, None, child))
                    else:
                        self.stack.append((r, 0, None, child))
                    children.insert(0, child)

                node.set_children(children)
        return self.root

    def init_state(self):
        self.stack_states = []
        self.root = AST(self.S, None, None)
        self.stack = [("$", None, None, None), (self.S, 0, 0, self.root)]

    def restore_state(self):
        self.stack = self.stack_states.pop()

    def save_state(self, E, r_idx, token_idx, node):
        self.stack_states.append(self.stack + [(E, r_idx, token_idx, node)])

    def get_token(self, token):
        if token[1] is not None:
            return token
        else:
            tok = next(filter(lambda x: x.lower() == token[0].lower(), self.Terminals))
            return (token[0], tok)


def makechanging():
    length = None

    def changing(d: dict) -> bool:
        nonlocal length
        if not length:
            length = dict()
            for k in d.keys():
                length[k] = len(d[k])
            return True

        isChanging = False
        for k in d.keys():
            isChanging |= length[k] != len(d[k])
            length[k] = len(d[k])
        return isChanging

    return changing


class AST:
    def __init__(self, attr: str, name: str = None, fa=None):
        self.__name = name
        self.__attr = attr
        self.__children = []
        self.fa = fa

    @property
    def children(self):
        return self.__children

    @property
    def name(self):
        return self.__name

    @property
    def attr(self):
        return self.__attr

    def set_children(self, children):
        self.__children = children

    def set_name(self, name):
        self.__name = name


def print_file_tree(root, prefix=""):
    if root.fa is None:
        connector = "└── "
        print(prefix + connector + f"({root.attr,root.name})")
        prefix = prefix + "    "
    for i, child in enumerate(root.children):

        if i == len(root.children) - 1:
            connector = "└── "
            new_prefix = prefix + "    "
        else:
            connector = "├── "
            new_prefix = prefix + "│   "

        print(prefix + connector + f"({child.attr,child.name})")
        if child.children:
            print_file_tree(child, new_prefix)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--grammer", type=str)
    parser.add_argument("--program", type=str)
    parser.add_argument("--type", type=str)
    args = parser.parse_args()
    print(args.grammer, args.program, args.type)
    with open(args.program, "r", encoding="UTF-8") as f:
        if args.type == "LL1":
            grammer = LL1Grammer(args.grammer)
            token_stream = TokenStream(f, grammer.Terminals)
            root = grammer.parse(token_stream)
        elif args.type == "RD":
            grammer = RecursiveDecentGrammer(args.grammer)
            token_stream = TokenStream(f, grammer.Terminals)
            tokens = []
            while t := token_stream.get_next_token():
                tokens += [t]
            print(tokens)
            root = grammer.parse(tokens)
            print(root)
    print_file_tree(root)


if __name__ == "__main__":
    main()
