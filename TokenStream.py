from io import TextIOWrapper, TextIOBase
import re
from enum import Enum
from typing import Union


def valid(char):
    return True


class TokenStream:
    def __init__(self, fp: TextIOBase, terminals: list[str]) -> None:
        self.fp = fp
        self.cur_char = " "
        self.terminals = terminals

    def get_next_token(self):
        while self.cur_char.isspace():

            self.cur_char = self.fp.read(1)
            if self.cur_char == "":
                return ""
        if self.cur_char.isalpha():
            cur_tok = self.cur_char
            self.cur_char = self.fp.read(1)
            while self.cur_char.isalnum():
                cur_tok += self.cur_char
                self.cur_char = self.fp.read(1)
            if (
                len(
                    list(filter(lambda x: x.lower() == cur_tok.lower(), self.terminals))
                )
                > 0
            ):
                return (cur_tok, None)
            return (cur_tok, "ID")
        elif self.cur_char.isdigit():
            cur_tok = self.cur_char
            self.cur_char = self.fp.read(1)
            while self.cur_char.isdigit():
                cur_tok += self.cur_char
                self.cur_char = self.fp.read(1)
            return (int(cur_tok), "INTC")
        elif self.cur_char == ":":
            cur_tok = self.cur_char
            self.cur_char = self.fp.read(1)
            # Exception process
            assert self.cur_char == "="
            cur_tok = ":="
            self.cur_char = self.fp.read(1)
            return (cur_tok, None)
        elif self.cur_char == ".":
            cur_tok = self.cur_char
            self.cur_char = self.fp.read(1)
            if self.cur_char == ".":
                cur_tok = ".."
                # print(">>> ..")
                self.cur_char = self.fp.read(1)
                return (cur_tok, None)
            else:
                # print(">>> .")
                return (cur_tok, None)
        elif valid(self.cur_char):
            cur_tok = self.cur_char
            self.cur_char = self.fp.read(1)
            return (cur_tok, None)
        else:
            assert False
