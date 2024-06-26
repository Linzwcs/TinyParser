# TinyParser

TinyParser is a demo implementing LL(1) grammar analysis and recursive descent parsing. The test sample is SNL, a toy language. Given the grammars and sample programs, the related LL(1) grammar table and abstract syntax tree are shown.

## Execute

```
python .\Grammer.py \
        --grammer .\rule\snl.txt \
        --program .\sample\snl_program.txt \
        --type LL1
```

You can choose other grammar rules, but if you want to execute an LL1 parser, you must ensure that the given grammar rules fit the LL1 criteria or you can choose the recursive decent parser.

## Result

![alt text](./fig/image.png)
