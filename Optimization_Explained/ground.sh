clingo --mode=gringo 1.lp > out1
clingo --mode=gringo 2.lp > out2
clingo --mode=gringo 3.lp > out3

grep -o 'scegli([^)]*)' out1 
grep -o 'scegli([^)]*)' out2 
grep -o 'scegli([^)]*)' out3 