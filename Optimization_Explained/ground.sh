clingo --mode=gringo 1.lp > out1
clingo --mode=gringo 2.lp > out2
clingo --mode=gringo 3.lp > out3

grep -o 'scegli([^)]*)' out1 | sed 's/scegli(//;s/)//' | sort -t',' -k1,1 -k2,2 -k3,3 -k4,4 | sed 's/^/scegli(/;s/$/)/' > out1
grep -o 'scegli([^)]*)' out2 | sed 's/scegli(//;s/)//' | sort -t',' -k1,1 -k2,2 -k3,3 -k4,4 | sed 's/^/scegli(/;s/$/)/' > out2
grep -o 'scegli([^)]*)' out3 | sed 's/scegli(//;s/)//' | sort -t',' -k1,1 -k2,2 -k3,3 -k4,4 | sed 's/^/scegli(/;s/$/)/' > out3

diff out1 out2
diff out1 out3
diff out2 out3









clingo --mode=gringo original_to_ground.lp > original_output
clingo --mode=gringo optimized_to_ground.lp > optimized_output


grep -o 'x([^)]*)' original \
  | sed -E 's/^x\(|\)$//g' \
  | LC_ALL=C sort -t, -n \
      -k1,1 -k2,2 -k3,3 -k4,4 \
      -k5,5 -k6,6 -k7,7 -k8,8 \
  | sed 's/^/x(/;s/$/)/' \
  > original_sorted
mv original_sorted original_grounded

grep -o 'x([^)]*)' optimized_output \
| sed -E 's/^x\(|\)$//g' \
| LC_ALL=C sort -t, -n \
    -k1,1 -k2,2 -k3,3 -k4,4 \
    -k5,5 -k6,6 -k7,7 -k8,8 \
| sed 's/^/x(/;s/$/)/' \
> optimized_sorted
mv optimized_sorted optimized_output

diff original_output optimized_output