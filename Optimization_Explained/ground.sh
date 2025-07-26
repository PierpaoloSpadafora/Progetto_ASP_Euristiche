# ----------------------------- APPROCCIO FACILE -----------------------------

cd facile
clingo --mode=gringo 1.lp > out1
clingo --mode=gringo 2.lp > out2

grep -o 'scegli([^)]*)' out1 | sed 's/scegli(//;s/)//' | sort -t',' -k1,1 -k2,2 -k3,3 -k4,4 | sed 's/^/scegli(/;s/$/)/' > out1
grep -o 'scegli([^)]*)' out2 | sed 's/scegli(//;s/)//' | sort -t',' -k1,1 -k2,2 -k3,3 -k4,4 | sed 's/^/scegli(/;s/$/)/' > out2
clear

diff out1 out2


# --- STATS ---


echo "-------------- FIRST APPROACH ---------------"
clingo 1.lp --stats=2 | grep -E 'Rules|Atoms|Bodies|Choices|Conflicts|Constraints|Equivalences|Variables'
echo "-------------- SECOND APPROACH ---------------"
clingo 2.lp --stats=2 | grep -E 'Rules|Atoms|Bodies|Choices|Conflicts|Constraints|Equivalences|Variables'
cd ..




# ----------------------------- APPROCCIO REALE -----------------------------



cd reale
clingo --mode=gringo input_to_ground.lp original_to_ground.lp > original_grounded
clingo --mode=gringo input_to_ground.lp optimized_to_ground.lp > optimized_grounded


grep -o 'x([^)]*)' original_grounded \
  | sed -E 's/^x\(|\)$//g' \
  | LC_ALL=C sort -t, -n \
      -k1,1 -k2,2 -k3,3 -k4,4 \
      -k5,5 -k6,6 -k7,7 -k8,8 \
  | sed 's/^/x(/;s/$/)/' \
  > original_sorted

mv original_sorted original_grounded

grep -o 'x([^)]*)' optimized_grounded \
| sed -E 's/^x\(|\)$//g' \
| LC_ALL=C sort -t, -n \
    -k1,1 -k2,2 -k3,3 -k4,4 \
    -k5,5 -k6,6 -k7,7 -k8,8 \
| sed 's/^/x(/;s/$/)/' \
> optimized_sorted
mv optimized_sorted optimized_grounded


clear


diff original_grounded optimized_grounded


# --- STATS ---

echo "-----------------------------"
clingo input_to_ground.lp original_to_ground.lp --stats=2 | grep -E 'Rules|Atoms|Bodies|Choices|Conflicts|Constraints|Equivalences|Variables'
echo "-----------------------------"
clingo input_to_ground.lp optimized_to_ground.lp --stats=2 | grep -E 'Rules|Atoms|Bodies|Choices|Conflicts|Constraints|Equivalences|Variables'
echo "-----------------------------"
cd ..