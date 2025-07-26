# ----------------------------- APPROCCIO FACILE -----------------------------

cd facile
clingo --mode=gringo 1.lp > out1
clingo --mode=gringo 2.lp > out2

grep -o 'scegli([^)]*)' out1 | sed 's/scegli(//;s/)//' | sort -t',' -k1,1 -k2,2 -k3,3 -k4,4 | sed 's/^/scegli(/;s/$/)/' > out1
grep -o 'scegli([^)]*)' out2 | sed 's/scegli(//;s/)//' | sort -t',' -k1,1 -k2,2 -k3,3 -k4,4 | sed 's/^/scegli(/;s/$/)/' > out2
clear

diff out1 out2

cd ..


# --- STATS ---


echo "-------------- FIRST APPROACH ---------------"
clingo 1.lp --stats=2 | grep -E 'Rules|Atoms|Bodies|Choices|Conflicts|Constraints|Equivalences|Variables'
echo "-------------- SECOND APPROACH ---------------"
clingo 2.lp --stats=2 | grep -E 'Rules|Atoms|Bodies|Choices|Conflicts|Constraints|Equivalences|Variables'





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
cd ..


# --- STATS ---

echo "-----------------------------"
clingo input_to_ground.lp original_to_ground.lp --stats=2 | grep -E 'Rules|Atoms|Bodies|Choices|Conflicts|Constraints|Equivalences|Variables'
echo "-----------------------------"
clingo input_to_ground.lp optimized_to_ground.lp --stats=2 | grep -E 'Rules|Atoms|Bodies|Choices|Conflicts|Constraints|Equivalences|Variables'
echo "-----------------------------"













% -----------------------------

Choices      : 2981024 
Conflicts    : 439385   (Analyzed: 439385)
Rules        : 120020   (Original: 119836)
Atoms        : 31967   
Bodies       : 15212    (Original: 15086)
Equivalences : 9327     (Atom=Atom: 56 Body=Body: 0 Other: 9271)
Variables    : 15233    (Eliminated:    0 Frozen: 15233)
Constraints  : 93514    (Binary:  92.8% Ternary:   0.3% Other:   6.9%)

% -----------------------------

Choices      : 2977773 
Conflicts    : 438173   (Analyzed: 438173)
Rules        : 95586    (Original: 95402)
Atoms        : 16507   
Bodies       : 15282    (Original: 15156)
Equivalences : 9536     (Atom=Atom: 125 Body=Body: 70 Other: 9341)
Variables    : 15233    (Eliminated:    0 Frozen: 15233)
Constraints  : 93514    (Binary:  92.8% Ternary:   0.3% Other:   6.9%)

% -----------------------------




% --------Input 2 day 5----------------
Choices      : 7623032 
Conflicts    : 28803    (Analyzed: 28803)
Rules        : 4199764  (Original: 4169072)
Atoms        : 800599   (Original: 785719 Auxiliary: 14880)
Bodies       : 973898   (Original: 951732)
Equivalences : 247963   (Atom=Atom: 268 Body=Body: 0 Other: 247695)
Variables    : 404881   (Eliminated:    0 Frozen: 390001)
Constraints  : 2963472  (Binary:  94.6% Ternary:   0.4% Other:   5.0%)

% -----------------------------

*** ERROR: (pyclingo): solving stopped by signal
*** Info : (pyclingo): Shutdown completed in 0.007 seconds
Choices      : 8167644 
Conflicts    : 23865    (Analyzed: 23865)
Rules        : 3541058  (Original: 3540222)
Atoms        : 396419  
Bodies       : 952648   (Original: 952082)
Equivalences : 242292   (Atom=Atom: 617 Body=Body: 350 Other: 241325)
Variables    : 390001   (Eliminated:    0 Frozen: 390001)
Constraints  : 2918736  (Binary:  94.8% Ternary:   0.2% Other:   5.0%)

% -----------------------------