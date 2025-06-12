% 2 + 3 + 4 + 10


% Euristica #2
#heuristic 
    x(R,P,S,A,O,SH,D,ST) :  registration(R,_,DUR,_,_,_,_), 
                            DUR <= 2, ST <= 2, 
                            registration(R2,_,DUR2,_,_,_,_), 
                            DUR2 >= 4, 
                            not x(R2,_,_,_,_,_,_,_). 
[8@5,false]

% Euristica #3
#heuristic 
    x(R,P,S,A,O,SH,D,ST) :  surgeon(S,_,SH), 
                            #count{R2 : x(R2,_,S,_,_,SH,D,_)} == 0. 
[-15@6,false]

% Euristica #4
#heuristic 
    x(R,P,S,A,O,SH,D,ST) :  x(R2,_,S,_,_,SH,D,ST2), 
                            R != R2, |ST - ST2| == 1. 
[-8@5,false]

% Euristica #10
#heuristic
x(R,P,S,A,O,SH,D,ST) :  surgeon(S,_,SH),
#count{R2 : x(R2,_,S,_,_,SH,D,_)} == 0,
registration(R,_,DUR,_,_,_,_), DUR < 4.
[-10@6,false]