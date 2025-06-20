% 3 + 11

% Euristica #3
#heuristic 
    x(R,P,S,A,O,SH,D,ST) :  surgeon(S,_,SH), 
                            #count{R2 : x(R2,_,S,_,_,SH,D,_)} == 0. 
[-15@6,false]

% Euristica #11
#heuristic 
    x(R,P,S,A,O,SH,D,ST) :  surgeon(S,_,SH), 
                            #count{R2 : x(R2,_,S,_,_,SH,D,_)} == 0,
                            registration(R,1,_,_,_,_,_). 
[-15@8,false]