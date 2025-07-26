
0 { 
    x(REGID,PRI,SRGID,ANID,OPROOM,SHIFT,DAY,ST) :
      registration(REGID,PRI,SURGDUR,_,SPECID,_,_),
      mss(OPROOM,SHIFT,SPECID,DAY),
      surgeon(SRGID,SPECID,SHIFT),
      an(ANID,SPECID,SHIFT),
      time(SHIFT,ST),
      ST+SURGDUR <= shift_duration
} 1 :- registration(REGID,_,_,_,_,_,_).

:- #count {REGID: x(REGID,_,SRGID,_,_,SHIFT,_,ST)} > 1, surgeon(SRGID,_,SHIFT), time(SHIFT,ST).
:- #count {REGID: x(REGID,_,SRGID,_,_,SHIFT,_,ST), registration(REGID,_,SURGDUR,_,_,_,_), T>= ST, T<ST+SURGDUR} > 1, surgeon(SRGID,_,SHIFT), time(SHIFT,T).
:- #count {REGID: x(REGID,_,_,ANID,_,SHIFT,_,ST)} > 1, an(ANID,_,SHIFT), time(SHIFT,ST).
:- #count {REGID: x(REGID,_,_,ANID,_,SHIFT,_,ST), registration(REGID,_,SURGDUR,_,_,_,_), T>= ST, T<ST+SURGDUR} > 1, an(ANID,_,SHIFT), time(SHIFT,T).
:- #sum {SURGDUR, REGID : x(REGID,_,SRGID,_,_,_,DAY,_),registration(REGID,_,SURGDUR,_,_,_,_) } > sWT, surgeryTime(sWT,SRGID,DAY).
:- #sum {SURGDUR, REGID : x(REGID,_,_,ANID,_,_,DAY,_),registration(REGID,_,SURGDUR,_,_,_,_) } > aWT, anaesthetistWT(aWT,ANID,DAY).
:- x(REGID1,_,_,_,OPROOMID,SHIFT,DAY,ST), x(REGID2,_,_,_,OPROOMID,SHIFT,DAY,ST), REGID1 != REGID2.
:- #count {REGID: x(REGID,_,_,_,OPROOMID,SHIFT,_,ST), registration(REGID,_,SURGDUR,_,_,_,_), T>= ST, T<ST+SURGDUR} > 1, mss(OPROOMID,SHIFT,_,_), time(SHIFT,T).
:- #count {REGID: x(REGID,1,_,_,_,_,_,_)} < totRegsP1.
:~ M = #count {REGID: x(REGID,2,_,_,_,_,_,_)}, N = totRegsP2 - M. [N@3]
:~ M = #count {REGID: x(REGID,3,_,_,_,_,_,_)}, N = totRegsP3 - M. [N@2]
#show x/8.



#heuristic x(R,P,S,A,O,SH,D,ST) :  #count{R2 : x(R2,_,S,_,_,SH,D,_)} > 2. [12@6,false]


