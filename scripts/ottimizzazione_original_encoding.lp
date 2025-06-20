% per leggere direttamente il codice dell'ORS
% ANDARE A RIGO 65


% mini esempio per spiegarmi meglio

numero(1..3).


% questa regola genera 1 choice rule per ogni numero(N)
% la choice rule può scegleire 0 o 1 indovina(D) per ogni numero(N)
0 { indovina(N) } 1 :- numero(N).

        % Answer: 1 (Time: 0.000s)
        % 

        % Answer: 2 (Time: 0.000s)
        % indovina(2)

        % Answer: 3 (Time: 0.000s)
        % indovina(3)

        % Answer: 4 (Time: 0.000s)
        % indovina(2) indovina(3)

        % Answer: 5 (Time: 0.000s)
        % indovina(1)

        % Answer: 6 (Time: 0.000s)
        % indovina(1) indovina(3)

        % Answer: 7 (Time: 0.000s)
        % indovina(1) indovina(2)

        % Answer: 8 (Time: 0.000s)
        % indovina(1) indovina(2) indovina(3)


% richiede l'aggiunta di un vincolo per indovinare un solo numero 
:- #count {N : indovina(N)} > 1.
        % Answer: 1 (Time: 0.000s)
        % 

        % Answer: 2 (Time: 0.000s)
        % indovina(3)

        % Answer: 3 (Time: 0.000s)
        % indovina(2)

        % Answer: 4 (Time: 0.000s)
        % indovina(1)


% questa regola invece genera una sola choice rule
% sceglie MASSIMO 1 indovina(N) tra tutti i numero(N)
% genera lo stesso AS ma con meno regole groundate 
% e meno tempo per scartare AS non validi
0 { indovina(N) : numero(N) } 1.






% Esempio con l'ORS

% questa è la regola originale....
% viene creata una choice rule per ogni combinazione di atomi
0 {
    x(REGID,PRI,SRGID,ANID,OPROOMID,SHIFT,DAY,ST): 
        ST+SURGDUR <= shift_duration
    } 
1 :- 
registration(REGID,PRI,SURGDUR,_,SPECID,_,_),
mss(OPROOMID,SHIFT,SPECID,DAY),
surgeon(SRGID,SPECID,SHIFT),
an(ANID,SPECID,SHIFT),
time(SHIFT,ST).

% se vengono generati AS validi con più atomi x con lo stesso REGID vengono 
% eliminati da questo vincolo
:- registration(REGID,_,_,_,_,_,_), #count{PRI,SRGID,ANID,OPROOMID,SHIFT,DAY,ST : x(REGID,PRI,SRGID,ANID,OPROOMID,SHIFT,DAY,ST)} > 1.

% però comunque il tempo per il grounding e poi per eliminare gli AS è stato sprecato

% questa regola invece forza a generare una sola choice rule per registration
0 { 
    x(REGID,PRI,SRGID,ANID,OPROOM,SHIFT,DAY,ST) :
      registration(REGID,PRI,SURGDUR,_,SPECID,_,_),
      mss(OPROOM,SHIFT,SPECID,DAY),
      surgeon(SRGID,SPECID,SHIFT),
      an(ANID,SPECID,SHIFT),
      time(SHIFT,ST),
      ST+SURGDUR <= shift_duration
} 1 :- registration(REGID,_,_,_,_,_,_).




% il codice dentro "original_encoding.lp" dovrebbe essere cambiato in questo modo
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