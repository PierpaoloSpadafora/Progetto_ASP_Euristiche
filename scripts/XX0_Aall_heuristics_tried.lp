
% ---------------------------------------------------------------------------
% -----------------  Provate e risultati interessanti  ----------------------
% ---------------------------------------------------------------------------

% Euristica #1
%   Penalizza fortemente l'assegnazione di un intervento 
%   quando un chirurgo S ha già più di 2 interventi nello stesso turno SH del giorno D.
#heuristic 
    x(R,P,S,A,O,SH,D,ST) :  #count{R2 : x(R2,_,S,_,_,SH,D,_)} > 2. 
[-12@6,sign]


% Euristica #11
%   Favorisce massimamente (priorità più alta di tutti) l'assegnazione di 
%   interventi di priorità 1 a chirurghi liberi.
#heuristic 
    x(R,P,S,A,O,SH,D,ST) :  surgeon(S,_,SH), 
                            #count{R2 : x(R2,_,S,_,_,SH,D,_)} == 0,
                            registration(R,1,_,_,_,_,_). 
[15@8,true]


% Euristica #2
%   Penalizza l'assegnazione di un intervento breve (≤2 ore) nelle prime ore del turno (ST≤2) 
%   quando esiste un intervento lungo (≥4 ore) non ancora schedulato.
#heuristic 
    x(R,P,S,A,O,SH,D,ST) :  registration(R,_,DUR,_,_,_,_), 
                            DUR <= 2, ST <= 2, 
                            registration(R2,_,DUR2,_,_,_,_), 
                            DUR2 >= 4, 
                            not x(R2,_,_,_,_,_,_,_). 
[-8@5,sign]


% Euristica #3
%   Favorisce fortemente l'assegnazione di interventi a chirurghi S 
%   che non hanno ancora interventi nel turno SH del giorno D.
#heuristic 
    x(R,P,S,A,O,SH,D,ST) :  surgeon(S,_,SH), 
                            #count{R2 : x(R2,_,S,_,_,SH,D,_)} == 0. 
[15@6,true]


% Euristica #4
%   Favorisce l'assegnazione di interventi consecutivi 
%   (slot temporali adiacenti) allo stesso chirurgo S.
#heuristic 
    x(R,P,S,A,O,SH,D,ST) :  x(R2,_,S,_,_,SH,D,ST2), 
                            R != R2, |ST - ST2| == 1. 
[8@5,true]


% Euristica #10
%   Favorisce l'assegnazione di interventi di durata media (<4 ore) 
%   a chirurghi ancora liberi nel turno.
#heuristic
    x(R,P,S,A,O,SH,D,ST) :  surgeon(S,_,SH),
                            #count{R2 : x(R2,_,S,_,_,SH,D,_)} == 0,
                            registration(R,_,DUR,_,_,_,_), DUR < 4.
[10@6,true]



% ---------------------------------------------------------------------------
% -----------------  Provate e non migliorano affatto  ----------------------
% ---------------------------------------------------------------------------


% Euristica #6
%   Calcola un punteggio composito che combina:
%   (4-P)*10: Priorità inversa (priorità 1 = 30 punti, priorità 3 = 10 punti)
%   DUR*2: Durata dell'intervento moltiplicata per 2
%   (6-ST): Slot temporale inverso (slot iniziali valgono di più)

%   unica formula per dare priorità agli interventi più urgenti, 
%   a parità di urgenza quelli più lunghi 
%   a parità di urgenza e durata quelli più vicini all'inizio del turno.

%   Fa abbastanza schifo :(
#heuristic 
    x(R,P,S,A,O,SH,D,ST) : registration(R,P,DUR,_,_,_,_), 
                            SCORE = (4-P)*10 + DUR*2 + (6-ST), 
                            SCORE >= 25. 
[SCORE@7,true]


% Euristica #8
%   Penalizza fortemente l'assegnazione di qualsiasi intervento di priorità 3  
%   finché non sono stati schedulati tutti gli interventi di priorità 1.
#heuristic 
    x(R,3,S,A,O,SH,D,ST) : #count{R1 : x(R1,1,_,_,_,_,_,_)} < TOTP1, 
                            totRegsP1(TOTP1). 
[-30@9,sign]




