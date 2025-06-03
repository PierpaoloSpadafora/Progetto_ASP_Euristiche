
% ---------------------------------------------------------------------------
% -----------------  Provate e risultati interessanti  ----------------------
% ---------------------------------------------------------------------------

% Euristica #1
% Evita di sovraccaricare singoli chirurghi
#heuristic 
    x(R,P,S,A,O,SH,D,ST) :  #count{R2 : x(R2,_,S,_,_,SH,D,_)} > 2. 
[-12@6,sign]


% Euristica #11
% Assegna prima interventi urgenti ai chirurghi con meno tempo disponibile
#heuristic 
    x(R,P,S,A,O,SH,D,ST) :  surgeon(S,_,SH), 
                            #count{R2 : x(R2,_,S,_,_,SH,D,_)} == 0,
                            registration(R,1,_,_,_,_,_). 
[15@8,true]


% Euristica #2
% Penalizza interventi corti negli slot iniziali se ci sono lunghi in attesa
#heuristic 
    x(R,P,S,A,O,SH,D,ST) :  registration(R,_,DUR,_,_,_,_), 
                            DUR <= 2, ST <= 2, 
                            registration(R2,_,DUR2,_,_,_,_), 
                            DUR2 >= 4, 
                            not x(R2,_,_,_,_,_,_,_). 
[-8@5,sign]


% Euristica #3
% Favorisci chirurghi sottoutilizzati
#heuristic 
    x(R,P,S,A,O,SH,D,ST) :  surgeon(S,_,SH), 
                            #count{R2 : x(R2,_,S,_,_,SH,D,_)} == 0. 
[15@6,true]


% Euristica #4
% Raggruppa interventi consecutivi
#heuristic 
    x(R,P,S,A,O,SH,D,ST) :  x(R2,_,S,_,_,SH,D,ST2), 
                            R != R2, |ST - ST2| == 1. 
[8@5,true]


% Euristica #10
% Prova ad assegnare prima interventi brevi ai chirurghi con meno tempo disponibile
#heuristic
x(R,P,S,A,O,SH,D,ST) :  surgeon(S,_,SH),
#count{R2 : x(R2,_,S,_,_,SH,D,_)} == 0,
registration(R,_,DUR,_,_,_,_), DUR < 4.
[10@6,true]




% ---------------------------------------------------------------------------
% -----------------  Provate e non migliorano affatto  ----------------------
% ---------------------------------------------------------------------------


% Euristica #5
% Favorisce specialità con più richieste urgenti
#heuristic x(R,1,S,A,O,SH,D,ST) : registration(R,1,_,_,SPEC,_,_), #count{R2 : registration(R2,1,_,_,SPEC,_,_)} >= 3. [15@8,true]


% Euristica #6
% Priorità + Durata + Posizione temporale
#heuristic x(R,P,S,A,O,SH,D,ST) : registration(R,P,DUR,_,_,_,_), SCORE = (4-P)*10 + DUR*2 + (6-ST), SCORE >= 25. [SCORE@7,true]


% Euristica #7
% Mantieni un po' di tempo libero alla fine 
#heuristic x(R,P,S,A,O,SH,D,ST) : registration(R,_,DUR,_,_,_,_), ST + DUR > 4. [-8@4,sign]


% Euristica #8
% Ritarda le priorità 3 finché le priorità 1-2 non sono sistemate
#heuristic x(R,3,S,A,O,SH,D,ST) : #count{R1 : x(R1,1,_,_,_,_,_,_)} < TOTP1, totRegsP1(TOTP1). [-30@9,sign]




