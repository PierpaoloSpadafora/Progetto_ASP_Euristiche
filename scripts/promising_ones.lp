% Euristica #1
% Evita di sovraccaricare singoli chirurghi
#heuristic 
    x(R,P,S,A,O,SH,D,ST) :  #count{R2 : x(R2,_,S,_,_,SH,D,_)} > 2. 
[12@6,false]
% Tempo medio: 33.3493s
% Modelli usati: 36
% Timeout: 0/1
% Miglior costo: [2, 25]

% Euristica #11
% Assegna prima interventi urgenti ai chirurghi con meno tempo disponibile
#heuristic 
    x(R,P,S,A,O,SH,D,ST) :  surgeon(S,_,SH), 
                            #count{R2 : x(R2,_,S,_,_,SH,D,_)} == 0,
                            registration(R,1,_,_,_,_,_). 
[-15@8,false]
% Tempo medio: 47.0926s
% Modelli usati: 45
% Timeout: 0/1
% Miglior costo: [2, 25]

% Euristica #2
% Penalizza interventi corti negli slot iniziali se ci sono lunghi in attesa
#heuristic 
    x(R,P,S,A,O,SH,D,ST) :  registration(R,_,DUR,_,_,_,_), 
                            DUR <= 2, ST <= 2, 
                            registration(R2,_,DUR2,_,_,_,_), 
                            DUR2 >= 4, 
                            not x(R2,_,_,_,_,_,_,_). 
[8@5,false]
% Tempo medio: 67.7370s
% Modelli usati: 62
% Timeout: 0/1
% Miglior costo: [2, 25]

% Euristica #3
% Favorisci chirurghi sottoutilizzati
#heuristic 
    x(R,P,S,A,O,SH,D,ST) :  surgeon(S,_,SH), 
                            #count{R2 : x(R2,_,S,_,_,SH,D,_)} == 0. 
[-15@6,false]
% Tempo medio: 78.1178s
% Modelli usati: 43
% Timeout: 0/1
% Miglior costo: [2, 25]

% Euristica #4
% Raggruppa interventi consecutivi
#heuristic 
    x(R,P,S,A,O,SH,D,ST) :  x(R2,_,S,_,_,SH,D,ST2), 
                            R != R2, |ST - ST2| == 1. 
[-8@5,false]
% Tempo medio: 82.4064s
% Modelli usati: 41
% Timeout: 0/1
% Miglior costo: [2, 25]

% Euristica #10
% Prova ad assegnare prima interventi brevi ai chirurghi con meno tempo disponibile
#heuristic
x(R,P,S,A,O,SH,D,ST) :  surgeon(S,_,SH),
#count{R2 : x(R2,_,S,_,_,SH,D,_)} == 0,
registration(R,_,DUR,_,_,_,_), DUR < 4.
[-10@6,false]
% Tempo medio: 81.3891s
% Modelli usati: 43
% Timeout: 0/1
% Miglior costo: [2, 25]