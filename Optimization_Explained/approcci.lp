cibo(latte). cibo(cetrioli).
cibo(banana). cibo(parmigiano).
cibo(pane). cibo(lattuga). 

proteina(tonno). proteina(vitello). 
proteina(salmone). proteina(maiale). 

% inizio approccio 1:

0 { scegli(P1,F2,F3,F4) : 
} 1 :-  
proteina(P1), 
cibo(F2), 
cibo(F3), 
cibo(F4).

:- #count {P1 : scegli(P1,_,_,_)} > 1.

%fine approccio 1


% inizio approccio 2:

1 { scegli(P1,F2,F3,F4) : 
            proteina(P1),
            cibo(F2), 
            cibo(F3), 
            cibo(F4)
} 1.

%fine approccio 2

% inizio approccio 3:
1 { scegli(P1,F2,F3,F4) : 
            cibo(F2), 
            cibo(F3), 
            cibo(F4)
} 1 :-
proteina(P1). 
% fine approccio 3

