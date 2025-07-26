% Definizione dei cibi
% carne o pesce costano tanto, fingiamo di dover scegliere un pasto e limitarci ad una sola fonte proteica costosa
cibo(latte). cibo(cetrioli).
cibo(banana). cibo(parmigiano).
cibo(pane). cibo(lattuga). 

proteina(tonno). proteina(vitello). 
proteina(salmone). proteina(maiale). 

% Scegli 1 proteina e 3 cibi
0 { scegli(P1,F2,F3,F4) : 
} 1 :-  
proteina(P1), 
cibo(F2), 
cibo(F3), 
cibo(F4).

% elimina le selezioni con più di una fonte proteica costosa
:- #count {P1 : scegli(P1,_,_,_)} > 1.


#show scegli/4.
