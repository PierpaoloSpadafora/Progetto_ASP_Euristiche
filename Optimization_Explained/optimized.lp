% Definizione dei cibi
% carne o pesce costano tanto, fingiamo di dover scegliere un pasto e limitarci ad una sola fonte proteica costosa
cibo(latte). cibo(cetrioli).
cibo(banana). cibo(parmigiano).
cibo(pane). cibo(lattuga). 

proteina(tonno). proteina(vitello). 
proteina(salmone). proteina(maiale). 

% Scegli 1 proteina e 3 cibi
1 { scegli(P1,F2,F3,F4) : 
            proteina(P1),
            cibo(F2), 
            cibo(F3), 
            cibo(F4)
} 1.


#show scegli/4.
