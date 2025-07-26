
cibo(latte). cibo(cetrioli). cibo(banana). 

%cibo(parmigiano). cibo(pane). cibo(lattuga). 

proteina(tonno). proteina(vitello). 
%proteina(salmone). proteina(maiale). 




% APPROCCIO 1 -> quello della repository di Caruso
% con 2 proteine e 3 cibi ci sono 1^2 * 3^3 = 2*27 = 54 combinazioni
0 { scegli(P1,F2,F3,F4) : 
} 1 :-  
proteina(P1), 
cibo(F2), 
cibo(F3), 
cibo(F4).

:- proteina(P1), #count {P1 : scegli(P1,_,_,_)} > 1.


% APPROCCIO 2 
1 { scegli(P1,F2,F3,F4) : 
            cibo(F2), 
            cibo(F3), 
            cibo(F4)
} 1 :-
proteina(P1). 


% APPROCCIO 3 
1 { scegli(P1,F2,F3,F4) : 
            proteina(P1),
            cibo(F2), 
            cibo(F3), 
            cibo(F4)
} 1.
